import json
import time

from nicegui import run, ui

from agents.coding.llm import LLMClient
from agents.coding.loop import agent_loop, run_one_step, run_snippet
from agents.coding.sandbox import DEFAULT_TIMEOUT
from agents.coding.tasks import CUSTOM, TASKS
from ui.common import format_event

SNIPPETS = {
    "1. Prints text (success)": "print('hello from the sandbox')\nprint(sum(range(10)))\n",
    "2. Raises an error (traceback)": "def divide(a, b):\n    return a / b\n\nprint(divide(10, 2))\nprint(divide(1, 0))\n",
    "3. Infinite loop (timeout kills it)": "import time\nprint('working...', flush=True)\nwhile True:\n    time.sleep(0.1)\n",
    "4. Looks for API keys (should find none)": ("import os\nhidden = [k for k in os.environ if k.endswith(('_KEY', '_TOKEN'))]\n"
                                                 "print('secret-looking variables visible to this code:', hidden)\n"),
    "5. Writes and reads a file": "open('note.txt', 'w').write('saved by a snippet')\nprint(open('note.txt').read())\n",
}
LLM_COLOR = {"online": "positive", "loading": "orange", "offline": "negative", "checking": "grey"}
FIRST_TASK = next(iter(TASKS))


def problem_line(ev: dict) -> str:
    """The most useful single line of a failed observation."""
    lines = [l for l in ev["text"].strip().splitlines() if l.strip()] or ["(no message)"]
    return (lines[-1] if ev["text"].startswith(("EXIT", "TIMEOUT")) else lines[0])[:140]


def action_summary(ev: dict) -> str:
    a = ev["args"]
    return (f"{ev['tool']} {a.get('path') or a.get('command') or a.get('query') or a.get('url') or (a.get('code') or '')[:30]}"
            .strip())


# every provider the demo switches can turn off (search chain first, then the page chain)
SEARCH_CHAIN = ("tavily", "exa", "serper", "ddgs")
FETCH_CHAIN = ("trafilatura", "jina", "tavily_extract")


def web_line(ev: dict):
    """One readable line for the 'Web activity' log, or None for events that are not web events."""
    k = ev["kind"]
    who = ev.get("agent", "")
    if k == "web_start":
        what = ev.get("query") or ev.get("url")
        return f"[{who}] {ev['tool']}: {what}" + ("  (cache skipped)" if ev.get("fresh") else "")
    if k == "cache_hit":
        return f"    cache hit ({ev['provider']}, saved {ev['age_s']} s ago): no network call"
    if k == "provider_try":
        return f"    trying {ev['provider']} (attempt {ev['attempt']})"
    if k == "retry":
        return f"    {ev['provider']} attempt {ev['attempt']} failed: {ev['reason']}; waiting {ev['wait_s']} s, then again"
    if k == "fallback":
        nxt = f"falling back to {ev['next']}" if ev.get("next") else "no provider left"
        return f"    {ev['provider']} {'skipped' if ev.get('skipped') else 'FAILED'}: {ev['reason']} -> {nxt}"
    if k == "stale_cache":
        return f"    every provider failed -> using an old saved copy ({ev['age_s']} s old)"
    if k == "web_result":
        size = f"{ev['results']} results" if "results" in ev else f"{ev['chars']} characters"
        return f"    OK via {ev['provider']}: {size}, {ev['attempts']} attempt(s), {ev['seconds']} s"
    if k == "web_failed":
        return "    FAILED: " + "; ".join(f"{n}: {r}" for n, r in ev["trail"])
    if k == "blocked" and ev.get("tool") == "fetch_page":
        return f"    URL refused: {ev['reason']}"
    return None


def build_agent_tab(ctx, pump) -> None:
    orch = ctx.orchestrator
    client = LLMClient(ctx.settings)            # no bus: health checks are not logged
    max_steps = ctx.settings.max_steps
    pending_fix = {}                            # the last failure that has no fix yet
    approval_rows = {}                          # request id -> {"card", "countdown", "deadline"}

    # ---------------------------------------------------------------- actions
    def start(target, **kwargs) -> None:
        try:
            orch.start("coding", target, **kwargs)
        except RuntimeError as e:
            ui.notify(str(e), type="warning")

    def run_agent() -> None:
        task = (task_box.value or "").strip()
        if not task:
            ui.notify("Type a task first", type="warning")
            return
        preset = TASKS.get(preset_select.value)
        seeded = preset is not None and preset.task == task        # seed files only for an unedited preset
        start(agent_loop, settings=ctx.settings, task=task, approvals=ctx.approvals, web=ctx.web,
              files=dict(preset.files) if seeded else None, protected=preset.protected if seeded else ())

    def run_step() -> None:
        task = (task_box.value or "").strip()
        if task:
            start(run_one_step, settings=ctx.settings, task=task)
        else:
            ui.notify("Type a task first", type="warning")

    def run_code() -> None:
        code = editor.value or ""
        if code.strip():
            start(run_snippet, settings=ctx.settings, code=code)
        else:
            ui.notify("Type some Python first", type="warning")

    def stop() -> None:
        if not orch.stop("coding"):
            ui.notify("Nothing is running", type="info")

    def pick_preset(e) -> None:
        if e.value in TASKS:
            task_box.set_value(TASKS[e.value].task)

    def task_edited(e) -> None:
        preset = TASKS.get(preset_select.value)
        if preset is not None and e.value != preset.task:
            preset_select.set_value(CUSTOM)

    def answer_approval(rid: str, ok: bool) -> None:
        row = approval_rows.get(rid)
        if not ctx.approvals.answer(rid, ok):
            ui.notify("That request is already closed", type="info")
        elif row:
            row["countdown"].set_text("Approved: running..." if ok else "Denied")
            for b in row["buttons"]:
                b.disable()

    def add_approval(ev: dict) -> None:
        rid = ev["id"]
        if rid in approval_rows or not ctx.approvals.is_pending(rid):    # skip requests replayed after a refresh
            return
        with approval_box:
            with ui.card().classes("w-full").style("border: 2px solid orange") as card:
                ui.label(f"Step {ev['step']}: the agent wants to run this shell command").classes("text-sm font-bold")
                ui.label(ev["command"]).classes("font-mono text-sm w-full").style(
                    "white-space: pre-wrap; background: #222; padding: 8px; border-radius: 4px")
                with ui.row().classes("items-center gap-3"):
                    b1 = ui.button("Approve", icon="check", color="positive",
                                   on_click=lambda r=rid: answer_approval(r, True))
                    b2 = ui.button("Deny", icon="close", color="negative",
                                   on_click=lambda r=rid: answer_approval(r, False))
                    countdown = ui.label("").classes("text-xs opacity-70")
        approval_rows[rid] = {"card": card, "countdown": countdown, "buttons": (b1, b2),
                              "deadline": ev["t"] + ev["timeout_s"]}
        approval_hint.set_visibility(False)
        set_status("waiting for your approval", "orange")

    def close_approval(ev: dict) -> None:
        row = approval_rows.pop(ev["id"], None)
        if row:
            approval_box.remove(row["card"])
        if not approval_rows:
            approval_hint.set_visibility(True)
        if not ev["ok"]:
            terminal.push(f"[shell command {ev['result']}: it was NOT run]\n")
        if ev["result"] != "stopped":
            set_status("running", "blue")

    def tick_countdown() -> None:
        for row in approval_rows.values():
            if row["countdown"].text.startswith(("Approved", "Denied")):
                continue
            row["countdown"].set_text(f"auto-deny in {max(0, int(row['deadline'] - time.time()))} s")

    def toggle_provider(name: str, on: bool) -> None:
        (ctx.web.disabled.discard if on else ctx.web.disabled.add)(name)
        ctx.bus.emit("web_switch", provider=name, enabled=on)

    def show_results(out) -> None:
        web_results.clear()
        with web_results:
            if not out.ok:
                ui.label(f"Search failed: {out.error}").classes("text-sm text-red-400")
                return
            ui.label(f"via {out.provider} | {len(out.results)} results | {out.attempts} attempt(s) | {out.seconds} s"
                     + (" | OLD COPY" if out.stale else "")).classes("text-xs opacity-70")
            for r in out.results:
                ui.link(r["title"], r["url"], new_tab=True).classes("text-sm font-bold")
                ui.label(f"{r['url']}  -  {r['snippet']}").classes("text-xs opacity-70")

    async def do_search() -> None:
        q = (web_query.value or "").strip()
        if not q:
            ui.notify("Type a search first", type="warning")
            return
        out = await run.io_bound(lambda: ctx.web.search(q, 5, fresh=not use_cache.value, tags={"agent": "search box"}))
        show_results(out)

    async def do_fetch() -> None:
        url = (web_url.value or "").strip()
        if not url:
            ui.notify("Type a URL first", type="warning")
            return
        out = await run.io_bound(lambda: ctx.web.fetch(url, fresh=not use_cache.value, tags={"agent": "search box"}))
        web_results.clear()
        with web_results:
            if not out.ok:
                ui.label(f"Fetch failed: {out.error}").classes("text-sm text-red-400")
                return
            ui.label(f"{out.title or '(no title)'} | via {out.provider} | {len(out.text)} characters | "
                     f"{out.attempts} attempt(s) | {out.seconds} s" + (" | OLD COPY" if out.stale else "")
                     ).classes("text-sm font-bold")
            ui.label(out.text[:1500] + ("..." if len(out.text) > 1500 else "")).classes("text-xs").style(
                "white-space: pre-wrap")

    async def refresh_llm() -> None:
        result = await run.io_bound(client.health)       # runs in a thread, the page stays responsive
        state, detail = result if result else ("offline", "no answer")
        llm_chip.set_text(f"model server: {state}")
        llm_chip.props(f"color={LLM_COLOR[state]}")
        llm_detail.set_text(detail)

    # ------------------------------------------------------------------- page
    with ui.card().classes("w-full"):
        with ui.row().classes("items-center gap-3"):
            ui.label("Coding agent").classes("text-lg font-bold")
            llm_chip = ui.chip("model server: checking", icon="memory", color="grey")
            llm_detail = ui.label("").classes("text-xs opacity-70")
        ui.label(f"Server address: {ctx.settings.llm_url}").classes("text-xs opacity-70")
        preset_select = ui.select([*TASKS, CUSTOM], value=FIRST_TASK, label="Test task",
                                  on_change=pick_preset).classes("w-96")
        task_box = ui.textarea("Task", value=TASKS[FIRST_TASK].task).props("outlined autogrow").classes("w-full")
        task_box.on_value_change(task_edited)
        with ui.row().classes("items-center gap-2"):
            ui.button("Run agent", icon="play_arrow", on_click=run_agent)
            ui.button("Stop", icon="stop", color="negative", on_click=stop)
            ui.button("Run one step", icon="skip_next", color="grey", on_click=run_step).props("outline")
            ui.button("Clear log", icon="delete", color="grey", on_click=lambda: event_log.clear())
        with ui.row().classes("items-center gap-3 w-full"):
            step_label = ui.label(f"step 0 / {max_steps}").classes("text-sm")
            progress = ui.linear_progress(value=0, show_value=False).classes("w-64")
            status_chip = ui.chip("idle", color="grey")

    with ui.card().classes("w-full"):
        ui.label("Approval queue (shell commands wait here until you click)").classes("font-bold")
        approval_hint = ui.label("No command is waiting for approval.").classes("text-xs opacity-70")
        approval_box = ui.column().classes("w-full gap-2")

    with ui.card().classes("w-full"):
        ui.label("Web tools: search and fetch with cache, retries and a fallback chain").classes("font-bold")
        ui.label("Search: Tavily, then Exa, then Serper, then ddgs.  Fetch: trafilatura, then Jina Reader, "
                 "then Tavily Extract.  Switch a provider off to watch the chain fall through.").classes("text-xs opacity-70")
        with ui.row().classes("items-center gap-4"):
            for name in (*SEARCH_CHAIN, *FETCH_CHAIN):
                ui.switch(name, value=True, on_change=lambda e, n=name: toggle_provider(n, e.value))
        credit_labels = {name: ui.label(line).classes("text-xs") for name, line in ctx.web.usage.lines().items()}
        with ui.row().classes("items-center gap-2 w-full no-wrap"):
            web_query = ui.input("Search the web", placeholder="top fintech startups 2026").props("outlined dense") \
                .classes("grow").on("keydown.enter", do_search)
            ui.button("Search", icon="search", on_click=do_search)
        with ui.row().classes("items-center gap-2 w-full no-wrap"):
            web_url = ui.input("Fetch a page", placeholder="https://en.wikipedia.org/wiki/Financial_technology") \
                .props("outlined dense").classes("grow").on("keydown.enter", do_fetch)
            ui.button("Fetch", icon="download", on_click=do_fetch)
        use_cache = ui.checkbox("Use the cache (untick to force a live call)", value=True)
        web_results = ui.column().classes("w-full gap-1")
        ui.label("Web activity (every cache hit, attempt, retry and fallback)").classes("text-xs font-bold")
        web_log = ui.log(max_lines=200).classes("w-full h-40 font-mono text-xs")

    with ui.row().classes("w-full no-wrap items-stretch"):
        with ui.card().classes("w-1/2"):
            ui.label("Plan").classes("font-bold")
            plan_label = ui.label("-").classes("text-sm").style("white-space: pre-wrap")
        with ui.card().classes("w-1/2"):
            ui.label("Thought").classes("font-bold")
            thought_label = ui.label("-").classes("text-sm")
            usage_label = ui.label("").classes("text-xs opacity-70")
            with ui.row().classes("items-center gap-2"):
                ui.label("Chosen action").classes("font-bold")
                tool_badge = ui.badge("-", color="deep-purple")
            action_code = ui.code("{}", language="json").classes("w-full")

    with ui.row().classes("w-full no-wrap items-stretch"):
        with ui.card().classes("w-1/2"):
            with ui.row().classes("items-center gap-2"):
                ui.label("Code").classes("font-bold")
                code_name = ui.badge("-", color="grey")
            code_view = ui.code("", language="python").classes("w-full")
            code_problem = ui.label("").classes("text-xs text-red-400").style("white-space: pre-wrap")
        with ui.card().classes("w-1/2"):
            ui.label("Failures and fixes").classes("font-bold")
            fixes_log = ui.log(max_lines=100).classes("w-full h-64 font-mono text-xs")

    with ui.card().classes("w-full"):
        ui.label("Output (terminal)").classes("font-bold")
        terminal = ui.log(max_lines=400).classes("w-full h-56 font-mono text-xs bg-black text-green-400")
        ui.button("Clear terminal", icon="delete", color="grey", on_click=lambda: terminal.clear())

    with ui.card().classes("w-full"):
        ui.label("Final answer").classes("font-bold")
        answer_label = ui.label("-").classes("text-sm").style("white-space: pre-wrap")
        answer_note = ui.label("").classes("text-xs opacity-70")

    with ui.card().classes("w-full"):
        ui.label("Event log").classes("font-bold")
        ui.label(f"Every event is also saved to: {ctx.bus.logfile}").classes("text-xs opacity-70")
        event_log = ui.log(max_lines=500).classes("w-full h-80 font-mono text-xs")

    with ui.expansion("Sandbox snippet tester (Step 3)", icon="science").classes("w-full"):
        first = next(iter(SNIPPETS))
        ui.select(list(SNIPPETS), value=first, label="Example",
                  on_change=lambda e: editor.set_value(SNIPPETS[e.value])).classes("w-96")
        editor = ui.codemirror(SNIPPETS[first], language="Python", theme="vscodeDark").classes("w-full h-48")
        with ui.row().classes("items-center gap-2"):
            ui.button("Run snippet", icon="play_arrow", on_click=run_code)
            ui.label(f"Own folder per run, {DEFAULT_TIMEOUT} s limit, keys hidden. Output appears in the terminal above.") \
                .classes("text-xs opacity-70")

    # ------------------------------------------------------- live event panels
    def set_status(text: str, color: str) -> None:
        status_chip.set_text(text)
        status_chip.props(f"color={color}")

    def on_event(ev: dict) -> None:
        event_log.push(format_event(ev))
        kind = ev["kind"]
        line = web_line(ev)
        if line:
            web_log.push(line)
        if kind == "credits" and ev["provider"] in credit_labels:
            credit_labels[ev["provider"]].set_text(ev["line"])
        if kind == "task_started":
            for lab in (thought_label, plan_label, answer_label):
                lab.set_text("-")
            usage_label.set_text("")
            answer_note.set_text("")
            code_view.set_content("")
            code_name.set_text("-")
            code_problem.set_text("")
            tool_badge.set_text("-")
            action_code.set_content("{}")
            fixes_log.clear()
            pending_fix.clear()
            step_label.set_text(f"step 0 / {ev.get('max_steps', max_steps)}")
            progress.set_value(0)
            set_status("running", "blue")
        elif kind == "plan":
            plan_label.set_text(("[new plan] " if ev["reason"] == "replan" else "") + ev["text"])
        elif kind == "thought":
            thought_label.set_text(ev["text"])
        elif kind == "action":
            tool_badge.set_text(ev["tool"])
            shown = {"tool": ev["tool"], "args": {k: (v[:120] + "..." if isinstance(v, str) and len(v) > 120 else v)
                                                  for k, v in ev["args"].items()}}
            action_code.set_content(json.dumps(shown, indent=2))
            step_label.set_text(f"step {ev['step']} / {max_steps}")
            progress.set_value(ev["step"] / max_steps)
            if pending_fix:
                fixes_log.push(f"   -> step {ev['step']}: the model reacts with {action_summary(ev)}")
        elif kind == "observation":
            if not ev["ok"]:
                pending_fix.update(step=ev["step"])
                fixes_log.push(f"step {ev['step']}: {ev['tool']} FAILED: {problem_line(ev)}")
            elif pending_fix and ev["tool"] == "run_python":
                fixes_log.push(f"step {ev['step']}: run_python OK -> the problem from step {pending_fix['step']} is fixed")
                pending_fix.clear()
        elif kind == "finish_refused":
            fixes_log.push(f"   finish refused: {ev['reason']}")
        elif kind == "repeat_guard":
            fixes_log.push("   same action 3 times: the agent must re-plan")
        elif kind == "compact":
            fixes_log.push(f"   history summarised ({ev['why']})")
        elif kind == "llm_call":
            usage_label.set_text(
                f"prompt {ev['prompt_tokens']} tokens | answer {ev['completion_tokens']} tokens | "
                f"{ev['seconds']} s | {ev['tokens_per_second']} tok/s")
        elif kind == "file_written":
            code_name.set_text(ev["path"])
            code_view.set_content(ev["content"])             # any text file: .py, .txt, .json, ...
            code_problem.set_text("" if ev["syntax_ok"] else ev["syntax_error"])
        elif kind == "done":
            answer_label.set_text(ev["answer"])
            notes = [f"finished in {ev['steps']} steps", "last run verified" if ev["verified"] else "NOT verified by a successful run"]
            if not ev["integrity_ok"]:
                notes.append("PROTECTED FILE CHANGED: " + ", ".join(ev["changed_protected"]))
            answer_note.set_text(" | ".join(notes))
            set_status("done" if ev["verified"] and ev["integrity_ok"] else "done, with warnings",
                       "positive" if ev["verified"] and ev["integrity_ok"] else "orange")
        elif kind == "budget_exhausted":
            set_status("step budget used up", "negative")
        elif kind == "stopped":
            set_status("stopped", "warning")
        elif kind == "error":
            thought_label.set_text(f"ERROR: {ev['error']}")
            set_status("failed", "negative")
        elif kind == "terminal_start":
            terminal.push(f"$ {ev['cmd']}    (limit {ev['timeout']:g} s, folder {ev['folder']})")
        elif kind == "terminal":
            for line in (ev["output"].rstrip("\n").split("\n") if ev["output"] else ["(no output)"]):
                terminal.push(line)
            if ev["truncated"]:
                terminal.push(f"[output cut: showing the last {len(ev['output'])} of {ev['total_chars']} characters]")
            if ev["timed_out"]:
                status = f"TIMEOUT after {ev['seconds']} s: process tree killed"
            elif ev["stopped"]:
                status = f"STOPPED after {ev['seconds']} s: process tree killed"
            else:
                status = f"exit code {ev['exit_code']} in {ev['seconds']} s"
            terminal.push(f"[{status}]\n")
        elif kind == "blocked":
            what = f" {ev.get('command') or ev.get('url') or ev.get('path') or ''}".rstrip()
            terminal.push(f"BLOCKED {ev['tool']}{what}: {ev['reason']}")
        elif kind == "approval_request":
            add_approval(ev)
        elif kind == "approval_result":
            close_approval(ev)

    pump.subscribe(on_event)
    ui.timer(3.0, refresh_llm)
    ui.timer(1.0, tick_countdown)