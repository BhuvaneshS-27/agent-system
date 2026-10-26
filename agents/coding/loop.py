"""Coding agent.  run_one_step(): Step 2.  run_snippet(): Step 3.  agent_loop(): Step 4.

agent_loop is  PLAN once  ->  repeat { ACT (one JSON action) -> OBSERVE (run the tool) ->
REFLECT (tell the model what failed) }  ->  stop on finish, budget, repeated action or Stop.
Step 5 adds run_shell (blocklist, then the user must Approve) and a streaming model call
that Stop can cancel in the middle of a reply (LLMStopped). Step 6 adds web_search and
fetch_page (research/web.py: cache, retries, provider fallback chain).
"""
import hashlib
import json
import traceback
from dataclasses import dataclass, field
from typing import Optional

from agents.coding.actions import DEFAULT_ENABLED, WEB_TOOLS, action_schema, validate_action
from agents.coding.llm import LLMClient, LLMError, LLMStopped
from agents.coding.prompts import PLAN_SYSTEM, SUMMARY_REQUEST, SYSTEM_PROMPT, build_system_prompt
from agents.coding.tools import ToolResult, Toolbox, clip
from core.jobs import JobFailed

SUMMARY_EVERY = 5           # steps between history summaries
CONTEXT_LIMIT = 0.65        # also summarise when the last prompt used this share of the context
FAIL_STREAK_REPLAN = 3      # consecutive failures that trigger a new plan
MAX_REPLANS = 2
MAX_FINISH_REFUSALS = 2


# ---------------------------------------------------------------- Step 2 and 3 helpers
def run_one_step(job, bus, settings, task: str) -> None:
    """One LLM call: the model picks the first action for the task. Nothing is executed yet."""
    tags = {"job": job.id, "agent": job.agent, "step": 1}
    bus.emit("task_started", job=job.id, agent=job.agent, task=task, model=settings.model_name)
    client = LLMClient(settings, bus)
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": task}]
    res = client.action(messages, tags=tags)
    bus.emit("thought", **tags, text=res.action.thought)
    bus.emit("action", **tags, tool=res.action.tool, args=res.action.args, attempts=res.attempts)


def run_snippet(job, bus, settings, code: str) -> None:
    """Step 3 test button: run user-typed code in a fresh run folder. No LLM involved.
    Output goes to the terminal panel through the 'terminal' events of the Toolbox."""
    tools = Toolbox.for_job(settings, bus, job)
    bus.emit("snippet_started", job=job.id, agent=job.agent, folder=tools.root.name, chars=len(code))
    result = tools.run_python(code=code)
    bus.emit("snippet_finished", job=job.id, agent=job.agent, ok=result.ok)


# ---------------------------------------------------------------- Step 4: the loop
@dataclass
class RunState:
    base: str = ""                  # first user message: task, files, plan
    plan: str = ""
    fail_streak: int = 0
    replans: int = 0
    repeat_trips: int = 0
    finish_refusals: int = 0
    dirty: bool = False             # a .py file was written after the last successful run
    last_run_ok: Optional[bool] = None
    last_prompt_tokens: int = 0
    recent: list = field(default_factory=list)
    log: list = field(default_factory=list)       # one line per step, for progress notes


def sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finish_problem(state: RunState) -> str:
    """Why 'finish' is too early (empty text = fine). This is code, not the model's opinion."""
    if state.last_run_ok is False:
        return "your last run failed. Fix the error and run again before you finish."
    if state.dirty:
        return "you changed a .py file after the last successful run. Run it again before you finish."
    return ""


def step_line(step: int, tool: str, args: dict, result: ToolResult) -> str:
    target = (args.get("path") or args.get("command") or args.get("query") or args.get("url")
              or (args.get("code") or "")[:30])
    first = (result.text.strip().splitlines() or [""])[-1 if result.text.startswith(("EXIT", "TIMEOUT")) else 0]
    return f"step {step}: {tool} {target} -> " + ("ok" if result.ok else f"ERROR {first[:100]}")


def progress_text(state: RunState, n: int = 8) -> str:
    return "\n".join(state.log[-n:]) or "(nothing yet)"


def make_plan(client, bus, tags, state, head: str, reason: str, problem: str = "") -> None:
    ask = head
    if state.plan:
        ask += f"\n\nOLD PLAN:\n{state.plan}"
    if problem:
        ask += f"\n\nWHAT WENT WRONG:\n{problem}"
    ask += "\n\nWrite the plan now."
    try:
        reply = client.chat([{"role": "system", "content": PLAN_SYSTEM},
                             {"role": "user", "content": ask}], tags=tags, max_tokens=300)
        state.plan = reply.content.strip()[:1200] or "(the model gave no plan)"
    except LLMError as e:
        state.plan = f"(planning failed: {e}) Plan as you go."
    bus.emit("plan", **tags, text=state.plan, reason=reason)


def rebuild(system: str, state: RunState, history: list, note: str) -> list:
    """New short history: the task with the current plan and a progress note, plus the last
    exchange (the model's last action and its observation)."""
    first = f"{state.base}\n\nYOUR PLAN:\n{state.plan}\n\nPROGRESS SO FAR:\n{note}\n\nTake the next action."
    return [{"role": "system", "content": system}, {"role": "user", "content": first}] + history[-2:]


def summarise(client, bus, tags, system, state, history, why: str) -> list:
    note = ""
    try:
        reply = client.chat(history + [{"role": "user", "content": SUMMARY_REQUEST}],
                            tags=tags, max_tokens=150)
        note = reply.content.strip()
    except LLMError:
        pass
    note = note or progress_text(state)             # fallback: our own step log
    bus.emit("compact", **tags, why=why, messages_before=len(history), summary=note)
    return rebuild(system, state, history, note)


def agent_loop(job, bus, settings, task: str, files: Optional[dict] = None, protected=(),
               client=None, enabled=DEFAULT_ENABLED, approvals=None, web=None) -> None:
    """Runs the loop. A Stop during a model call ends it quietly with a 'stopped' event."""
    try:
        _agent_loop(job, bus, settings, task, files, protected, client, enabled, approvals, web)
    except LLMStopped:
        bus.emit("stopped", job=job.id, agent=job.agent, during="model call")


def _agent_loop(job, bus, settings, task, files, protected, client, enabled, approvals, web) -> None:
    client = client or LLMClient(settings, bus)
    if client.stop_event is None:
        client.stop_event = job.stop_event             # streaming calls, so Stop can cancel them
    if approvals is None:                               # no approval gate -> no shell tool
        enabled = tuple(t for t in enabled if t != "run_shell")
    enabled = tuple(t for t in enabled if t not in WEB_TOOLS) + (WEB_TOOLS if web is not None else ())
    tools = Toolbox.for_job(settings, bus, job, approvals=approvals, web=web)
    tags = tools.tags                               # job, agent and the current step
    schema = action_schema(enabled)
    system = build_system_prompt(enabled)
    state = RunState()
    stopped = job.stop_event.is_set

    bus.emit("task_started", job=job.id, agent=job.agent, task=task, model=settings.model_name,
             folder=tools.root.name, max_steps=settings.max_steps)
    hashes = {}
    if files:
        for path, content in files.items():
            tools.write_file(path, content)
        hashes = {p: sha(tools.root / p) for p in protected}
        bus.emit("seeded", **tags, files=sorted(files), protected=list(protected))

    head = f"TASK:\n{task}"
    listing = tools.list_files().text if files else ""
    if listing:
        head += f"\n\nFILES IN THE WORKSPACE:\n{listing}"
    state.base = head
    make_plan(client, bus, tags, state, head, "initial")
    history = [{"role": "system", "content": system},
               {"role": "user", "content": f"{head}\n\nYOUR PLAN:\n{state.plan}\n\nTake the first action."}]

    for step in range(1, settings.max_steps + 1):
        tags["step"] = step
        if stopped():
            bus.emit("stopped", **tags)
            return

        # keep the history short: every SUMMARY_EVERY steps, or when the prompt gets too big
        too_big = state.last_prompt_tokens > CONTEXT_LIMIT * settings.ctx
        if len(history) >= 6 and ((step > 1 and (step - 1) % SUMMARY_EVERY == 0) or too_big):
            history = summarise(client, bus, tags, system, state, history,
                                "prompt too large" if too_big else f"every {SUMMARY_EVERY} steps")

        res = client.action(history, tags=tags, schema=schema,
                            validate=lambda a: validate_action(a, enabled))     # LLMError ends the job
        state.last_prompt_tokens = res.reply.prompt_tokens
        if stopped():                                   # Stop pressed while the model was thinking
            bus.emit("stopped", **tags)
            return
        act, args = res.action, res.args
        bus.emit("thought", **tags, text=act.thought)
        bus.emit("action", **tags, tool=act.tool, args=args, attempts=res.attempts)

        # guard: the same action three times in a row
        sig = (act.tool, json.dumps(args, sort_keys=True))
        state.recent.append(sig)
        if state.recent[-3:] == [sig] * 3:
            state.repeat_trips += 1
            state.recent.clear()
            bus.emit("repeat_guard", **tags, tool=act.tool, trips=state.repeat_trips)
            if state.repeat_trips >= 2:
                raise JobFailed(f"stuck: repeated the same {act.tool} action again after re-planning")
            state.replans += 1
            make_plan(client, bus, tags, state, state.base, "replan",
                      f"You repeated the same {act.tool} action 3 times.\n{progress_text(state)}")
            history = rebuild(system, state, history, progress_text(state))
            continue

        if act.tool == "finish":
            problem = finish_problem(state)
            if problem and state.finish_refusals < MAX_FINISH_REFUSALS:
                state.finish_refusals += 1
                bus.emit("finish_refused", **tags, reason=problem, n=state.finish_refusals)
                history += [{"role": "assistant", "content": res.raw},
                            {"role": "user", "content": f"FINISH REFUSED: {problem}"}]
                continue
            changed = [p for p, h in hashes.items() if sha(tools.root / p) != h]
            bus.emit("done", **tags, answer=args["answer"], steps=step, verified=not problem,
                     integrity_ok=not changed, changed_protected=changed)
            return

        # ACT: run the tool
        try:
            result = getattr(tools, act.tool)(**args)
        except Exception as e:                          # a bug in a tool must not kill the loop
            result = ToolResult(False, f"ERROR: {type(e).__name__}: {e}")
            bus.emit("tool_error", **tags, tool=act.tool, error=str(e), traceback=traceback.format_exc())

        # OBSERVE
        if act.tool == "write_file" and args["path"].lower().endswith(".py"):
            state.dirty = True
        if act.tool == "run_python":
            state.last_run_ok = result.ok
            state.dirty = state.dirty and not result.ok
        state.fail_streak = 0 if result.ok else state.fail_streak + 1
        state.log.append(step_line(step, act.tool, args, result))
        bus.emit("observation", **tags, tool=act.tool, ok=result.ok, text=clip(result.text))

        # REFLECT
        obs = f"OBSERVATION from {act.tool} (step {step}):\n{result.text}"
        if not result.ok:
            obs += "\nThis failed. Fix the cause; do not repeat the same call."
            bus.emit("reflect", **tags, failures_in_a_row=state.fail_streak,
                     note="the failure is sent back to the model with 'fix the cause'")
        history += [{"role": "assistant", "content": res.raw}, {"role": "user", "content": obs}]

        if state.fail_streak >= FAIL_STREAK_REPLAN and state.replans < MAX_REPLANS:
            state.replans += 1
            state.fail_streak = 0
            make_plan(client, bus, tags, state, state.base, "replan",
                      f"{FAIL_STREAK_REPLAN} failures in a row.\n{progress_text(state)}")
            history = rebuild(system, state, history, progress_text(state))

    bus.emit("budget_exhausted", **tags, steps=settings.max_steps)
    raise JobFailed(f"step budget exhausted ({settings.max_steps} steps) without finish")