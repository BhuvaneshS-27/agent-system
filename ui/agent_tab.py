import json

from nicegui import run, ui

from agents.coding.llm import LLMClient
from agents.coding.loop import run_one_step
from core.demo_jobs import fake_job
from ui.common import format_event

DEFAULT_TASK = "Write a Python function is_prime(n) and test it with a few numbers."
LLM_COLOR = {"online": "positive", "loading": "orange", "offline": "negative", "checking": "grey"}


def build_agent_tab(ctx, pump) -> None:
    orch = ctx.orchestrator
    client = LLMClient(ctx.settings)            # no bus: health checks are not logged

    # ---------------------------------------------------------------- actions
    def run_step() -> None:
        task = (task_box.value or "").strip()
        if not task:
            ui.notify("Type a task first", type="warning")
            return
        try:
            orch.start("coding", run_one_step, settings=ctx.settings, task=task)
        except RuntimeError as e:
            ui.notify(str(e), type="warning")

    def stop() -> None:
        if not orch.stop("coding"):
            ui.notify("Nothing is running", type="info")

    def start_fake(**kwargs) -> None:
        try:
            orch.start("coding", fake_job, **kwargs)
        except RuntimeError as e:
            ui.notify(str(e), type="warning")

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
        task_box = ui.textarea("Task", value=DEFAULT_TASK).props("outlined autogrow").classes("w-full")
        with ui.row().classes("items-center gap-2"):
            ui.button("Run one step", icon="play_arrow", on_click=run_step)
            ui.button("Stop", icon="stop", color="negative", on_click=stop)
            ui.button("Clear log", icon="delete", color="grey", on_click=lambda: event_log.clear())
        ui.label("Step 2: the model picks the first action. Nothing is executed yet (Steps 3 and 4).") \
            .classes("text-xs opacity-70")

    with ui.row().classes("w-full no-wrap items-stretch"):
        with ui.card().classes("w-1/2"):
            ui.label("Thought").classes("font-bold")
            thought_label = ui.label("-").classes("text-sm")
            usage_label = ui.label("").classes("text-xs opacity-70")
        with ui.card().classes("w-1/2"):
            with ui.row().classes("items-center gap-2"):
                ui.label("Chosen action").classes("font-bold")
                tool_badge = ui.badge("-", color="deep-purple")
            action_code = ui.code("{}", language="json").classes("w-full")

    with ui.card().classes("w-full"):
        ui.label(f"Every event is also saved to: {ctx.bus.logfile}").classes("text-xs opacity-70")
        event_log = ui.log(max_lines=500).classes("w-full h-80 font-mono text-xs")

    with ui.expansion("Step 1 test bench (fake jobs)", icon="science").classes("w-full"):
        with ui.row().classes("items-center gap-2"):
            ui.button("Start fake job", icon="play_arrow", on_click=lambda: start_fake())
            ui.button("Start failing job", icon="bug_report", color="orange",
                      on_click=lambda: start_fake(fail_after=3))

    # ------------------------------------------------------- live event panels
    def on_event(ev: dict) -> None:
        event_log.push(format_event(ev))
        kind = ev["kind"]
        if kind == "task_started":
            thought_label.set_text("(waiting for the model...)")
            usage_label.set_text("")
            tool_badge.set_text("-")
            action_code.set_content("{}")
        elif kind == "thought":
            thought_label.set_text(ev["text"])
        elif kind == "action":
            tool_badge.set_text(ev["tool"])
            action_code.set_content(json.dumps({"tool": ev["tool"], "args": ev["args"]}, indent=2))
        elif kind == "llm_call":
            usage_label.set_text(
                f"prompt {ev['prompt_tokens']} tokens | answer {ev['completion_tokens']} tokens | "
                f"{ev['seconds']} s | {ev['tokens_per_second']} tok/s")
        elif kind == "error":
            thought_label.set_text(f"ERROR: {ev['error']}")

    pump.subscribe(on_event)
    ui.timer(3.0, refresh_llm)