from nicegui import ui

from core.demo_jobs import fake_job
from ui.common import format_event


def build_agent_tab(ctx, pump) -> None:
    orch = ctx.orchestrator

    def start(**kwargs) -> None:
        try:
            orch.start("coding", fake_job, **kwargs)
        except RuntimeError as e:
            ui.notify(str(e), type="warning")

    def stop() -> None:
        if not orch.stop("coding"):
            ui.notify("Nothing is running", type="info")

    with ui.card().classes("w-full"):
        ui.label("Coding agent").classes("text-lg font-bold")
        ui.label("Step 1 test bench: a fake background job writes one event per second.").classes("text-sm opacity-70")
        with ui.row().classes("items-center gap-2"):
            ui.button("Start fake job", icon="play_arrow", on_click=lambda: start())
            ui.button("Start failing job", icon="bug_report", color="orange",
                      on_click=lambda: start(fail_after=3))
            ui.button("Stop", icon="stop", color="negative", on_click=stop)
            ui.button("Clear log", icon="delete", color="grey", on_click=lambda: event_log.clear())
        ui.label(f"Events are also saved to: {ctx.bus.logfile}").classes("text-xs opacity-70")
        event_log = ui.log(max_lines=500).classes("w-full h-96 font-mono text-xs")

    pump.subscribe(lambda ev: event_log.push(format_event(ev)))
    