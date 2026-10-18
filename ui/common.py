"""Pieces shared by all tabs: the per-browser event pump, the event formatter, the header."""
import json
import logging
import time

from nicegui import ui

from core.orchestrator import AGENTS

log = logging.getLogger("ui")

STATUS_COLOR = {
    "idle": "grey", "queued": "grey", "running": "blue", "waiting_approval": "orange",
    "done": "positive", "failed": "negative", "stopped": "warning",
}


class EventPump:
    """One per browser tab. Reads new events from the bus and hands each one to the
    functions that tabs registered with subscribe(). Run tick() from a ui.timer."""

    def __init__(self, bus, replay: int = 200):
        self.bus = bus
        self.cursor = max(0, bus.last_seq - replay)   # a refreshed page shows the last events
        self.handlers = []

    def subscribe(self, fn) -> None:
        self.handlers.append(fn)

    def tick(self) -> None:
        events, self.cursor = self.bus.read_since(self.cursor)
        for ev in events:
            for fn in self.handlers:
                try:
                    fn(ev)
                except Exception:
                    log.exception("event handler failed")


def format_event(ev: dict) -> str:
    stamp = time.strftime("%H:%M:%S", time.localtime(ev["t"]))
    extra = {k: v for k, v in ev.items() if k not in ("seq", "t", "kind", "traceback")}
    line = f"{stamp}  {ev['kind']:<14} {json.dumps(extra, default=str)}"
    if ev.get("traceback"):
        line += "\n" + ev["traceback"].rstrip()
    return line


def build_header(ctx) -> None:
    s = ctx.settings
    with ui.header().classes("items-center justify-between"):
        with ui.row().classes("items-center gap-3"):
            ui.label(s.title).classes("text-xl font-bold")
            ui.badge(f"profile: {s.profile}", color="deep-purple")
            ui.label(s.summary).classes("text-xs opacity-70")
        chips = {}
        with ui.row().classes("items-center gap-1"):
            for agent in AGENTS:
                chips[agent] = ui.chip(f"{agent}: idle", icon="circle", color="grey")

        def refresh() -> None:
            for agent, chip in chips.items():
                job = ctx.orchestrator.latest(agent)
                status = job.status if job else "idle"
                chip.set_text(f"{agent}: {status.replace('_', ' ')}")
                chip.props(f"color={STATUS_COLOR.get(status, 'grey')}")

        ui.timer(1.0, refresh)
        