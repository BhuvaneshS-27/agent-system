"""Starts the dashboard:  python app.py   then open http://localhost:8080"""
import time

from nicegui import ui

from core.bus import EventBus
from core.context import AppContext
from core.orchestrator import Orchestrator
from core.settings import load_settings
from ui.agent_tab import build_agent_tab
from ui.common import EventPump, build_header
from ui.research_tab import build_research_tab
from ui.team_tab import build_team_tab
from ui.video_tab import build_video_tab

# Note: our folder is called ui/, and NiceGUI also exposes "ui". Always write
# "from ui.xxx import yyy" for our files, never "import ui.xxx".

settings = load_settings()
bus = EventBus(settings.runs_dir / f"session-{time.strftime('%Y%m%d-%H%M%S')}" / "events.jsonl")
ctx = AppContext(settings=settings, bus=bus, orchestrator=Orchestrator(bus))
bus.emit("app_started", profile=settings.profile, model=settings.model_name, ctx=settings.ctx)


@ui.page("/")
def index() -> None:
    pump = EventPump(bus)                       # one pump per browser tab
    build_header(ctx)

    with ui.tabs().classes("w-full") as tabs:
        t_team = ui.tab("Team & Image", icon="palette")
        t_video = ui.tab("Video", icon="movie")
        t_agent = ui.tab("Coding Agent", icon="terminal")
        t_research = ui.tab("Startup Research", icon="query_stats")

    with ui.tab_panels(tabs, value=t_agent).classes("w-full"):
        with ui.tab_panel(t_team):
            build_team_tab(ctx, pump)
        with ui.tab_panel(t_video):
            build_video_tab(ctx, pump)
        with ui.tab_panel(t_agent):
            build_agent_tab(ctx, pump)
        with ui.tab_panel(t_research):
            build_research_tab(ctx, pump)

    ui.timer(0.5, pump.tick)                    # drains the bus into the tabs


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(host=settings.host, port=settings.port, title=settings.title, dark=True,
           reload=False, show=settings.open_browser)