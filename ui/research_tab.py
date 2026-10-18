from nicegui import ui


def build_research_tab(ctx, pump) -> None:
    with ui.card().classes("w-full"):
        ui.label("Startup research").classes("text-lg font-bold")
        ui.label("Coming in Step 8: category box, top-10 table, formula panel, chart, news feed and preset selector.")