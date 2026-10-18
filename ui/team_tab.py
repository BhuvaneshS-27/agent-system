from nicegui import ui


def build_team_tab(ctx, pump) -> None:
    with ui.card().classes("w-full"):
        ui.label("Team and logo").classes("text-lg font-bold")
        ui.label("Coming in Step 9: team name, hint, Generate / Regenerate, the logo and the provider chip.")
        ui.label(f"Output folder for the logo: {ctx.settings.outputs_dir}").classes("text-xs opacity-70")
        