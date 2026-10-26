"""Team & Image tab: team name, hint, Generate / Regenerate, the logo and the provider chip."""
import time

from nicegui import app, ui

from agents.image.job import generate_logo, read_meta
from ui.common import format_event

_static = {"done": False}


def build_team_tab(ctx, pump) -> None:
    orch = ctx.orchestrator
    out_dir = ctx.settings.outputs_dir
    if not _static["done"]:                                  # serve outputs/ once, so the page can show logo.png
        app.add_static_files("/outputs", str(out_dir))
        _static["done"] = True
    meta = read_meta(out_dir)
    last = {"team": meta.get("team", ""), "hint": meta.get("hint", "")}

    with ui.card().classes("w-full"):
        ui.label("Team and logo").classes("text-lg font-bold")
        ui.label("The model draws a text-free emblem; Pillow writes the team name so it is always spelled right. "
                 "Chain: Cloudflare Worker (FLUX) > Pollinations.").classes("text-xs opacity-70")
        with ui.row().classes("items-center gap-3 w-full"):
            team = ui.input("Team name", value=meta.get("team", "")).classes("w-64")
            hint = ui.input("Hint for the emblem", value=meta.get("hint", ""), placeholder="e.g. blue owl, circuit lines").classes("w-96")
        with ui.row().classes("items-center gap-4"):
            sw_cf = ui.switch("Cloudflare", value=True)
            sw_pol = ui.switch("Pollinations", value=True)
            ui.label("(switch off to demo the fallback)").classes("text-xs opacity-60")
        with ui.row().classes("items-center gap-2"):
            gen_btn = ui.button("Generate", icon="auto_awesome", on_click=lambda: start())
            regen_btn = ui.button("Regenerate (new seed)", icon="refresh", on_click=lambda: start(reuse=True))
            name_btn = ui.button("Update name only", icon="title", on_click=lambda: start(name_only=True)).props("outline")
            stop_btn = ui.button("Stop", icon="stop", color="negative", on_click=lambda: stop())
            ui.button("Download", icon="download", on_click=lambda: ui.download("/outputs/logo.png")).props("outline")
        chip = ui.chip("provider: -", icon="image", color="grey")
        if meta.get("label"):
            chip.set_text(f"provider: {meta['label']} (attempt {meta.get('attempts', 1)})")
            chip.props("color=positive")
        status = ui.label("Idle.").classes("text-sm")
        with ui.row().classes("items-center gap-3") as err_row:
            err = ui.label("").classes("text-negative text-sm")
            ui.button("Retry", icon="replay", on_click=lambda: start(reuse=True)).props("color=negative outline")
        err_row.set_visibility(False)
        image = ui.image("/outputs/logo.png").classes("w-96")
        image.set_visibility((out_dir / "logo.png").exists())
        ui.label(f"Saved as {out_dir / 'logo.png'}. If both providers fail, this last saved logo stays.").classes("text-xs opacity-60")
        with ui.expansion("Image activity (every step as an event)", icon="list").classes("w-full"):
            log = ui.log(max_lines=200).classes("w-full h-48")

    def disabled() -> tuple:
        return tuple(k for k, sw in (("cloudflare", sw_cf), ("pollinations", sw_pol)) if not sw.value)

    def start(reuse: bool = False, name_only: bool = False) -> None:
        name = (last["team"] if reuse and last["team"] else team.value or "").strip()
        hint_text = last["hint"] if reuse and last["team"] else (hint.value or "")
        if not name:
            ui.notify("Type a team name first", type="warning")
            return
        last.update(team=name, hint=hint_text)
        try:
            orch.start("image", generate_logo, settings=ctx.settings, team_name=name, hint=hint_text,
                       disabled=disabled(), name_only=name_only)
        except RuntimeError as e:
            ui.notify(str(e), type="warning")

    def stop() -> None:
        if not orch.stop("image"):
            ui.notify("Nothing is running", type="info")

    def on_event(ev: dict) -> None:
        if ev.get("agent") != "image":
            return
        log.push(format_event(ev))
        k = ev["kind"]
        if k == "job_started":
            err_row.set_visibility(False)
        elif k == "image_start":
            status.set_text(f"Generating (seed {ev['seed']}) ...")
        elif k == "image_provider":
            chip.set_text(f"provider: {ev['label']} (trying)")
            chip.props("color=blue")
            status.set_text(f"Trying {ev['label']} ...")
        elif k == "image_retry":
            status.set_text(f"{ev['provider']}: attempt {ev['attempt']} failed ({ev['reason'][:80]}), retrying in {ev['wait']} s")
        elif k == "image_fallback":
            status.set_text(f"{ev['provider']} failed: {ev['reason'][:80]}" + (f" -> {ev['next']}" if ev["next"] else ""))
        elif k == "image_done":
            chip.set_text(f"provider: {ev['label']} (attempt {ev['attempts']})")
            chip.props("color=positive")
            status.set_text("Done." if not ev.get("name_only") else "Name updated.")
            image.set_source(f"/outputs/logo.png?t={int(time.time() * 1000)}")
            image.set_visibility(True)
        elif k == "image_stopped":
            status.set_text("Stopped.")
        elif k == "job_failed":
            chip.set_text("provider: none worked")
            chip.props("color=negative")
            err.set_text(str(ev.get("error", ""))[:300])
            err_row.set_visibility(True)
            status.set_text("Failed. The last saved logo is kept.")

    pump.subscribe(on_event)

    def buttons() -> None:
        running = orch.is_running("image")
        for b in (gen_btn, regen_btn, name_btn):
            b.set_enabled(not running)
        stop_btn.set_enabled(running)

    ui.timer(1.0, buttons)