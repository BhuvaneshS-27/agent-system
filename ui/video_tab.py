"""Video tab: Generate, 'slide 3 of 8' progress, intro provider chip, player and download."""
import json
import time

from nicegui import app, ui

from agents.video.job import generate_video
from agents.video.tts import VOICE_NAMES
from ui.common import format_event
from ui.team_tab import _static


def build_video_tab(ctx, pump) -> None:
    orch = ctx.orchestrator
    out_dir = ctx.settings.outputs_dir
    if not _static["done"]:
        app.add_static_files("/outputs", str(out_dir))
        _static["done"] = True
    try:
        vmeta = json.loads((out_dir / "video_meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        vmeta = {}
    try:
        team_default = json.loads((out_dir / "logo_meta.json").read_text(encoding="utf-8")).get("team", "")
    except (OSError, ValueError):
        team_default = ""
    prog = {"frames": 1}

    with ui.card().classes("w-full"):
        ui.label("Explainer video").classes("text-lg font-bold")
        ui.label("8 slides, narrated offline with Kokoro, built with ffmpeg. The model is not used, so it can run next to "
                 "the coding agent.").classes("text-xs opacity-70")
        with ui.row().classes("items-center gap-3"):
            team = ui.input("Team name (blank = from the logo)", value=team_default).classes("w-72")
            voice = ui.select(VOICE_NAMES, value=VOICE_NAMES[0], label="Voice").classes("w-40")
            use_intro = ui.switch("Use cached Magic Hour intro", value=True)
        with ui.row().classes("items-center gap-2"):
            gen_btn = ui.button("Generate explainer video", icon="movie", on_click=lambda: start())
            stop_btn = ui.button("Stop", icon="stop", color="negative", on_click=lambda: stop())
            ui.button("Download mp4", icon="download", on_click=lambda: ui.download("/outputs/explainer.mp4")).props("outline")
        chip = ui.chip("intro: -", icon="movie_filter", color="grey")
        if vmeta.get("intro"):
            chip.set_text(f"intro: {vmeta['intro']}")
            chip.props("color=positive")
        bar = ui.linear_progress(value=0, show_value=False).classes("w-full")
        status = ui.label("Idle.").classes("text-sm")
        with ui.row().classes("items-center gap-3") as err_row:
            err = ui.label("").classes("text-negative text-sm")
            ui.button("Retry", icon="replay", on_click=lambda: start()).props("color=negative outline")
        err_row.set_visibility(False)
        player = ui.video("/outputs/explainer.mp4").classes("w-full max-w-4xl")
        player.set_visibility((out_dir / "explainer.mp4").exists())
        ui.label(f"Saved as {out_dir / 'explainer.mp4'}. If a run fails, the last saved video stays.").classes("text-xs opacity-60")
        with ui.expansion("Video activity (every step as an event)", icon="list").classes("w-full"):
            log = ui.log(max_lines=300).classes("w-full h-48")

    def start() -> None:
        try:
            orch.start("video", generate_video, settings=ctx.settings, team_name=team.value or "",
                       voice=voice.value, use_intro=use_intro.value)
        except RuntimeError as e:
            ui.notify(str(e), type="warning")

    def stop() -> None:
        if not orch.stop("video"):
            ui.notify("Nothing is running", type="info")

    def on_event(ev: dict) -> None:
        if ev.get("agent") != "video":
            return
        log.push(format_event(ev))
        k = ev["kind"]
        if k == "job_started":
            err_row.set_visibility(False)
            bar.set_value(0)
            status.set_text("Starting ...")
        elif k == "video_start":
            prog["frames"] = max(1, ev["frames"])
            status.set_text(f"Preparing {ev['slides']} slides (voice {ev['voice']}) ...")
        elif k == "video_diagram":
            status.set_text(f"Diagram {ev['i']} of {ev['of']} ({'cached' if ev['cached'] else 'rendered'})")
        elif k == "video_fallback":
            status.set_text(f"Fallback: {ev['what']} -> {ev['use']} ({str(ev['reason'])[:80]})")
        elif k == "video_slide":
            status.set_text(f"Slide {ev['n']} of {ev['total']}: {ev['title']}")
        elif k == "video_intro":
            chip.set_text(f"intro: {ev['label']}")
            chip.props("color=positive" if ev["label"].startswith("magic") else "color=orange")
        elif k == "video_segment":
            bar.set_value(ev["k"] / prog["frames"])
            status.set_text(f"Slide {ev['slide']} of {ev['total']} (part {ev['frame']} of {ev['frames']}), "
                            f"segment {ev['k']} of {ev['of']}")
        elif k == "video_join":
            status.set_text("Joining segments with ffmpeg ...")
        elif k == "video_done":
            bar.set_value(1)
            status.set_text(f"Done: about {ev['seconds']} s of video, built in {ev['took']} s.")
            player.set_source(f"/outputs/explainer.mp4?t={int(time.time() * 1000)}")
            player.set_visibility(True)
        elif k == "video_stopped":
            status.set_text("Stopped.")
        elif k == "job_failed":
            err.set_text(str(ev.get("error", ""))[:300])
            err_row.set_visibility(True)
            status.set_text("Failed. The last saved video is kept.")

    pump.subscribe(on_event)

    def buttons() -> None:
        running = orch.is_running("video")
        gen_btn.set_enabled(not running)
        stop_btn.set_enabled(running)

    ui.timer(1.0, buttons)