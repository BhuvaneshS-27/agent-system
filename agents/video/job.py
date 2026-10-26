"""Video agent job: facts -> slides -> Mermaid PNGs -> Kokoro narration -> ffmpeg segments -> outputs/explainer.mp4.

Events (agent="video"): video_start, video_diagram, video_fallback, video_slide, video_intro, video_segment,
video_join, video_done, video_stopped. It never calls the LLM. If it fails, the last explainer.mp4 stays.
"""
import json
import os
import time

import soundfile as sf

from agents.image.job import read_meta
from agents.video import assemble, plan, slides, tts
from agents.video.diagrams import diagram_slide, render_many
from agents.video.intro import intro_segment
from core.jobs import JobFailed
from core.settings import ROOT

PAD = 0.4


def generate_video(job, bus, *, settings, team_name="", voice="af_heart", use_intro=True):
    tags = {"agent": "video", "job": job.id}
    t0 = time.time()
    fonts = ROOT / "assets" / "fonts"
    work = settings.cache_dir / "video"
    seg_dir, slide_dir, diag_dir = work / "segments", work / "slides", work / "diagrams"
    for d in (seg_dir, slide_dir, diag_dir):
        d.mkdir(parents=True, exist_ok=True)
    for old in seg_dir.glob("*"):
        old.unlink()

    meta = read_meta(settings.outputs_dir)
    team = " ".join(str(team_name).split()) or meta.get("team") or "Our team"
    events, run_facts = plan.find_run(settings.runs_dir)
    facts = {"team": team, "model": settings.model_name, "image": meta.get("label") or "the image agent",
             "steps": 0, "approvals": 0, "errors": 0, **run_facts}
    deck = plan.build_slides(facts, events)
    total = len(deck)
    n_frames = sum(len(s["frames"]) for s in deck)
    if not tts.files_ok():
        raise JobFailed("Kokoro files are missing in assets/voices/ (kokoro-v1.0.onnx and voices-v1.0.bin).")
    bus.emit("video_start", **tags, team=team, slides=total, frames=n_frames, voice=voice,
             real_run=bool(events), model=settings.model_name)

    # 1. diagrams: one Chromium start, cached by diagram text
    codes = [f["code"] for s in deck for f in s["frames"] if f.get("code")]

    def on_diag(i, n, cached, ok, err):
        bus.emit("video_diagram", **tags, i=i, of=n, cached=cached, ok=ok, error=err)

    try:
        pngs = render_many(codes, diag_dir, ROOT / "assets" / "mermaid.min.js", on_each=on_diag, stop=job.stop_event)
    except Exception as e:                                   # no Chromium / no mermaid.js: text slides instead
        pngs = {}
        bus.emit("video_fallback", **tags, what="diagrams", reason=str(e)[:200], use="text slides")
    if job.stop_event.is_set():
        bus.emit("video_stopped", **tags)
        return

    # 2. one segment per narrated frame
    segments, seconds, k = [], 0.0, 0
    logo = settings.outputs_dir / "logo.png"
    for n, slide in enumerate(deck, 1):
        bus.emit("video_slide", **tags, n=n, total=total, title=slide["title"], frames=len(slide["frames"]))
        for j, fr in enumerate(slide["frames"], 1):
            if job.stop_event.is_set():
                bus.emit("video_stopped", **tags)
                return
            k += 1
            wav, mp4 = seg_dir / f"{k:02d}.wav", seg_dir / f"{k:02d}.mp4"
            png = slide_dir / f"{k:02d}.png"
            seconds += tts.synth(fr["text"], wav, voice=voice) + PAD
            if slide["kind"] == "intro":
                slides.title_slide(team, "Multi-Agent System", "Coding, image, video and research agents", logo, png, fonts)
                label, note = intro_segment(team, "Multi-Agent System", wav, mp4, png, slide_dir / "intro_overlay.png",
                                            fonts, assemble.make_segment, use_clip=use_intro)
                job.provider_used = label
                bus.emit("video_intro", **tags, label=label, note=note)
            else:
                if slide["kind"] == "text":
                    slides.text_slide(slide["title"], slide["bullets"], png, fonts)
                elif pngs.get(fr.get("code")):
                    diagram_slide(pngs[fr["code"]], slide["title"], fr["text"], png, fonts)
                else:
                    slides.text_slide(slide["title"], [fr["text"]], png, fonts)
                    bus.emit("video_fallback", **tags, what="diagram", slide=n, reason="no diagram image", use="text slide")
                assemble.make_segment(png, wav, mp4)
            segments.append(mp4)
            job.attempts += 1
            bus.emit("video_segment", **tags, k=k, of=n_frames, slide=n, total=total, frame=j,
                     frames=len(slide["frames"]), seconds=round(seconds, 1))

    # 3. join and publish atomically
    if job.stop_event.is_set():
        bus.emit("video_stopped", **tags)
        return
    bus.emit("video_join", **tags, segments=len(segments))
    tmp = work / "explainer_tmp.mp4"
    assemble.concat(segments, tmp)
    out = settings.outputs_dir / "explainer.mp4"
    os.replace(tmp, out)
    (settings.outputs_dir / "video_meta.json").write_text(json.dumps(
        {"intro": job.provider_used, "voice": voice, "slides": total, "seconds": round(seconds),
         "created": time.strftime("%Y-%m-%d %H:%M:%S")}), encoding="utf-8")
    bus.emit("video_done", **tags, path=str(out), intro=job.provider_used, seconds=round(seconds),
             took=round(time.time() - t0, 1))