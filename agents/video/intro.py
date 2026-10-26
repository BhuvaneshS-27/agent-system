"""Optional intro: a Magic Hour clip generated ONCE by hand and cached in assets/intro/intro.mp4.
A normal video run never spends credits. If the clip is missing or ffmpeg fails, the Pillow title slide is used."""
import os
import shutil

from PIL import Image, ImageDraw

from agents.image.compose import _font
from agents.video.assemble import FF, run
from core.settings import ROOT

INTRO = ROOT / "assets" / "intro" / "intro.mp4"
PROMPT = ("abstract glowing network of connected nodes and data streams, dark blue background, "
          "cinematic slow camera movement, no text, no letters, no logos")


def generate_intro_clip() -> str:
    """Run ONCE by hand (scripts/make_intro.py). Costs Magic Hour credits. Needs MAGIC_HOUR_API_KEY in .env."""
    from magic_hour import Client            # verify the package name and the call in Magic Hour's docs
    raw = ROOT / "assets" / "intro" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    client = Client(token=os.environ["MAGIC_HOUR_API_KEY"])
    client.v1.text_to_video.generate(
        end_seconds=5.0, aspect_ratio="16:9", style={"prompt": PROMPT}, name="Agent system intro",
        resolution="480p", wait_for_completion=True, download_outputs=True, download_directory=str(raw))
    newest = max(raw.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
    shutil.copy(newest, INTRO)
    return str(INTRO)


def make_overlay(team, subtitle, out, fonts_dir) -> str:
    im = Image.new("RGBA", (1280, 720), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 520, 1280, 720], fill=(15, 23, 42, 200))
    size = 64
    while size > 28 and d.textlength(team, font=_font(fonts_dir, size)) > 1180:
        size -= 4
    d.text((640, 590), team, font=_font(fonts_dir, size), fill="white", anchor="mm")
    d.text((640, 665), subtitle, font=_font(fonts_dir, 30), fill="#94a3b8", anchor="mm")
    im.save(out)
    return str(out)


def intro_segment(team, subtitle, wav, out, title_slide_png, overlay_png, fonts_dir, make_segment, use_clip=True):
    """Returns (provider label, note) for the dashboard chip and the log."""
    note = ""
    if not use_clip:
        note = "switched off in the dashboard"
    elif not INTRO.exists():
        note = "no cached clip (assets/intro/intro.mp4)"
    else:
        try:
            make_overlay(team, subtitle, overlay_png, fonts_dir)
            run([FF, "-y", "-i", INTRO, "-i", overlay_png, "-i", wav, "-filter_complex",
                 "[0:v]scale=1280:720,fps=25,tpad=stop_mode=clone:stop_duration=30[v0];[v0][1:v]overlay=0:0[v]",
                 "-map", "[v]", "-map", "2:a", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                 "-c:a", "aac", "-ar", "24000", "-ac", "1", "-af", "apad=pad_dur=0.4", "-shortest", out])
            return "magic_hour (cached clip)", ""
        except Exception as e:
            note = f"clip failed: {str(e)[:150]}"
    make_segment(title_slide_png, wav, out)
    return "pillow title slide", note