"""ffmpeg (bundled by imageio-ffmpeg): one segment per frame, then join them."""
import subprocess
from pathlib import Path

import imageio_ffmpeg

FF = imageio_ffmpeg.get_ffmpeg_exe()


def run(cmd) -> None:
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError("ffmpeg failed: " + r.stderr.strip()[-300:])


def make_segment(png, wav, out) -> None:
    run([FF, "-y", "-loop", "1", "-framerate", "25", "-i", png, "-i", wav,
         "-c:v", "libx264", "-tune", "stillimage", "-preset", "veryfast", "-pix_fmt", "yuv420p",
         "-vf", "scale=1280:720", "-c:a", "aac", "-ar", "24000", "-ac", "1",
         "-af", "apad=pad_dur=0.4", "-shortest", out])


def concat(segments, out) -> None:
    lst = Path(out).with_suffix(".txt")
    lst.write_text("".join(f"file '{Path(s).resolve().as_posix()}'\n" for s in segments), encoding="utf-8")
    run([FF, "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c:v", "libx264", "-preset", "veryfast",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart", out])