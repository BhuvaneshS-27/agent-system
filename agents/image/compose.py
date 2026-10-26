"""Draws the team name under the emblem with Pillow, so the text is always spelled correctly."""
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SYSTEM_FONTS = ["C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/segoeuib.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
BAND_COLOR = (15, 23, 42)


def _font(fonts_dir, size: int):
    files = sorted(Path(fonts_dir).glob("*.ttf")) + sorted(Path(fonts_dir).glob("*.otf"))
    files.sort(key=lambda p: "bold" not in p.name.lower())          # bold fonts first
    for path in [*map(str, files), *SYSTEM_FONTS]:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default(size=size)


def add_team_name(src, name: str, out, fonts_dir) -> Path:
    """Emblem on top, a plain band below it with the name. The font shrinks until the name fits 90% of the width."""
    im = Image.open(src).convert("RGB")
    W, H = im.size
    band = int(H * 0.22)
    canvas = Image.new("RGB", (W, H + band), BAND_COLOR)
    canvas.paste(im, (0, 0))
    d = ImageDraw.Draw(canvas)
    size = int(H * 0.12)
    font = _font(fonts_dir, size)
    while size > 12:
        font = _font(fonts_dir, size)
        if d.textlength(name, font=font) <= W * 0.9:
            break
        size -= 2
    d.text((W / 2, H + band / 2), name, font=font, fill="white", anchor="mm")
    out = Path(out)
    tmp = out.with_name(out.stem + ".tmp.png")
    canvas.save(tmp)
    tmp.replace(out)                                   # a failed save never damages the last good logo
    return out