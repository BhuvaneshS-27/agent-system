"""Pillow slides (1280x720): title slide with the logo, and bullet text slides."""
from pathlib import Path

from PIL import Image, ImageDraw

from agents.image.compose import _font

W, H = 1280, 720
BG, FG, MUTED, ACCENT = "#0f172a", "white", "#94a3b8", "#f59e0b"


def wrap(draw, text, font, max_w):
    lines, cur = [], ""
    for word in str(text).split():
        trial = f"{cur} {word}".strip()
        if not cur or draw.textlength(trial, font=font) <= max_w:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def text_slide(title, bullets, out, fonts_dir) -> str:
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text((64, 44), title, font=_font(fonts_dir, 52), fill=FG)
    d.rectangle([64, 118, 264, 124], fill=ACCENT)
    f = _font(fonts_dir, 34)
    y = 165
    for b in bullets:
        d.ellipse([64, y + 14, 76, y + 26], fill=ACCENT)
        for line in wrap(d, b, f, W - 64 - 100 - 40):
            d.text((100, y), line, font=f, fill=FG)
            y += 46
        y += 16
    im.save(out)
    return str(out)


def title_slide(team, title, subtitle, logo_path, out, fonts_dir) -> str:
    """Fallback intro (and slide 1 of the video when there is no Magic Hour clip)."""
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    x = 90
    if logo_path and Path(logo_path).exists():
        try:
            lg = Image.open(logo_path).convert("RGBA")
            lg.thumbnail((520, 560))
            im.paste(lg, (64, (H - lg.height) // 2), lg)
            x = 640
        except OSError:
            pass
    d.text((x, 230), team, font=_font(fonts_dir, 36), fill=ACCENT)
    y = 290
    for line in wrap(d, title, _font(fonts_dir, 62), W - x - 60):
        d.text((x, y), line, font=_font(fonts_dir, 62), fill=FG)
        y += 76
    for line in wrap(d, subtitle, _font(fonts_dir, 30), W - x - 60):
        d.text((x, y + 16), line, font=_font(fonts_dir, 30), fill=MUTED)
        y += 40
    im.save(out)
    return str(out)