"""Mermaid text -> PNG (offline, Playwright), highlighted frames, diagram slides, timeline diagram.

Many diagrams are rendered with ONE Chromium start. PNGs are cached by the hash of the diagram text, so an
unchanged diagram is never rendered twice. Call render_many from a job thread, not from an async handler.
"""
import hashlib
from pathlib import Path

from PIL import Image, ImageDraw

from agents.image.compose import _font

W, H = 1280, 720
BG = "#0f172a"


def highlight(code: str, node_id: str) -> str:
    """Same diagram with one node coloured, so the picture follows the narration."""
    return code + f"\nclassDef active fill:#f59e0b,stroke:#ffffff,color:#000000\nclass {node_id} active"


def png_path(out_dir, code: str) -> Path:
    return Path(out_dir) / (hashlib.sha1(code.encode("utf-8")).hexdigest()[:16] + ".png")


def render_many(codes, out_dir, mermaid_js, on_each=None, stop=None) -> dict:
    """Returns {code: png path or None}. on_each(i, n, cached, ok, error) is called after every diagram."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    unique = list(dict.fromkeys(codes))
    result = {c: (str(png_path(out_dir, c)) if png_path(out_dir, c).exists() else None) for c in unique}
    todo = [c for c in unique if result[c] is None]
    n = len(unique)
    done = n - len(todo)
    for i in range(done):
        if on_each:
            on_each(i + 1, n, True, True, "")
    if not todo:
        return result
    if not Path(mermaid_js).exists():
        raise FileNotFoundError(f"{mermaid_js} is missing (download mermaid.min.js, see the setup commands)")
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 720}, device_scale_factor=2)
            page.set_content(f"<html><body style='margin:0;padding:24px;background:{BG}'><div id='box'></div></body></html>")
            page.add_script_tag(path=str(Path(mermaid_js).resolve()))
            page.evaluate("mermaid.initialize({startOnLoad: false, theme: 'dark', flowchart: {useMaxWidth: false}})")
            for k, code in enumerate(todo):
                if stop is not None and stop.is_set():
                    break
                target = png_path(out_dir, code)
                try:
                    page.evaluate("""async ([code, id]) => {
                        const out = await mermaid.render(id, code);
                        document.getElementById('box').innerHTML = out.svg;
                    }""", [code, f"g{k}"])
                    page.locator("#box svg").screenshot(path=str(target))
                    result[code] = str(target)
                    ok, err = True, ""
                except Exception as e:                         # one bad diagram must not kill the rest
                    ok, err = False, str(e)[:200]
                done += 1
                if on_each:
                    on_each(done, n, False, ok, err)
        finally:
            browser.close()
    return result


def diagram_slide(diagram_png, title, caption, out_png, fonts_dir) -> str:
    slide = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(slide)
    d.text((48, 26), title, font=_font(fonts_dir, 40), fill="white")
    img = Image.open(diagram_png).convert("RGBA")
    area_h = H - 110 - 90
    img.thumbnail((W - 96, area_h))
    slide.paste(img, ((W - img.width) // 2, 105 + (area_h - img.height) // 2), img)
    f = _font(fonts_dir, 24)
    words, lines, cur = str(caption).split(), [], ""
    for w in words:
        t = f"{cur} {w}".strip()
        if cur and d.textlength(t, font=f) > W - 96:
            lines.append(cur)
            cur = w
        else:
            cur = t
    if cur:
        lines.append(cur)
    for i, line in enumerate(lines[:2]):
        d.text((48, H - 78 + i * 32), line, font=f, fill="#94a3b8")
    slide.save(out_png)
    return str(out_png)


def clean(s, n=40) -> str:
    return "".join(c for c in str(s) if c.isalnum() or c in " _./-")[:n].strip() or "x"


def timeline_mermaid(events: list, limit: int = 10) -> str:
    """Sequence diagram of key events of a real run (events from events.jsonl)."""
    lines = ["sequenceDiagram", "  participant L as LLM", "  participant T as Tools", "  participant U as You"]
    n = 0
    for e in events:
        k = e.get("kind")
        if k == "action":
            lines.append(f"  L->>T: {clean(e.get('tool'))}")
        elif k == "observation":
            lines.append(f"  T-->>L: {'ok' if e.get('ok', True) else 'error'}")
        elif k == "approval_request":
            lines.append(f"  T->>U: approve {clean(e.get('command') or e.get('cmd'), 25)}")
        elif k == "approval_result":
            good = e.get("ok", e.get("approved", False))
            lines.append(f"  U-->>T: {'approved' if good else 'denied'}")
        else:
            continue
        n += 1
        if n >= limit:
            break
    return "\n".join(lines)


SAMPLE_TIMELINE = timeline_mermaid([
    {"kind": "action", "tool": "write_file"}, {"kind": "observation", "ok": True},
    {"kind": "action", "tool": "run_python"}, {"kind": "observation", "ok": False},
    {"kind": "action", "tool": "write_file"}, {"kind": "action", "tool": "run_python"},
    {"kind": "observation", "ok": True}, {"kind": "action", "tool": "finish"}])