"""Image agent job: provider chain (Cloudflare Worker > Pollinations) with retries, then the name overlay.

Events (agent="image"): image_start, image_provider, image_retry, image_fallback, image_done, image_failed,
image_stopped. If every provider fails the job fails and the last saved outputs/logo.png stays untouched.
"""
import json
import random
import time
from io import BytesIO

from PIL import Image
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential

from agents.image.compose import add_team_name
from agents.image.providers import ImageError, ProviderSkipped, cloudflare, pollinations
from core.jobs import JobFailed
from core.settings import ROOT

PROVIDERS = [("cloudflare", "Cloudflare FLUX", cloudflare), ("pollinations", "Pollinations", pollinations)]
STYLE = "emblem, flat vector style, centered, solid background, no text, no letters"


class ImageStopped(Exception):
    pass


def read_meta(outputs_dir) -> dict:
    try:
        return json.loads((outputs_dir / "logo_meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def generate_logo(job, bus, *, settings, team_name, hint="", disabled=(), seed=None, name_only=False):
    tags = {"agent": "image", "job": job.id}
    name = " ".join(str(team_name).split())
    if not name:
        raise JobFailed("Type a team name first.")
    out = settings.outputs_dir
    logo, raw, meta_path = out / "logo.png", out / "logo_raw.png", out / "logo_meta.json"
    fonts = ROOT / "assets" / "fonts"
    t0 = time.time()

    if name_only:                                                    # redraw the name on the saved emblem, no network
        if not raw.exists():
            raise JobFailed("There is no emblem yet. Press Generate first.")
        meta = read_meta(out)
        add_team_name(raw, name, logo, fonts)
        meta["team"] = name
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        job.provider_used = meta.get("label", "")
        bus.emit("image_done", **tags, path=str(logo), label=meta.get("label", ""), attempts=meta.get("attempts", 1),
                 seed=meta.get("seed"), name_only=True, seconds=round(time.time() - t0, 1))
        return

    seed = seed if seed is not None else random.randint(1, 2**31 - 1)
    prompt = f"{hint.strip() or 'abstract team emblem'}, {STYLE}"
    bus.emit("image_start", **tags, team=name, prompt=prompt, seed=seed, disabled=sorted(disabled))
    trail = []
    try:
        for i, (key, label, fn) in enumerate(PROVIDERS):
            nxt = PROVIDERS[i + 1][1] if i + 1 < len(PROVIDERS) else ""
            if job.stop_event.is_set():
                raise ImageStopped()
            if key in disabled:
                trail.append((label, "switched off"))
                bus.emit("image_fallback", **tags, provider=label, next=nxt, reason="switched off in the dashboard")
                continue
            calls = [0]

            def once(fn=fn):
                calls[0] += 1
                return fn(prompt, seed)

            def on_retry(rs, label=label):
                bus.emit("image_retry", **tags, provider=label, attempt=rs.attempt_number,
                         reason=str(rs.outcome.exception())[:150], wait=round(rs.next_action.sleep, 1))

            def sleep(seconds):
                if job.stop_event.wait(seconds):
                    raise ImageStopped()

            retrying = Retrying(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=8),
                                retry=retry_if_exception(lambda e: isinstance(e, ImageError) and e.temporary),
                                before_sleep=on_retry, sleep=sleep, reraise=True)
            bus.emit("image_provider", **tags, provider=key, label=label)
            try:
                data = retrying(once)
                img = Image.open(BytesIO(data))
                img.load()
            except (ProviderSkipped, ImageError, OSError) as e:      # OSError: Pillow cannot read the bytes
                job.attempts += calls[0]
                trail.append((label, str(e)[:150]))
                bus.emit("image_fallback", **tags, provider=label, next=nxt, reason=str(e)[:150])
                continue
            job.attempts += calls[0]
            job.provider_used = label
            img.convert("RGB").save(raw)
            add_team_name(raw, name, logo, fonts)
            meta = {"team": name, "hint": hint, "provider": key, "label": label, "attempts": calls[0], "seed": seed,
                    "created": time.strftime("%Y-%m-%d %H:%M:%S")}
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
            bus.emit("image_done", **tags, path=str(logo), label=label, attempts=calls[0], seed=seed,
                     name_only=False, seconds=round(time.time() - t0, 1))
            return
    except ImageStopped:
        bus.emit("image_stopped", **tags)
        return                                                       # stop_event is set: the job shows as "stopped"
    summary = "; ".join(f"{label}: {why}" for label, why in trail)
    bus.emit("image_failed", **tags, trail=trail)
    raise JobFailed(f"No image provider worked. {summary}")