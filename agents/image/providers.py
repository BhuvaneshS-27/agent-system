"""Image providers. Each function returns the image bytes, or raises ImageError / ProviderSkipped."""
import os
from urllib.parse import quote

import requests

TIMEOUT = (10, 120)                       # (connect, read) seconds
HEADERS = {"User-Agent": "AgentSystemStudentProject/1.0"}


class ImageError(Exception):
    """temporary=True (timeout, connection, 429, 5xx) means: worth retrying."""

    def __init__(self, message: str, temporary: bool = False):
        super().__init__(message)
        self.temporary = temporary


class ProviderSkipped(Exception):
    """Not configured in .env: no network call is made."""


def _image_bytes(method: str, url: str, **kw) -> bytes:
    headers = {**HEADERS, **kw.pop("headers", {})}
    try:
        r = requests.request(method, url, timeout=TIMEOUT, headers=headers, **kw)
    except (requests.Timeout, requests.ConnectionError) as e:
        raise ImageError(f"{type(e).__name__}: network problem or timeout", temporary=True) from e
    except requests.RequestException as e:
        raise ImageError(f"{type(e).__name__}: {e}"[:150]) from e
    if r.status_code >= 400:
        raise ImageError(f"HTTP {r.status_code}: {r.text[:120]!r}", temporary=r.status_code in (429, 500, 502, 503, 504))
    data = r.content
    is_image = data[:3] == b"\xff\xd8\xff" or data[:4] == b"\x89PNG" or (data[:4] == b"RIFF" and data[8:12] == b"WEBP")
    if not is_image:
        raise ImageError(f"HTTP {r.status_code} but the body is not an image (starts with {data[:30]!r})")
    if len(data) < 2000:
        raise ImageError("the image is tiny, probably a placeholder")
    return data


def cloudflare(prompt: str, seed: int) -> bytes:
    """Your own Cloudflare Worker (FLUX schnell). The Worker takes only a prompt."""
    url, token = os.getenv("CF_WORKER_URL", "").strip(), os.getenv("CF_WORKER_TOKEN", "").strip()
    if not url or not token:
        raise ProviderSkipped("CF_WORKER_URL / CF_WORKER_TOKEN are empty in .env")
    return _image_bytes("POST", url, headers={"Authorization": f"Bearer {token}"}, json={"prompt": prompt})


def pollinations(prompt: str, seed: int) -> bytes:
    """gen.pollinations.ai with a key; otherwise the legacy keyless endpoint (slow, about 1 request per 15 s)."""
    key = os.getenv("POLLINATIONS_API_KEY", "").strip()
    params = {"width": 1024, "height": 1024, "seed": seed, "model": "flux"}
    if key:
        return _image_bytes("GET", f"https://gen.pollinations.ai/image/{quote(prompt, safe='')}",
                            headers={"Authorization": f"Bearer {key}"}, params=params)
    params["nologo"] = "true"
    return _image_bytes("GET", f"https://image.pollinations.ai/prompt/{quote(prompt, safe='')}", params=params)