"""Tests every API and service of the project and explains each problem.

    python scripts/check_keys.py                  run all checks
    python scripts/check_keys.py --no-generate    skip the two checks that generate an image (saves free quota)
    python scripts/check_keys.py --only tavily,exa
    python scripts/check_keys.py --list           show the check names
    python scripts/check_keys.py --only worker_status,worker_auth,worker

Each line shows: result, service, seconds, what happened (HTTP status and the provider's own
message), and for problems a concrete fix. No secret value is ever printed.
"""
import argparse
import os
import re
import sys
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import quote, urlparse

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
TIMEOUT = 20                      # seconds, for normal requests
GDELT_RETRY_WAIT = 6              # GDELT allows 1 request per 5 seconds

BASE = {                          # the tests can replace these addresses
    "tavily": "https://api.tavily.com",
    "exa": "https://api.exa.ai",
    "serper": "https://google.serper.dev",
    "jina": "https://r.jina.ai",
    "gdelt": "https://api.gdeltproject.org",
    "wikipedia": "https://en.wikipedia.org",
    "wikidata": "https://www.wikidata.org",
    "sec": "https://data.sec.gov",
    "pollinations": "https://gen.pollinations.ai",
    "magichour": "https://api.magichour.ai",
}

SECRET_NAMES = ["TAVILY_API_KEY", "EXA_API_KEY", "SERPER_API_KEY", "CF_WORKER_TOKEN",
                "POLLINATIONS_API_KEY", "MAGIC_HOUR_API_KEY"]
WHERE = {                         # service -> (variable in .env, where to create the key)
    "tavily": ("TAVILY_API_KEY", "app.tavily.com"),
    "exa": ("EXA_API_KEY", "dashboard.exa.ai (API Keys)"),
    "serper": ("SERPER_API_KEY", "serper.dev (dashboard)"),
    "pollinations": ("POLLINATIONS_API_KEY", "enter.pollinations.ai"),
    "magichour": ("MAGIC_HOUR_API_KEY", "magichour.ai (developer settings, API keys)"),
}
IMAGE_PROMPT = "simple blue owl emblem, flat vector style, centered, solid background, no text, no letters"


# ------------------------------------------------------------------ small helpers
class Skip(Exception):
    """Not configured or not installed yet."""


class Problem(Exception):
    def __init__(self, msg: str, fix: str = ""):
        super().__init__(msg)
        self.msg, self.fix = msg, fix


class Fail(Problem):
    """The service does not work."""


class Warn(Problem):
    """The service works, but something needs attention."""


def env(name: str) -> str:
    return (os.environ.get(name) or "").strip()


def need(name: str) -> str:
    value = env(name)
    if not value:
        raise Skip(f"{name} is empty in .env")
    return value


def redact(text: str) -> str:
    """Safety net: a secret can never appear in the output, even if a server echoes it back."""
    for name in SECRET_NAMES:
        value = env(name)
        if len(value) >= 6:
            text = text.replace(value, f"***{name}***")
    return text


def host(url) -> str:
    return urlparse(str(url or "")).netloc or "unknown site"


def dig(obj, depth=0) -> str:
    """Finds the human-readable message inside an error answer of any shape."""
    if depth > 4:
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        for key in ("message", "error", "detail", "details", "msg", "title", "reason"):
            if key in obj:
                found = dig(obj[key], depth + 1)
                if found:
                    return found
    if isinstance(obj, list):
        for item in obj:
            found = dig(item, depth + 1)
            if found:
                return found
    return ""


def vendor_message(r: requests.Response) -> str:
    try:
        j = r.json()
        if isinstance(j, dict) and isinstance(j.get("error"), str) and isinstance(j.get("details"), str):
            msg = f"{j['error']} - {j['details']}"
        else:
            msg = dig(j)
    except ValueError:
        msg = ""
    if not msg:
        msg = " ".join(re.sub(r"<[^>]+>", " ", r.text or "").split())
    return (msg[:180] + "...") if len(msg) > 180 else (msg or "(empty answer)")


def scalars(obj, allow) -> str:
    """'name=value' for the fields in the allow-list only (numbers, flags, short words).
    Anything else, such as an e-mail address, an account name or a key, is never shown."""
    parts = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in allow and isinstance(v, (bool, int, float, str)) and len(str(v)) <= 24:
                parts.append(f"{k}={v}")
    return ", ".join(parts)


POLLINATIONS_KEY_FIELDS = ("valid", "type", "rateLimitEnabled", "questPollenOnly")
POLLINATIONS_BALANCE_FIELDS = ("balance",)
MAGIC_HOUR_FIELDS = ("tier", "credits", "total", "status")


def status_fix(status: int, service: str = "", text: str = "") -> str:
    env_name, where = WHERE.get(service, ("", ""))
    if status == 401 or (status == 403 and env_name):
        if env_name:
            return (f"{env_name} is wrong, revoked or belongs to another service. Create a new key at {where} "
                    f"and paste it into {env_name} in .env (no quotes, no spaces). "
                    + ("If the key is certainly right, your plan or network is being refused." if status == 403 else ""))
        return "Authentication failed: the key is missing or wrong."
    if status == 402:
        if service == "pollinations":
            return ("Your Pollinations 'pollen' balance is empty. It refills over time (see enter.pollinations.ai); "
                    "until then the Cloudflare Worker is the main image provider.")
        return "Out of credits or balance. Check the provider's dashboard; the chain falls back to the next provider."
    if status == 403:
        return "The key is valid but not allowed to do this (plan, permissions, region), or this network is blocked."
    if status == 404:
        return "Address not found. The API may have changed, or the address in .env is wrong."
    if status == 405:
        return "Wrong HTTP method for this address."
    if status in (408, 504, 524):
        return "The service timed out. Try again in a minute."
    if status in (400, 422):
        return "The service rejected the request format. Its API may have changed: send me this line."
    if status == 429:
        return "Rate limit or free quota used up. Wait a minute, then look at the provider's usage page."
    if status >= 500:
        return "The provider has a problem on its side. Try again later; the chain falls back to the next provider."
    return ""


def http(method: str, url: str, **kw) -> requests.Response:
    """requests.request with every network failure translated into a clear sentence."""
    timeout = kw.pop("timeout", None) or TIMEOUT
    headers = {"User-Agent": "AgentSystemStudentProject/1.0", **kw.pop("headers", {})}
    try:
        return requests.request(method, url, headers=headers, timeout=timeout, **kw)
    except requests.exceptions.ProxyError:
        raise Fail("a proxy refused the connection", "Your network or proxy blocks this site. Try a phone hotspot.")
    except requests.exceptions.SSLError:
        raise Fail("HTTPS certificate error", "An antivirus or proxy may be intercepting HTTPS. Try another network.")
    except requests.exceptions.ConnectTimeout:
        raise Fail(f"could not connect within {timeout} s",
                   "The site may be blocked on this network (college Wi-Fi?). Try a phone hotspot.")
    except requests.exceptions.ReadTimeout:
        raise Fail(f"connected, but no answer within {timeout} s",
                   "The service is slow or overloaded. Run the check again in a minute.")
    except requests.exceptions.ConnectionError as e:
        text = str(e)
        if re.search(r"getaddrinfo|Name or service|NameResolution|Temporary failure|nodename", text, re.I):
            raise Fail("DNS lookup failed: the address does not exist or there is no internet",
                       f"Check the spelling of the address ({host(url)}) and your internet connection.")
        if re.search(r"refused|actively refused", text, re.I):
            raise Fail("connection refused: nothing is listening at this address",
                       f"Is the service running? Address: {host(url)}")
        raise Fail("could not connect (no internet, or the site is blocked)", "Try a phone hotspot.")
    except requests.exceptions.RequestException as e:
        raise Fail(f"request failed: {type(e).__name__}", "Run the check again; if it repeats, send me this line.")


def http_text(r: requests.Response, text: str) -> str:
    return f"HTTP {r.status_code}" + (f" {r.reason}" if r.reason else "") + f": {text}"


def need_ok(r: requests.Response, service: str) -> requests.Response:
    if r.status_code >= 400:
        text = vendor_message(r)
        raise Fail(http_text(r, text), status_fix(r.status_code, service, text))
    return r


def need_json(r: requests.Response, what: str):
    try:
        return r.json()
    except ValueError:
        snippet = " ".join((r.text or "")[:120].split())
        raise Fail(f"{what} did not answer with JSON: {snippet!r}",
                   "The service may be overloaded, or its API has changed.")


def save_image(r: requests.Response, filename: str) -> str:
    data, ctype = r.content, r.headers.get("Content-Type", "")
    if data[:3] == b"\xff\xd8\xff":
        kind = "JPEG"
    elif data[:4] == b"\x89PNG":
        kind = "PNG"
    elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        kind = "WEBP"
    else:
        raise Fail(f"HTTP 200 but the body is not an image (Content-Type {ctype!r}, starts with {data[:40]!r})",
                   "The service answered with something else, for example an error page.")
    if len(data) < 2000:
        raise Fail(f"the {kind} is only {len(data)} bytes: probably a placeholder, not a real image",
                   "Look at the saved file; the service may have refused the prompt.")
    CACHE.mkdir(exist_ok=True)
    (CACHE / filename).write_bytes(data)
    return f"{kind} image, {len(data) / 1024:.0f} KB, saved as cache/{filename}"


# ------------------------------------------------------------------------ checks
def check_llm():
    url = (env("LLM_URL") or "http://127.0.0.1:8081").rstrip("/")
    try:
        r = requests.get(f"{url}/health", timeout=3)
    except requests.RequestException:
        raise Fail(f"nothing answers at {url}", "Start the model server in another window:  python scripts/start_llm.py")
    if r.status_code == 200:
        return f"model loaded and ready at {url}"
    if r.status_code == 503:
        raise Warn("the server is up but the model is still loading", "Wait a minute and run the check again.")
    raise Fail(f"HTTP {r.status_code} from {url}/health", "Another program may be using this port.")


def check_tavily():
    key = need("TAVILY_API_KEY")
    r = http("POST", f"{BASE['tavily']}/search", headers={"Authorization": f"Bearer {key}"},
             json={"query": "top fintech startups 2026", "max_results": 1})
    need_ok(r, "tavily")
    results = need_json(r, "Tavily").get("results") or []
    if not results:
        raise Warn("the key works but the search returned no results", "Run the check again.")
    return f"1 result from {host(results[0].get('url'))} (used 1 of 1,000 monthly credits)"


def check_exa():
    key = need("EXA_API_KEY")
    r = http("POST", f"{BASE['exa']}/search", headers={"x-api-key": key},
             json={"query": "top fintech startups 2026", "numResults": 1})
    need_ok(r, "exa")
    results = need_json(r, "Exa").get("results") or []
    if not results:
        raise Warn("the key works but the search returned no results", "Run the check again.")
    return f"1 result from {host(results[0].get('url'))}"


def check_serper():
    key = need("SERPER_API_KEY")
    r = http("POST", f"{BASE['serper']}/search", headers={"X-API-KEY": key},
             json={"q": "top fintech startups 2026", "num": 1})
    need_ok(r, "serper")
    results = need_json(r, "Serper").get("organic") or []
    if not results:
        raise Warn("the key works but the search returned no results", "Run the check again.")
    return f"1 result from {host(results[0].get('link'))} (used 1 of 2,500 one-time queries)"


def check_ddgs():
    try:
        from ddgs import DDGS
    except ImportError:
        raise Skip("package 'ddgs' is not installed yet (Step 6: pip install ddgs)")
    try:
        results = list(DDGS().text("top fintech startups 2026", max_results=3))
    except Exception as e:
        raise Warn(f"{type(e).__name__}: {str(e)[:150]}",
                   "ddgs is unofficial and often rate-limited. It is only the last fallback, so this is acceptable.")
    if not results:
        raise Warn("returned an empty list", "Normal for ddgs (rate limit). It is only the last fallback.")
    return f"{len(results)} results"


def check_trafilatura():
    try:
        import trafilatura
    except ImportError:
        raise Skip("package 'trafilatura' is not installed yet (Step 6: pip install trafilatura)")
    html = trafilatura.fetch_url(f"{BASE['wikipedia']}/wiki/Financial_technology")
    if not html:
        raise Fail("could not download the test page", "Check your internet connection, or whether Wikipedia is blocked here.")
    text = trafilatura.extract(html) or ""
    if len(text) < 500:
        raise Fail(f"extracted only {len(text)} characters of text", "Run:  pip install -U trafilatura")
    return f"extracted {len(text):,} characters of article text"


def check_jina():
    r = http("GET", f"{BASE['jina']}/https://example.com", timeout=40)
    need_ok(r, "jina")
    if "Example Domain" not in r.text:
        raise Warn("answered, but the text of the test page is missing", "Jina may have changed its output; try again.")
    return "read example.com as clean text (keyless tier: 20 requests per minute)"


def check_gdelt():
    def ask():
        return http("GET", f"{BASE['gdelt']}/api/v2/doc/doc", timeout=40,
                    params={"query": "fintech", "mode": "artlist", "format": "json", "maxrecords": 3})

    r = ask()
    if r.status_code == 429:                  # asked too soon: wait as requested and try once more
        time.sleep(GDELT_RETRY_WAIT)
        r = ask()
    if r.status_code == 429:
        raise Warn("GDELT is reachable but is rate-limiting this connection (HTTP 429: 1 request per 5 seconds, "
                   "counted per IP address, so other users on the same network count too)",
                   "Not a key problem. Run the check again in a minute. The agent will space GDELT calls 6 seconds "
                   "apart and fall back to Tavily with topic=news.")
    need_ok(r, "gdelt")
    try:
        j = r.json()
    except ValueError:
        text = " ".join((r.text or "")[:150].split())
        raise Fail(f"GDELT answered with text instead of JSON: {text!r}",
                   "GDELT allows about 1 request per 5 seconds and is often slow. Wait 10 seconds and retry; "
                   "Tavily with topic=news is the fallback.")
    articles = j.get("articles") or []
    if not articles:
        raise Warn("no articles returned", "Run the check again; GDELT sometimes answers empty.")
    return f"{len(articles)} articles, the first from {articles[0].get('domain', 'unknown')}"


def contact_agent() -> str:
    return env("SEC_USER_AGENT") or "AgentSystemStudentProject/1.0 (student project)"


def check_wikipedia():
    r = http("GET", f"{BASE['wikipedia']}/api/rest_v1/page/summary/Financial_technology",
             headers={"User-Agent": contact_agent()})
    need_ok(r, "wikipedia")
    extract = need_json(r, "Wikipedia").get("extract") or ""
    if not extract:
        raise Warn("answered without text", "Run the check again.")
    return f"summary of 'Financial technology' ({len(extract)} characters)"


def check_wikidata():
    r = http("GET", f"{BASE['wikidata']}/w/api.php", headers={"User-Agent": contact_agent()},
             params={"action": "wbsearchentities", "search": "Stripe", "language": "en", "format": "json", "limit": 1})
    need_ok(r, "wikidata")
    found = need_json(r, "Wikidata").get("search") or []
    if not found:
        raise Warn("no entity found for 'Stripe'", "Run the check again.")
    return f"found {found[0].get('id')} ({found[0].get('label')})"


def check_sec():
    agent = contact_agent()
    r = http("GET", f"{BASE['sec']}/submissions/CIK0000320193.json", headers={"User-Agent": agent})
    if r.status_code == 403:
        raise Fail("HTTP 403: SEC EDGAR blocked the request",
                   "SEC requires a real contact in the User-Agent. Set SEC_USER_AGENT=YourName your-real-email in .env.")
    need_ok(r, "sec")
    name = need_json(r, "SEC EDGAR").get("name", "unknown")
    if "your-email@example.com" in agent or agent.startswith("TeamName"):
        raise Warn(f"works (test company: {name}) but SEC_USER_AGENT still holds the placeholder text",
                   "Set SEC_USER_AGENT=YourName your-real-email in .env before the demo; SEC may block anonymous agents.")
    return f"filings of '{name}' readable"


def worker_cfg():
    url, token = env("CF_WORKER_URL"), env("CF_WORKER_TOKEN")
    missing = [n for n, v in (("CF_WORKER_URL", url), ("CF_WORKER_TOKEN", token)) if not v]
    if missing:
        raise Skip(f"{' and '.join(missing)} {'is' if len(missing) == 1 else 'are'} empty in .env")
    if not url.startswith(("https://", "http://")):
        raise Fail("CF_WORKER_URL must start with https://", "Example: https://image-api.your-name.workers.dev")
    return url, token


def worker_fix(status: int, text: str) -> str:
    t = text.lower()
    if status == 401:
        return ("CF_WORKER_TOKEN is not accepted. It must be exactly the same text as the Worker secret IMAGE_API_KEY "
                "(Cloudflare dashboard > Workers > your Worker > Settings > Variables and Secrets). "
                "After changing a secret, deploy the Worker again.")
    if "5006" in t or "unevaluated properties" in t or "required property" in t:
        return ("The Worker sent the model a field it does not accept (error 5006, schema check). Open the Worker code: "
                "env.AI.run must receive only the documented fields, which are prompt and optionally steps (1-8).")
    if "binding" in t or "reading 'run'" in t:
        return "The Worker has no Workers AI binding. Add one: Worker > Settings > Bindings > Add > Workers AI, variable name AI, then deploy."
    if "daily free allocation" in t or "neurons" in t or "4006" in t:
        return "The free daily allowance (10,000 neurons) is used up. It resets at 00:00 UTC. Pollinations is the fallback until then."
    if "1010" in t or "browser_signature" in t:
        return "Cloudflare's security rules blocked this client (error 1010). Turn off Browser Integrity Check for the Worker's route."
    if "1101" in t or "1102" in t or "worker threw" in t:
        return "The Worker crashed with an uncaught exception. Open the Worker > Observability > Logs for the stack trace."
    if status == 404:
        return "No Worker at this address. Copy CF_WORKER_URL from the Worker's overview page and check that it is deployed."
    return status_fix(status, "", text)


def worker_fail(r: requests.Response):
    text = vendor_message(r)
    return Fail(http_text(r, text), worker_fix(r.status_code, text))


def check_worker_status():
    url, token = worker_cfg()
    r = http("GET", url, headers={"Authorization": f"Bearer {token}"})
    if r.status_code == 405:
        return "reachable (older Worker version without a status endpoint; the image test checks the token)"
    if r.status_code >= 400:
        raise worker_fail(r)
    j = need_json(r, "Worker")
    if not j.get("ai_binding"):
        raise Fail("the Worker runs and accepts the token, but has no Workers AI binding named AI", worker_fix(500, "binding"))
    return f"address and token accepted, AI binding present, model {j.get('model', '?')} (costs no neurons)"


def check_worker_auth():
    url, _ = worker_cfg()
    r = http("POST", url, json={"prompt": "authentication test"}, timeout=60)
    if r.status_code == 401:
        return "a request without a token is rejected (good)"
    if r.status_code == 200:
        raise Warn("the Worker generated an image WITHOUT any token: anyone who finds the URL can use up your free allowance",
                   "Set the secret IMAGE_API_KEY on the Worker (Settings > Variables and Secrets) and deploy again.")
    raise worker_fail(r)


def check_worker():
    url, token = worker_cfg()
    r = http("POST", url, headers={"Authorization": f"Bearer {token}"}, json={"prompt": IMAGE_PROMPT}, timeout=120)
    if r.status_code >= 400:
        raise worker_fail(r)
    return save_image(r, "check_worker.jpg")


def check_pollinations_key():
    key = need("POLLINATIONS_API_KEY")
    auth = {"Authorization": f"Bearer {key}"}
    r = http("GET", f"{BASE['pollinations']}/account/key", headers=auth)
    need_ok(r, "pollinations")
    info = scalars(need_json(r, "Pollinations"), POLLINATIONS_KEY_FIELDS)
    detail = "key accepted" + (f" ({info})" if info else "")
    try:
        b = http("GET", f"{BASE['pollinations']}/account/balance", headers=auth)
        if b.status_code == 200:
            bal = scalars(b.json(), POLLINATIONS_BALANCE_FIELDS)
            if bal:
                detail += f"; balance: {bal}"
    except (Problem, ValueError):
        pass
    return detail


def check_pollinations():
    key = need("POLLINATIONS_API_KEY")
    r = http("GET", f"{BASE['pollinations']}/image/{quote(IMAGE_PROMPT, safe='')}",
             headers={"Authorization": f"Bearer {key}"}, params={"width": 512, "height": 512}, timeout=120)
    need_ok(r, "pollinations")
    return save_image(r, "check_pollinations.jpg")


def check_magichour():
    key = need("MAGIC_HOUR_API_KEY")
    r = http("GET", f"{BASE['magichour']}/v1/account", headers={"Authorization": f"Bearer {key}"})
    need_ok(r, "magichour")
    info = need_json(r, "Magic Hour")
    parts = [scalars(info, MAGIC_HOUR_FIELDS)] + [scalars(v, MAGIC_HOUR_FIELDS) for v in info.values() if isinstance(v, dict)]
    extra = "; ".join(p for p in parts if p)
    return "key accepted" + (f" ({extra})" if extra else "") + ", no credits used"


# ------------------------------------------------------------------- the check list
@dataclass
class Check:
    name: str
    title: str
    chain: str
    provider: str
    func: Callable
    generates: bool = False       # uses image-generation quota
    counts: bool = True           # counts as a usable provider in the chain summary


CHAINS = {                        # key -> (heading, required?)
    "model": ("Local model server", True),
    "search": ("Web search chain: Tavily > Exa > Serper > ddgs", True),
    "fetch": ("Page fetch chain: trafilatura > Jina Reader", True),
    "news": ("News: GDELT", False),
    "facts": ("Facts cross-check: Wikipedia, Wikidata, SEC EDGAR", False),
    "image": ("Image chain: your Cloudflare Worker > Pollinations", True),
    "video": ("Video intro, optional: Magic Hour", False),
}

CHECKS = [
    Check("llm", "llama-server (local model)", "model", "llama-server", check_llm),
    Check("tavily", "Tavily search", "search", "Tavily", check_tavily),
    Check("exa", "Exa search", "search", "Exa", check_exa),
    Check("serper", "Serper search", "search", "Serper", check_serper),
    Check("ddgs", "ddgs search (last fallback)", "search", "ddgs", check_ddgs),
    Check("trafilatura", "trafilatura page fetch", "fetch", "trafilatura", check_trafilatura),
    Check("jina", "Jina Reader", "fetch", "Jina Reader", check_jina),
    Check("gdelt", "GDELT news", "news", "GDELT", check_gdelt),
    Check("wikipedia", "Wikipedia", "facts", "Wikipedia", check_wikipedia),
    Check("wikidata", "Wikidata", "facts", "Wikidata", check_wikidata),
    Check("sec", "SEC EDGAR", "facts", "SEC EDGAR", check_sec),
    Check("worker_status", "Worker: address, token, AI binding", "image", "", check_worker_status, counts=False),
    Check("worker_auth", "Worker: rejects missing token", "image", "", check_worker_auth, counts=False),
    Check("worker", "Worker: FLUX image", "image", "Cloudflare Worker", check_worker, generates=True),
    Check("pollinations_key", "Pollinations: key and balance", "image", "", check_pollinations_key, counts=False),
    Check("pollinations", "Pollinations: image", "image", "Pollinations", check_pollinations, generates=True),
    Check("magichour", "Magic Hour: key and credits", "video", "Magic Hour", check_magichour),
]


@dataclass
class Result:
    check: Check
    status: str                   # OK, WARN, FAIL, SKIP
    seconds: float
    detail: str
    fix: str = ""


def run_one(c: Check) -> Result:
    started = time.time()
    try:
        detail = c.func()
        return Result(c, "OK", time.time() - started, detail)
    except Skip as e:
        return Result(c, "SKIP", 0.0, str(e))
    except Warn as e:
        return Result(c, "WARN", time.time() - started, e.msg, e.fix)
    except Fail as e:
        return Result(c, "FAIL", time.time() - started, e.msg, e.fix)
    except Exception as e:        # a bug in this script must not hide the other results
        return Result(c, "FAIL", time.time() - started, f"unexpected {type(e).__name__}: {e}",
                      "This is a bug in the check script. Send me this line.")


def show(res: Result) -> None:
    tag = {"OK": "[ OK ]", "WARN": "[WARN]", "FAIL": "[FAIL]", "SKIP": "[SKIP]"}[res.status]
    secs = f"{res.seconds:4.1f}s" if res.status != "SKIP" else "     "
    head = f"{tag} {res.check.title:<36} {secs}  "
    lines = textwrap.wrap(redact(res.detail), width=66) or [""]
    print(head + lines[0])
    for line in lines[1:]:
        print(" " * len(head) + line)
    if res.fix:
        for line in textwrap.wrap("fix: " + redact(res.fix), width=108, initial_indent="       ",
                                  subsequent_indent="            "):
            print(line)


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Test every API and service of the project.")
    ap.add_argument("--only", help="comma-separated check names, e.g. tavily,exa")
    ap.add_argument("--no-generate", action="store_true", help="skip the checks that generate an image")
    ap.add_argument("--list", action="store_true", help="list the check names and exit")
    args = ap.parse_args(argv)

    if args.list:
        for c in CHECKS:
            print(f"{c.name:<18} {c.title}" + ("   [generates an image]" if c.generates else ""))
        return 0

    load_dotenv(ROOT / ".env", override=True)
    if not (ROOT / ".env").exists():
        print("WARNING: no .env file found; every key will show as empty.\n")

    selected = CHECKS
    if args.only:
        wanted = [w.strip() for w in args.only.split(",") if w.strip()]
        unknown = [w for w in wanted if w not in {c.name for c in CHECKS}]
        if unknown:
            print(f"Unknown check name(s): {', '.join(unknown)}.  Use --list to see the names.")
            return 2
        selected = [c for c in CHECKS if c.name in wanted]
    skipped_generate = []
    if args.no_generate:
        skipped_generate = [c for c in selected if c.generates]
        selected = [c for c in selected if not c.generates]

    print(f"Running {len(selected)} checks in parallel (can take up to a minute)...\n")
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(run_one, selected))

    for chain, (heading, _required) in CHAINS.items():
        group = [r for r in results if r.check.chain == chain]
        if not group:
            continue
        print(f"== {heading} ==")
        for r in group:
            show(r)
        print()

    counts = {s: sum(1 for r in results if r.status == s) for s in ("OK", "WARN", "FAIL", "SKIP")}
    print(f"SUMMARY: {counts['OK']} OK | {counts['WARN']} WARN | {counts['FAIL']} FAIL | {counts['SKIP']} SKIP"
          + (f" | {len(skipped_generate)} not run (--no-generate)" if skipped_generate else ""))
    print("\nWhat the agent can really use:")
    for chain, (heading, required) in CHAINS.items():
        providers = [r for r in results if r.check.chain == chain and r.check.counts]
        if not providers:
            continue
        working = [r.check.provider for r in providers if r.status == "OK"]
        warned = [r.check.provider for r in providers if r.status == "WARN"]
        label = heading.split(":")[0]
        if working:
            print(f"  {label:<22} usable   {len(working)} of {len(providers)}: {', '.join(working)}")
        elif warned:
            print(f"  {label:<22} degraded (WARN): {', '.join(warned)}")
        else:
            print(f"  {label:<22} {'NOT USABLE' if required else 'not available (optional)'}")

    fails = [r for r in results if r.status == "FAIL"]
    if fails:
        print("\nFix these first:")
        for i, r in enumerate(fails, 1):
            print(f"  {i}. {r.check.title}")
    print("\nNo secret value is printed by this script. Images saved by the checks are in the cache folder.")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())