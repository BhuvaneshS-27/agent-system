"""News per startup: GDELT DOC 2.0 (free, no key) and Tavily with topic=news, merged.

Both go through WebTools.search(topic="news", providers=[...]) so they get the cache, the retries, the
'retry' / 'fallback' events and the Stop button for free. Every article becomes
{"date": "YYYY-MM-DD", "title", "url", "domain", "via"}.

GDELT rules (docs checked 23 Oct 2026): one request per 5 seconds per IP address (we wait 6 s, like
scripts/check_keys.py), HTTP 429 when you are too fast, and a text/plain error or an EMPTY body instead of
JSON for a bad query or no match. An empty answer is a valid result (no news), not a failure.
GDELT is often slow, so it waits at most 12 s per attempt and has a circuit breaker: after 2 failed chains
in a row it is skipped for the rest of the run (WebTools.reset_breakers() starts each run clean).
On some networks Python's HTTPS request to GDELT never finishes its TLS handshake (PowerShell's does):
switch GDELT off with --no-gdelt (web.disabled.add("gdelt")); news then comes from Tavily alone.

Tavily is asked for the last 90 days, but its answer is not trusted to obey: find_news() drops every
article older than 90 days (and every article without a date) itself. In a real run all 10 articles for
one company were 5 to 11 months old although days=90 had been sent.
"""
from datetime import date, timedelta
from typing import Optional

import requests
from ddgs import DDGS
from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException

from core.cache import LIMITS
from research.errors import ProviderError, Skip, request_problem, status_problem
from research.pagedate import parse_any_date as parse_news_date
from research.search import clean
from research.verify import alnum, registrable_domain
from research.web import SearchOutcome

GDELT_GAP = 6.0                 # seconds between two GDELT requests
GDELT_READ_TIMEOUT = 12         # seconds to wait for GDELT's answer (it used to be 40, three times per company)
GDELT_BREAKER = 2               # failed chains in a row before GDELT is skipped for the rest of the run
NEWS_DAYS = 90
GDELT_MAX = 75
TAVILY_NEWS_MAX = 15
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; AgentSystemStudentProject/1.0)"}


def article(title, url, date_value, via, snippet="") -> Optional[dict]:
    url = str(url or "").strip()
    if not url.startswith(("http://", "https://")):
        return None
    d = parse_news_date(date_value)
    return {"date": d.isoformat() if d else "", "title": clean(title, 160) or "(no title)", "url": url,
            "domain": registrable_domain(url), "via": via, "snippet": clean(snippet, 200)}


# ------------------------------------------------------------------ provider 1: GDELT
def gdelt_news(w, query: str) -> list:
    w.pace("gdelt", GDELT_GAP)                                   # at most 1 request per 6 s, Stop can end the wait
    try:
        r = requests.get(f"{w.endpoint('gdelt')}/api/v2/doc/doc", headers=HEADERS, timeout=(5, GDELT_READ_TIMEOUT),
                         params={"query": query, "mode": "artlist", "format": "json", "sort": "datedesc",
                                 "maxrecords": GDELT_MAX, "timespan": f"{NEWS_DAYS}d"})
    except requests.RequestException as e:
        raise request_problem(e) from None
    status_problem(r)                                            # 429 -> retryable ProviderError
    w.usage.add("gdelt")
    body = (r.text or "").strip()
    if not body:
        return []                                                # GDELT answers an empty body when nothing matches
    try:
        data = r.json()
    except ValueError:
        slow = "limit requests" in body.lower() or "too many" in body.lower()
        raise ProviderError("GDELT answered with text, not JSON: " + clean(body, 70), retryable=slow) from None
    rows = (article(a.get("title"), a.get("url"), a.get("seendate"), "gdelt") for a in data.get("articles") or [])
    return [a for a in rows if a]


gdelt_news.breaker = GDELT_BREAKER                               # read by WebTools._chain_run


# ------------------------------------------------------------------ provider 2: Tavily topic=news
def tavily_news(w, query: str) -> list:
    key = w.key("TAVILY_API_KEY")
    used = w.usage.month("tavily")["credits"]
    if used >= LIMITS["tavily"]:
        raise Skip(f"the local counter says the monthly limit is used ({used:g} / {LIMITS['tavily']} credits)")
    headers = {"Authorization": f"Bearer {key}"}
    body = {"query": query, "topic": "news", "days": NEWS_DAYS, "max_results": TAVILY_NEWS_MAX,
            "search_depth": "basic", "start_date": (date.today() - timedelta(days=NEWS_DAYS)).isoformat()}
    try:
        data = w.post_json(f"{w.endpoint('tavily')}/search", headers, body)
    except ProviderError as e:
        if e.reason not in ("HTTP 400", "HTTP 422"):
            raise
        body.pop("start_date")                       # the docs disagree on days + start_date: ask once without it
        data = w.post_json(f"{w.endpoint('tavily')}/search", headers, body)
    w.count("tavily", credits=1)
    rows = (article(r.get("title"), r.get("url"), r.get("published_date"), "tavily", r.get("content"))
            for r in data.get("results") or [])
    return [a for a in rows if a]


# ------------------------------------------------------------------ provider 3: ddgs news (last resort)
def ddgs_news(w, query: str) -> list:
    try:
        rows = DDGS(timeout=10).news(query, max_results=TAVILY_NEWS_MAX)
    except RatelimitException:
        raise ProviderError("DuckDuckGo rate limit", True) from None
    except TimeoutException:
        raise ProviderError("timeout", True) from None
    except DDGSException as e:
        raise ProviderError(f"no results or blocked ({clean(e, 60)})", True) from None
    w.count("ddgs")
    out = (article(r.get("title"), r.get("url") or r.get("href"), r.get("date"), "ddgs", r.get("body"))
           for r in rows or [])
    return [a for a in out if a]


GDELT_CHAIN = [("gdelt", gdelt_news)]
TAVILY_CHAIN = [("tavily", tavily_news), ("ddgs", ddgs_news)]    # same names as the dashboard demo switches


def gdelt_query(name: str, category: str) -> str:
    """"Name" ("category" OR startup OR funding) sourcelang:english   (GDELT: quoted phrase, OR in brackets)."""
    terms = ([f'"{category.strip()}"'] if category.strip() else []) + ["startup", "funding"]
    return f'"{name.strip()}" ({" OR ".join(terms)}) sourcelang:english'


def in_window(art: dict, today: Optional[date] = None) -> bool:
    """True for an article of the last NEWS_DAYS days. No date or a future date: False / not older."""
    today = today or date.today()
    try:
        d = date.fromisoformat(art.get("date") or "")
    except ValueError:
        return False
    return d >= today - timedelta(days=NEWS_DAYS)


def find_news_detail(web, name: str, category: str, stop_event=None, tags: Optional[dict] = None,
                     today: Optional[date] = None):
    """Both news sources, merged, de-duplicated by URL and cut to the last 90 days.
    Returns (articles, ok, sources): ok is False only when BOTH sources failed (then 'no news data',
    which is not the same as 'zero articles'); sources lists the ones that answered: 'gdelt' and/or 'web'
    (Tavily or ddgs). research/scoring.py uses it to compare companies on the same sources only."""
    tags = tags if tags is not None else {}
    if "gdelt" in web.disabled:                       # --no-gdelt / dashboard switch: no request and no log noise
        a = SearchOutcome(False, "", error="GDELT is switched off")
    else:
        a = web.search(gdelt_query(name, category), max_results=GDELT_MAX, tags=tags, stop_event=stop_event,
                       topic="news", providers=GDELT_CHAIN)
    b = web.search(f"{name} {category} startup news", max_results=TAVILY_NEWS_MAX, tags=tags,
                   stop_event=stop_event, topic="news", providers=TAVILY_CHAIN)
    key, seen, merged = alnum(name), set(), []
    for out in (a, b):
        for art in out.results if out.ok else []:
            if art["url"] in seen or not in_window(art, today):
                continue
            # GDELT matched the exact phrase in the article text; the others are semantic, so they
            # must show the name in the headline or the snippet.
            if art.get("via") != "gdelt" and f" {key} " not in f" {alnum(art['title'] + ' ' + art.get('snippet', ''))} ":
                continue
            seen.add(art["url"])
            merged.append(art)
    merged.sort(key=lambda x: x["date"], reverse=True)
    return merged, (a.ok or b.ok), [fam for fam, out in (("gdelt", a), ("web", b)) if out.ok]


def find_news(web, name: str, category: str, stop_event=None, tags: Optional[dict] = None):
    """Same as find_news_detail, without the list of sources: (articles, ok)."""
    articles, ok, _ = find_news_detail(web, name, category, stop_event, tags)
    return articles, ok