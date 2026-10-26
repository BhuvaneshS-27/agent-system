"""WebTools: web search and page fetch with a cache, retries and a provider fallback chain.

How one call works (this is what you explain to the professor):
  1. look in the SQLite cache (key = tool + normalised query or URL); a hit makes no network call
  2. try the providers in order; a provider is SKIPPED (no key, switched off, free quota used up)
     or TRIED with up to 3 attempts (tenacity, waits 2 s then 4 s) when the failure is temporary
     (timeout, connection error, HTTP 429, HTTP 5xx)
  3. when a provider gives up, a 'fallback' event with the reason is logged and the next one runs
  4. when every provider fails, an old cached copy (any age) is used if there is one
  5. every step is an event on the bus: web_start, cache_hit, provider_try, retry, fallback,
     stale_cache, web_result or web_failed, and 'credits' after each counted call

Providers are plain functions fn(web, query_or_url) in research/search.py and research/fetch.py.
They return a result, raise Skip (not tried) or raise ProviderError (tried and failed).
"""
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential

from core.cache import Usage, WebCache, normalise_query, normalise_url
from research import fetch as fetch_mod
from research import search as search_mod
from research.errors import ProviderError, Skip, WebStopped, request_problem, status_problem
from research.pagedate import best_date

log = logging.getLogger("web")

SEARCH_TTL = 24 * 3600            # a cached search is "fresh" for one day
FETCH_TTL = 7 * 24 * 3600         # a cached page for one week
MAX_TRIES = 3                     # attempts per provider for temporary failures
DEFAULT_ENDPOINTS = {
    "tavily": "https://api.tavily.com",
    "exa": "https://api.exa.ai",
    "serper": "https://google.serper.dev",
    "jina": "https://r.jina.ai",
    "gdelt": "https://api.gdeltproject.org",
}


def cached_date(value: dict) -> str:
    """Publication date of a cached page. Pages cached before Step 7b have no 'date' key: use the URL's date."""
    return value["date"] if "date" in value else best_date("", value.get("url", ""))


class ChainFailed(Exception):
    def __init__(self, trail, attempts):
        super().__init__("; ".join(f"{n}: {r}" for n, r in trail))
        self.trail, self.attempts = trail, attempts


@dataclass
class SearchOutcome:
    ok: bool
    query: str
    results: list = field(default_factory=list)       # [{"title", "url", "snippet"}]
    provider: str = ""                                # "tavily", or "cache (tavily)"
    cached: bool = False
    stale: bool = False                               # an old copy used because every provider failed
    attempts: int = 0
    seconds: float = 0.0
    error: str = ""
    age_s: float = 0.0
    tool: str = "web_search"                          # "news_search" for topic="news"


@dataclass
class FetchOutcome:
    ok: bool
    url: str
    title: str = ""
    text: str = ""
    provider: str = ""
    cached: bool = False
    stale: bool = False
    attempts: int = 0
    seconds: float = 0.0
    error: str = ""
    age_s: float = 0.0
    date: str = ""                                    # publication date of the page ('YYYY-MM-DD'), '' if unknown


class WebTools:
    def __init__(self, bus, cache_dir, env=None, sleep=None, retry_wait=None,
                 allow_private: bool = False, timeout=(5, 25)):
        """env: where the API keys are read from (default os.environ, filled from .env).
        sleep / retry_wait / allow_private exist for the tests."""
        self.bus = bus
        self.env = env if env is not None else os.environ
        cache_dir = Path(cache_dir)
        self.cache = WebCache(cache_dir / "web_cache.sqlite")
        self.usage = Usage(cache_dir / "usage.json")
        self.disabled: set = set()                    # provider names switched off in the dashboard
        self.endpoints = dict(DEFAULT_ENDPOINTS)
        self.allow_private = allow_private
        self.timeout = timeout
        self._sleep = sleep
        self._wait = retry_wait or wait_exponential(multiplier=2, min=2, max=8)
        self._local = threading.local()               # the Stop event of the call that is running now
        self._pace_lock = threading.Lock()
        self._last_call: dict = {}                    # provider -> time of its last request (for pace)
        self._fails: dict = {}                        # provider -> failed chains in a row (for the circuit breaker)

    # ------------------------------------------------------------ helpers for the providers
    def key(self, name: str) -> str:
        value = (self.env.get(name) or "").strip()
        if not value:
            raise Skip(f"no {name} in .env")
        return value

    def endpoint(self, name: str) -> str:
        return self.endpoints[name].rstrip("/")

    def count(self, provider: str, **kw) -> None:
        self.usage.add(provider, **kw)
        self._emit({}, "credits", provider=provider, line=self.usage.lines()[provider])

    def post_json(self, url: str, headers: dict, body: dict) -> dict:
        try:
            r = requests.post(url, headers=headers, json=body, timeout=self.timeout)
        except requests.RequestException as e:
            raise request_problem(e) from None
        status_problem(r)
        try:
            return r.json()
        except ValueError:
            raise ProviderError("the answer was not JSON") from None

    def pace(self, name: str, gap_s: float) -> None:
        """Keep at least gap_s seconds between two requests to one provider (GDELT: 1 request per
        5 s per IP). Called by a provider just before it sends. The wait can be ended by Stop."""
        stop_event = getattr(self._local, "stop", None)
        with self._pace_lock:                         # other threads queue up behind this one
            wait = self._last_call.get(name, float("-inf")) + gap_s - time.monotonic()
            if wait > 0:
                self._sleeper(stop_event)(wait)
            self._last_call[name] = time.monotonic()

    def reset_breakers(self) -> None:
        """Forget earlier failures. A provider whose function has a 'breaker' number (research/news.py: GDELT)
        is skipped for the rest of a run after that many failed chains in a row; each run starts clean."""
        self._fails.clear()

    def _emit(self, tags: dict, kind: str, **data) -> None:
        if self.bus:
            self.bus.emit(kind, **{**tags, **data})

    # ------------------------------------------------------------ the chain
    def _sleeper(self, stop_event):
        if self._sleep:
            return self._sleep

        def sleep(seconds: float) -> None:            # Stop works during the wait between attempts
            if stop_event is not None:
                if stop_event.wait(seconds):
                    raise WebStopped()
            else:
                time.sleep(seconds)
        return sleep

    def _attempt(self, tool, name, fn, arg, tags, stop_event, counter):
        def before(state) -> None:
            if stop_event is not None and stop_event.is_set():
                raise WebStopped()
            counter[0] += 1
            self._emit(tags, "provider_try", tool=tool, provider=name, attempt=state.attempt_number)

        def before_sleep(state) -> None:
            self._emit(tags, "retry", tool=tool, provider=name, attempt=state.attempt_number,
                       reason=state.outcome.exception().reason, wait_s=round(state.next_action.sleep, 1))

        def guarded():
            try:
                return fn(self, arg)
            except (Skip, ProviderError, WebStopped):
                raise
            except Exception as e:                    # a bug or a changed answer format: never crash the chain
                log.exception("provider %s failed unexpectedly", name)
                raise ProviderError(f"unexpected {type(e).__name__}: {str(e)[:80]}") from None

        retrying = Retrying(stop=stop_after_attempt(MAX_TRIES), wait=self._wait, reraise=True,
                            retry=retry_if_exception(lambda e: isinstance(e, ProviderError) and e.retryable),
                            before=before, before_sleep=before_sleep, sleep=self._sleeper(stop_event))
        return retrying(guarded)

    def _chain(self, tool, providers, arg, tags, stop_event):
        self._local.stop = stop_event
        try:
            return self._chain_run(tool, providers, arg, tags, stop_event)
        finally:
            self._local.stop = None

    def _chain_run(self, tool, providers, arg, tags, stop_event):
        trail, counter = [], [0]
        for i, (name, fn) in enumerate(providers):
            nxt = providers[i + 1][0] if i + 1 < len(providers) else None
            try:
                if name in self.disabled:
                    raise Skip("switched off with the dashboard demo switch")
                limit = getattr(fn, "breaker", None)
                if limit and self._fails.get(name, 0) >= limit:
                    raise Skip(f"circuit open: {name} failed {self._fails[name]} times in a row in this run")
                result = self._attempt(tool, name, fn, arg, tags, stop_event, counter)
                self._fails[name] = 0
                return name, result, counter[0], trail
            except Skip as e:
                reason, skipped = str(e), True
            except ProviderError as e:
                self._fails[name] = self._fails.get(name, 0) + 1
                reason, skipped = e.reason, False
            trail.append((name, reason))
            self._emit(tags, "fallback", tool=tool, provider=name, next=nxt, reason=reason, skipped=skipped)
        raise ChainFailed(trail, counter[0])

    # ------------------------------------------------------------ search
    def search(self, query: str, max_results: int = 5, fresh: bool = False,
               tags: Optional[dict] = None, stop_event=None, topic: str = "general",
               providers: Optional[list] = None) -> SearchOutcome:
        """fresh=True ignores the cache (the new answer is still saved).
        topic / providers: used by the news search (research/news.py). With providers=None the normal
        chain runs (Tavily, Exa, Serper, ddgs). A custom provider list gets its own cache entries,
        and a provider may return an empty list without that counting as a failure."""
        tags = tags if tags is not None else {}
        tool, q, t0 = ("web_search" if topic == "general" else f"{topic}_search"), \
            " ".join(str(query or "").split()), time.time()
        if not q:
            return SearchOutcome(False, "", error="the query is empty")

        def done(o: SearchOutcome) -> SearchOutcome:
            o.tool = tool
            return self._search_done(tags, o)

        chain = providers if providers is not None else search_mod.PROVIDERS
        key = normalise_query(q) if providers is None else "+".join(n for n, _ in chain) + "|" + normalise_query(q)
        self._emit(tags, "web_start", tool=tool, query=q[:200], fresh=fresh)
        if not fresh:
            hit = self.cache.get(tool, key, SEARCH_TTL)
            if hit:
                self._emit(tags, "cache_hit", tool=tool, query=q[:200], provider=hit.provider, age_s=round(hit.age_s))
                return done(SearchOutcome(
                    True, q, hit.value["results"][:max_results], f"cache ({hit.provider})", cached=True,
                    seconds=time.time() - t0, age_s=hit.age_s))
        try:
            name, rows, attempts, _ = self._chain(tool, chain, q, tags, stop_event)
        except ChainFailed as e:
            old = self.cache.get(tool, key, None)
            if old:
                self._emit(tags, "stale_cache", tool=tool, query=q[:200], age_s=round(old.age_s),
                           reason="every live provider failed")
                return done(SearchOutcome(
                    True, q, old.value["results"][:max_results], f"cache ({old.provider})", cached=True,
                    stale=True, attempts=e.attempts, seconds=time.time() - t0, age_s=old.age_s))
            self._emit(tags, "web_failed", tool=tool, query=q[:200], trail=[list(x) for x in e.trail])
            return SearchOutcome(False, q, attempts=e.attempts, seconds=round(time.time() - t0, 2),
                                 error="every provider failed: " + str(e))
        self.cache.put(tool, key, {"results": rows}, name)
        return done(SearchOutcome(True, q, rows[:max_results], name, attempts=attempts,
                                  seconds=time.time() - t0))

    def _search_done(self, tags, out: SearchOutcome) -> SearchOutcome:
        out.seconds = round(out.seconds, 2)
        self._emit(tags, "web_result", tool=out.tool, query=out.query[:200], provider=out.provider,
                   cached=out.cached, stale=out.stale, results=len(out.results), attempts=out.attempts,
                   seconds=out.seconds)
        return out

    # ------------------------------------------------------------ fetch
    def fetch(self, url: str, fresh: bool = False, tags: Optional[dict] = None, stop_event=None) -> FetchOutcome:
        tags = tags if tags is not None else {}
        tool, url, t0 = "fetch_page", str(url or "").strip(), time.time()
        problem = fetch_mod.check_url(url, self.allow_private)
        if problem:
            self._emit(tags, "blocked", tool=tool, url=url[:300], reason=problem)
            return FetchOutcome(False, url, error=f"URL not allowed: {problem}")
        key = normalise_url(url)
        self._emit(tags, "web_start", tool=tool, url=url[:300], fresh=fresh)
        if not fresh:
            hit = self.cache.get(tool, key, FETCH_TTL)
            if hit:
                self._emit(tags, "cache_hit", tool=tool, url=url[:300], provider=hit.provider, age_s=round(hit.age_s))
                return self._fetch_done(tags, FetchOutcome(
                    True, hit.value["url"], hit.value["title"], hit.value["text"], f"cache ({hit.provider})",
                    cached=True, seconds=time.time() - t0, age_s=hit.age_s, date=cached_date(hit.value)))
        try:
            name, page, attempts, _ = self._chain(tool, fetch_mod.PROVIDERS, url, tags, stop_event)
        except ChainFailed as e:
            old = self.cache.get(tool, key, None)
            if old:
                self._emit(tags, "stale_cache", tool=tool, url=url[:300], age_s=round(old.age_s),
                           reason="every live provider failed")
                return self._fetch_done(tags, FetchOutcome(
                    True, old.value["url"], old.value["title"], old.value["text"], f"cache ({old.provider})",
                    cached=True, stale=True, attempts=e.attempts, seconds=time.time() - t0, age_s=old.age_s,
                    date=cached_date(old.value)))
            self._emit(tags, "web_failed", tool=tool, url=url[:300], trail=[list(x) for x in e.trail])
            return FetchOutcome(False, url, attempts=e.attempts, seconds=round(time.time() - t0, 2),
                                error="every provider failed: " + str(e))
        self.cache.put(tool, key, page, name)
        return self._fetch_done(tags, FetchOutcome(True, page["url"], page["title"], page["text"], name,
                                                   attempts=attempts, seconds=time.time() - t0,
                                                   date=page.get("date", "")))

    def _fetch_done(self, tags, out: FetchOutcome) -> FetchOutcome:
        out.seconds = round(out.seconds, 2)
        self._emit(tags, "web_result", tool="fetch_page", url=out.url[:300], provider=out.provider,
                   cached=out.cached, stale=out.stale, chars=len(out.text), attempts=out.attempts,
                   seconds=out.seconds)
        return out