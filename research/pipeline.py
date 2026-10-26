"""The startup-research pipeline (no UI): candidates -> per company: two searches, download about 5 pages, let
the model read the 2 newest, extract, verify, news -> a ResearchRun that research/scoring.py can rank.

Everything is an event on the bus (agent="research"), so the dashboard and the log show every step:
  research_start, candidates, research_company (phase searching / reading / news), extracted,
  verified (one per figure), news_result, error, research_done, research_stopped, research_failed
plus the normal web events (web_start, cache_hit, retry, fallback ...) and llm_call from the clients.

Failure isolation: one company that fails (a page cannot be read, the model answers badly) gets a note
and the run continues. Only 'the model server is down' and Stop end the whole run.
The model is called one request at a time, so do not run it while the coding agent is using the model.
"""
import json
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

from agents.coding.llm import LLMError, LLMStopped, LLMUnavailable
from research.errors import WebStopped
from research.extract import ExtractError, extract_facts, find_candidates, pick_pages
from research.news import find_news_detail
from research.scoring import CompanyRecord
from research.verify import Page, check_fact, merge_all, reclassify_rounds

RUN_VERSION = 1


class ResearchStopped(Exception):
    """The Stop button was pressed."""


@dataclass
class ResearchRun:
    category: str
    created: str
    model: str
    records: list
    stats: dict = field(default_factory=dict)
    seconds: float = 0.0
    path: str = ""


# ------------------------------------------------------------------ saving and loading a run
def run_path(settings, category: str) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "_", category.lower()).strip("_") or "run"
    return Path(settings.cache_dir) / "research" / f"{slug}.json"


def save_run(run: ResearchRun, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"version": RUN_VERSION, "category": run.category, "created": run.created, "model": run.model,
            "seconds": run.seconds, "stats": run.stats, "records": [r.to_dict() for r in run.records]}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    tmp.replace(path)
    run.path = str(path)
    return path


def load_run(path) -> ResearchRun:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return ResearchRun(d["category"], d["created"], d.get("model", ""), [CompanyRecord.from_dict(r) for r in d["records"]],
                       d.get("stats") or {}, float(d.get("seconds") or 0), str(path))


# ------------------------------------------------------------------ one company
def _check(stop_event) -> None:
    if stop_event is not None and stop_event.is_set():
        raise ResearchStopped()


def research_company(name, category, web, llm, settings, bus, tags, stats, *, pages_per_company=2,
                     extra_pages=3, with_news=True, fresh=False, stop_event=None) -> CompanyRecord:
    """pages_per_company pages go to the model. extra_pages more are only DOWNLOADED (no model call), so the
    model can read the newest-dated ones: search results come in a different order every day.
    Two searches feed the pool: the general one, and 'latest funding round {month year}' that favours new articles."""
    rec, checked = CompanyRecord(name), []
    today = date.today()
    bus.emit("research_company", **tags, name=name, phase="searching")
    out = web.search(f"{name} {category} startup total funding raised valuation {today.year}",
                     max_results=8, fresh=fresh, tags=tags, stop_event=stop_event)
    if not out.ok:
        rec.notes.append("funding search failed: " + out.error[:120])
    results = list(out.results) if out.ok else []
    _check(stop_event)
    latest = web.search(f"{name} latest funding round {today.strftime('%B %Y')}", max_results=8, fresh=fresh,
                        tags=tags, stop_event=stop_event)
    seen = {r["url"] for r in results}
    results += [r for r in (latest.results if latest.ok else []) if r["url"] not in seen]
    pool = pick_pages(results, name, pages_per_company + extra_pages)
    fetched = []
    for r in pool:
        _check(stop_event)
        bus.emit("research_company", **tags, name=name, phase="fetching", url=r["url"])
        fo = web.fetch(r["url"], fresh=fresh, tags=tags, stop_event=stop_event)
        if not fo.ok:
            rec.notes.append(f"could not read {r['url'][:80]}: {fo.error[:100]}")
            continue
        fetched.append(fo)
    fetched.sort(key=lambda fo: fo.date or "", reverse=True)        # newest first; undated pages last; ties keep search order
    for fo in fetched[:pages_per_company]:
        _check(stop_event)
        bus.emit("research_company", **tags, name=name, phase="reading", url=fo.url, page_date=fo.date)
        page = Page(fo.url, fo.title, fo.text, fo.date)
        rec.urls.append(fo.url)
        stats["pages"] += 1
        try:
            facts = extract_facts(llm, settings, web, bus, name, page, fresh=fresh, tags=tags)
        except LLMUnavailable:
            raise
        except (LLMError, ExtractError) as e:
            rec.notes.append(f"the model gave no usable answer for {page.url[:60]}: {str(e)[:100]}")
            bus.emit("error", **tags, name=name, where="extract", message=str(e)[:200])
            continue
        verdicts = [(f, check_fact(f, page, name)) for f in facts]
        verdicts += reclassify_rounds(verdicts, page, name)
        checked += verdicts
        stats["facts"] += len(facts)
        bus.emit("extracted", **tags, name=name, url=page.url, page_date=page.date, facts=len(facts),
                 verified=sum(v.status == "verified" for _, v in verdicts),
                 corrected=sum("corrected" in v.note for _, v in verdicts))
    rec.figures = merge_all(checked)
    for fig in rec.figures.values():
        stats[fig.status] = stats.get(fig.status, 0) + 1
        bus.emit("verified", **tags, name=name, field=fig.field, status=fig.status, value=fig.value,
                 domain=fig.domain, domains=fig.domains, source_url=fig.source_url, usable=fig.usable,
                 as_of=fig.as_of, reason=fig.reason)
    if with_news:
        _check(stop_event)
        bus.emit("research_company", **tags, name=name, phase="news")
        rec.news, rec.news_ok, rec.news_sources = find_news_detail(web, name, category, stop_event=stop_event,
                                                                   tags=tags)
        rec.news = rec.news[:60]
        stats["news_articles"] += len(rec.news)
        bus.emit("news_result", **tags, name=name, articles=len(rec.news), ok=rec.news_ok)
        if not rec.news_ok:
            rec.notes.append("no news data: both news sources failed")
    return rec


# ------------------------------------------------------------------ the whole run
def research(category: str, web, llm, settings, bus, *, max_candidates: int = 15, pages_per_company: int = 2,
             extra_pages: int = 3, with_news: bool = True, fresh: bool = False, stop_event=None,
             save: bool = True) -> ResearchRun:
    """Run the pipeline for one category. llm should be an LLMClient built with the same stop_event."""
    category = " ".join(category.split())
    tags = {"agent": "research"}
    stats = {"candidates": 0, "pages": 0, "facts": 0, "high": 0, "verified": 0, "not_confirmed": 0,
             "news_articles": 0, "failed_companies": 0}
    t0 = time.time()
    web.reset_breakers()                                              # every run gives GDELT a fresh chance
    bus.emit("research_start", **tags, category=category, max_candidates=max_candidates, model=settings.model_name)
    records = []
    try:
        _check(stop_event)
        names, sources = find_candidates(web, llm, settings, bus, category, max_candidates, fresh, stop_event, tags)
        stats["candidates"] = len(names)
        bus.emit("candidates", **tags, names=names, count=len(names), sources=sources[:5])
        for i, name in enumerate(names, 1):
            _check(stop_event)
            bus.emit("research_company", **tags, name=name, index=i, total=len(names), phase="start")
            try:
                records.append(research_company(name, category, web, llm, settings, bus, tags, stats,
                                                pages_per_company=pages_per_company, extra_pages=extra_pages,
                                                with_news=with_news, fresh=fresh, stop_event=stop_event))
            except (ResearchStopped, LLMStopped, WebStopped, LLMUnavailable):
                raise
            except Exception as e:                                  # failure isolation: this company only
                stats["failed_companies"] += 1
                bus.emit("error", **tags, name=name, where="company", message=f"{type(e).__name__}: {str(e)[:200]}")
                records.append(CompanyRecord(name, notes=[f"failed: {type(e).__name__}: {str(e)[:120]}"]))
    except (ResearchStopped, LLMStopped, WebStopped):
        bus.emit("research_stopped", **tags, category=category, done=len(records))
        raise ResearchStopped() from None
    except (LLMUnavailable, ExtractError, LLMError) as e:
        bus.emit("research_failed", **tags, category=category, reason=str(e)[:300])
        raise
    run = ResearchRun(category, datetime.now().isoformat(timespec="seconds"), settings.model_name, records, stats,
                      round(time.time() - t0, 1))
    if save:
        save_run(run, run_path(settings, category))
    bus.emit("research_done", **tags, category=category, companies=len(records), seconds=run.seconds,
             stats=stats, path=run.path)
    return run