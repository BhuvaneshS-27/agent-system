"""Run the startup research for one category and print the ranking table.

  python scripts/run_research.py "FinTech" --limit 5          # real run: needs llama-server (scripts/start_llm.py)
  python scripts/run_research.py "FinTech" --saved            # no model, no network: re-rank the saved run
  python scripts/run_research.py "FinTech" --saved --preset all --detail --drop-stale
  python scripts/run_research.py "FinTech" --limit 5 --fresh --gdelt

A real run saves cache/research/<category>.json; later runs of the same category reuse the cached
search results, pages and model answers, so they are fast and spend no quota.

Per company the program runs two searches, DOWNLOADS about 5 pages (no model call, a few seconds each) and the
model reads the 2 newest-dated ones (--pages 2). --extra-pages 0 reads the first 2 search results as before.

While it runs, "(34s)" in front of a line is the time since the previous line, i.e. how long the
step before that line took (hidden events such as model calls count towards the next visible line).

Marks after a figure:  (nothing) verified   ++ confirmed by a second domain   ? not confirmed (includes dates
                       inferred from the article's date)   !  its source page is older than 18 months
                       rejected = never used
Cov = share of the preset's weight that has data for that company. News = articles of the last 90 days.
LastRound shows year and month. --drop-stale leaves figures from pages older than 18 months out of the score.
GDELT (a second news source) is OFF unless you add --gdelt: from many networks it needs 15 s per request and is
rate-limited. News then comes from Tavily only. Run scripts/check_gdelt.py to see whether it works for you.
"""
import argparse
import sys
import threading
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agents.coding.llm import LLMClient, LLMError                       # noqa: E402
from core.bus import EventBus                                           # noqa: E402
from core.settings import load_settings                                 # noqa: E402
from research.extract import ExtractError                               # noqa: E402
from research.pipeline import ResearchStopped, load_run, research, run_path   # noqa: E402
from research.scoring import (DISCLAIMER, PRESETS, SIGNALS, compute_signals, figure_is_stale, news_in_window,   # noqa: E402
                              rank_signals)
from research.verify import FIELDS                                      # noqa: E402
from research.web import WebTools                                       # noqa: E402

SHORT = {"funding": "Fund", "valuation": "Valu", "recent": "Rcnt", "news": "News", "traction": "Trac", "maturity": "Matu"}


def money(v) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "-"
    for unit, size in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if v >= size:
            return f"${v / size:.3g}{unit}"
    return f"${v:.0f}"


def month(v) -> str:
    """'2026-05-20' -> '2026-05' (the table shows year and month only)."""
    return str(v)[:7]


def mark(fig) -> str:
    return "" if fig is None else "++" if fig.status == "high" else "?" if fig.status == "not_confirmed" else ""


def cell(rec, field_name, fmt=str, today=None) -> str:
    fig = rec.figures.get(field_name)
    if fig is not None and not fig.usable:
        return "rejected"                              # not in its quote, or probably the wrong thing: never used
    if fig is None:
        return "-"
    return fmt(fig.value) + mark(fig) + ("!" if figure_is_stale(fig, today) else "")


TABLE_HEAD = (f"{'#':>2} {'Company':<14} {'Score':>5} {'Conf':<11} {'Cov':>4} {'Raised':<9} {'Valuation':<9} "
              f"{'LastRound':<10} {'Founded':<7} {'News':>4} | " + " ".join(f"{SHORT[s]:>4}" for s in SIGNALS))


def table_row(r, today) -> str:
    rec = r.record
    sig = " ".join(f"{(round(r.signals[s] * 100) if r.signals.get(s) is not None else '--'):>4}" for s in SIGNALS)
    return (f"{r.rank:>2} {r.name[:14]:<14} {r.score:>5.1f} {r.label + ' ' + format(r.confidence, '.2f'):<11} "
            f"{r.coverage * 100:>3.0f}% {cell(rec, 'total_raised_usd', money, today):<9} "
            f"{cell(rec, 'valuation_usd', money, today):<9} {cell(rec, 'last_round_date', month, today):<10} "
            f"{cell(rec, 'founded_year', str, today):<7} {news_in_window(rec.news, today):>4} | {sig}")


def print_table(rows, preset: str, top: int, today=None) -> None:
    today = today or date.today()
    print(f"\nPreset: {preset}   ({DISCLAIMER})")
    print(TABLE_HEAD)
    print("-" * len(TABLE_HEAD))
    for r in rows[:top]:
        print(table_row(r, today))


def print_detail(rows, top: int, today=None) -> None:
    """Every figure of the top companies with its proof: value, status, age of the source, domain, quote."""
    today = today or date.today()
    print("\nDETAIL (source = date of the page the figure came from)")
    for r in rows[:top]:
        print(f"\n{r.rank}. {r.name}")
        for name in FIELDS:
            fig = r.record.figures.get(name)
            if fig is None:
                continue
            shown = money(fig.value) if name in ("total_raised_usd", "last_round_amount", "valuation_usd") else fig.value
            state = "REJECTED" if not fig.usable else fig.status
            age = fig.as_of or "date unknown"
            stale = " STALE" if figure_is_stale(fig, today) else ""
            print(f"   {name:<18} {str(shown):<12} {state:<13} source {age}{stale} ({fig.domain})")
            print(f"      \"{fig.evidence_quote[:110]}\"" + (f"   [{fig.reason[:90]}]" if fig.reason else ""))
        for note in r.record.notes[:3]:
            print(f"   note: {note[:110]}")


EVENT_LINES = {
    "research_start": lambda e: f"START {e['category']} (up to {e['max_candidates']} companies, model {e['model']})",
    "candidates": lambda e: f"CANDIDATES {e['count']}: " + ", ".join(e["names"]),
    "research_company": lambda e: (f"[{e['index']}/{e['total']}] {e['name']}" if e["phase"] == "start"
                                   else f"    {e['name']}: {e['phase']}" + (f" {e['url'][:70]}" if e.get("url") else "")
                                   + (f" (page date {e['page_date']})" if e.get("page_date") else "")),
    "extracted": lambda e: (f"    model read {e['url'][:60]} (page date {e.get('page_date') or 'unknown'}): "
                            f"{e['facts']} facts, {e['verified']} verified"
                            + (f", {e['corrected']} quotes corrected" if e.get("corrected") else "")),
    "verified": lambda e: (f"    {e['field']} = {e['value']} -> {e['status']}"
                           + (f" as of {e['as_of']}" if e.get("as_of") else "")
                           + (f" ({e['reason'][:60]})" if e.get("reason") else "")),
    "news_result": lambda e: f"    news: {e['articles']} articles from the last 90 days" + ("" if e["ok"] else " (both sources failed)"),
    "retry": lambda e: f"    retry: {e.get('provider') or e.get('what')} attempt {e['attempt']}: {e['reason'][:70]}",
    "fallback": lambda e: (f"    fallback: {e['provider']} -> {e['next']}: {e['reason'][:70]}" if e.get("next")
                           else f"    gave up on {e['provider']} (nothing left in this chain): {e['reason'][:70]}"),
    "error": lambda e: f"    ERROR {e.get('name', '')}: {e['message'][:100]}",
    "research_done": lambda e: f"DONE {e['companies']} companies in {e['seconds']} s, saved to {e['path']}",
}


def event_lines(events, last_t=None):
    """Printable lines for the events that have a formatter -> (lines, last_t).
    The time in front of a line is counted from the previous PRINTED line, so a model call (an event that is
    not printed) is counted in the line that follows it, e.g. 'model read ...'."""
    out = []
    for e in events:
        fn = EVENT_LINES.get(e["kind"])
        if not fn:
            continue
        gap = 0.0 if last_t is None else e["t"] - last_t
        last_t = e["t"]
        out.append((f"({gap:>3.0f}s) " if gap >= 1 else "       ") + fn(e))
    return out, last_t


def watch(bus, stop: threading.Event) -> None:
    cursor, last_t = 0, None
    while True:
        events, cursor = bus.read_since(cursor)
        lines, last_t = event_lines(events, last_t)
        for line in lines:
            print(line, flush=True)
        if stop.is_set() and not events:
            return
        time.sleep(0.4)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("category")
    ap.add_argument("--limit", type=int, default=15, help="how many companies to research (default 15; use 5 on the 4B model)")
    ap.add_argument("--pages", type=int, default=2, help="pages the MODEL reads per company (default 2; one model call each)")
    ap.add_argument("--extra-pages", type=int, default=3, help="more pages only downloaded (no model call) so the newest can be chosen (default 3)")
    ap.add_argument("--preset", default="all", help=f"{' / '.join(PRESETS)} / all (default all)")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--saved", action="store_true", help="do not research: load the saved run and rank it")
    ap.add_argument("--fresh", action="store_true", help="ignore cached search results, pages and model answers")
    ap.add_argument("--no-news", action="store_true")
    ap.add_argument("--gdelt", action="store_true", help="also use the GDELT news source (off by default: slow and rate-limited from many networks)")
    ap.add_argument("--strict", action="store_true", help="never use a not-confirmed figure in the score")
    ap.add_argument("--drop-stale", action="store_true", help="leave figures from pages older than 18 months out of the score")
    ap.add_argument("--detail", action="store_true", help="after the tables, print every figure with its source date and quote")
    return ap


def main() -> int:
    a = build_parser().parse_args()
    presets = list(PRESETS) if a.preset.lower() == "all" else [p for p in PRESETS if p.lower() == a.preset.lower()]
    if not presets:
        print(f"Unknown preset {a.preset!r}. Use one of: {', '.join(PRESETS)}, all")
        return 2

    settings = load_settings()
    if a.saved:
        path = run_path(settings, a.category)
        if not path.exists():
            print(f"No saved run at {path}. Run once without --saved first.")
            return 1
        run = load_run(path)
        print(f"Loaded {len(run.records)} companies, researched {run.created} with {run.model}.")
    else:
        bus = EventBus(settings.runs_dir / f"research_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
        stop_event = threading.Event()
        llm = LLMClient(settings, bus, stop_event=stop_event)
        state, detail = llm.health()
        if state != "online":
            print(f"The model server is {state} ({detail}). Start it first:  python scripts/start_llm.py")
            return 1
        web = WebTools(bus, settings.cache_dir)
        if not a.gdelt:
            web.disabled.add("gdelt")                      # opt-in: nothing is sent to GDELT and nothing is logged about it
        printer_stop = threading.Event()
        threading.Thread(target=watch, args=(bus, printer_stop), daemon=True).start()
        try:
            run = research(a.category, web, llm, settings, bus, max_candidates=a.limit, pages_per_company=a.pages,
                           extra_pages=a.extra_pages, with_news=not a.no_news, fresh=a.fresh, stop_event=stop_event)
        except KeyboardInterrupt:
            stop_event.set()
            print("Stopped.")
            return 130
        except ResearchStopped:
            print("Stopped.")
            return 130
        except (LLMError, ExtractError) as e:
            print(f"Research failed: {e}")
            return 1
        finally:
            time.sleep(0.6)
            printer_stop.set()
        print(f"\nEvent log: {bus.logfile}")

    today, use_unconfirmed, use_stale = date.today(), not a.strict, not a.drop_stale
    signals = compute_signals(run.records, today, use_unconfirmed, use_stale)
    rows = {}
    for preset in presets:
        rows[preset] = rank_signals(run.records, signals, preset, use_unconfirmed, today, use_stale)
        print_table(rows[preset], preset, a.top, today)
    print("\nMarks: (none) verified   ++ confirmed by a second domain   ? not confirmed or inferred   "
          "! source older than 18 months   rejected = never used.\nCov = share of the preset's weight with data. "
          "News = articles of the last 90 days. Signals are 0-100 after scaling across these companies; '--' = no data."
          + ("\n--drop-stale is ON: figures from pages older than 18 months are left out of the score." if a.drop_stale else ""))
    if a.detail:
        print_detail(rows[presets[0]], a.top, today)
    return 0


if __name__ == "__main__":
    sys.exit(main())