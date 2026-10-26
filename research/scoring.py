"""Scoring: six signals scaled 0..1 across the candidate pool, three weight presets, one formula.

Switching the preset only recomputes a weighted sum on the stored signals (rank_signals), so the
dashboard can re-sort instantly with no network and no LLM call.

    Score_c = 100 * sum_{i in A_c} w_i * s_ci  /  sum_{i in A_c} w_i        decay(t) = 2 ** (-t / h)

A_c = the signals that have data for company c. A missing figure is not punished in the score;
it lowers the CONFIDENCE instead:  0.5*coverage + 0.3*verified share + 0.2*min(domains, 3)/3.
"""
import math
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from research.verify import Figure, parse_date_value

DISCLAIMER = "Based on public web sources, not investment advice."
SIGNALS = ("funding", "valuation", "recent", "news", "traction", "maturity")
SIGNAL_LABELS = {"funding": "Funding scale", "valuation": "Valuation", "recent": "Recent funding",
                 "news": "News momentum", "traction": "Traction", "maturity": "Maturity"}
FIGURE_OF = {"funding": "total_raised_usd", "valuation": "valuation_usd", "recent": "last_round_date",
             "traction": "traction_evidence", "maturity": "founded_year"}      # news has no extracted figure

PRESETS = {
    "Established":    {"funding": .30, "valuation": .30, "recent": .05, "news": .05, "traction": .15, "maturity": .15},
    "Fastest rising": {"funding": .10, "valuation": .10, "recent": .30, "news": .25, "traction": .20, "maturity": .05},
    "Early-stage":    {"funding": .10, "valuation": .05, "recent": .30, "news": .20, "traction": .15, "maturity": .20},
}
YOUNGER_IS_BETTER = {"Early-stage"}

STALE_MONTHS = 18                   # a figure whose source page is older than this is marked as stale
STALE_FIELDS = ("total_raised_usd", "last_round_amount", "last_round_date", "valuation_usd", "traction_evidence")
                                    # a founding year never goes out of date, the others do
RECENT_HALF_LIFE_MONTHS = 12
NEWS_HALF_LIFE_DAYS = 30
NEWS_WINDOW_DAYS = 90
MATURITY_CAP_YEARS = 15

FORMULA = "Score = 100 x (sum of weight x signal over the signals that have data) / (sum of those weights)"
NORMALISATION = [
    "Funding scale: log10(total raised in USD), then min-max across the candidates.",
    "Valuation: log10(latest valuation in USD), then min-max across the candidates.",
    "Recent funding: 2 ** (-months since the last round / 12), a half-life of 12 months.",
    "News momentum: sum over articles of the last 90 days of 2 ** (-age in days / 30), one article per "
    "domain, then log1p and min-max across the candidates.",
    "Traction: 0, 0.5 or 1 by rules on the verified evidence quote (numbers of customers, users or revenue = 1).",
    "Maturity: age in years (capped at 15) / 15. The Early-stage preset uses 1 minus this.",
    "Confidence = 0.5 x coverage + 0.3 x verified share of the figures used (a figure from a page older than "
    "18 months does not count as verified) + 0.2 x min(distinct domains, 3) / 3. High at 0.75 or more, Medium at "
    "0.5 or more, Low below.",
]


# ------------------------------------------------------------------ the guide's building blocks
def decay(age_days, half_life):
    return 2 ** (-age_days / half_life)


def minmax(vals):
    lo, hi = min(vals), max(vals)
    return [0.5 if hi == lo else (v - lo) / (hi - lo) for v in vals]


def score(signals, preset):                  # signals: {name: value in 0..1 or None}
    w = PRESETS[preset]
    used = {k: v for k, v in signals.items() if v is not None}
    if not used:
        return 0.0, 0.0
    s = {k: (1 - v if k == "maturity" and preset in YOUNGER_IS_BETTER else v) for k, v in used.items()}
    total_w = sum(w[k] for k in used)
    value = 100 * sum(w[k] * s[k] for k in used) / total_w
    coverage = total_w / sum(w.values())
    return value, coverage


# ------------------------------------------------------------------ one company
@dataclass
class CompanyRecord:
    name: str
    figures: dict = field(default_factory=dict)        # field name -> Figure
    news: list = field(default_factory=list)           # [{"date": "YYYY-MM-DD", "title", "url", "domain", "via"}]
    news_ok: bool = False                              # False: the news search failed, so "no data", not "zero"
    urls: list = field(default_factory=list)           # pages that were read
    notes: list = field(default_factory=list)          # problems, shown in the dashboard
    news_sources: list = field(default_factory=list)   # news sources that answered: "gdelt" and/or "web"

    def to_dict(self) -> dict:
        return {"name": self.name, "figures": {k: f.to_dict() for k, f in self.figures.items()},
                "news": self.news, "news_ok": self.news_ok, "urls": self.urls, "notes": self.notes,
                "news_sources": self.news_sources}

    @classmethod
    def from_dict(cls, d: dict) -> "CompanyRecord":
        return cls(d["name"], {k: Figure.from_dict(v) for k, v in (d.get("figures") or {}).items()},
                   d.get("news") or [], bool(d.get("news_ok")), d.get("urls") or [], d.get("notes") or [],
                   d.get("news_sources") or [])


def _use(fig: Optional[Figure], use_unconfirmed: bool, today: Optional[date] = None,
         use_stale: bool = True) -> Optional[Figure]:
    """The figure if the score may use it: trusted ones always; a not-confirmed one only when it is not
    fatal (its number IS in a real quote) and use_unconfirmed is on (nothing better exists for this field).
    With use_stale=False a figure whose source is older than 18 months is left out as well."""
    if fig is None or fig.value in (None, ""):
        return None
    if not use_stale and figure_is_stale(fig, today):
        return None
    if fig.trusted or (use_unconfirmed and fig.usable):
        return fig
    return None


def round_date(text) -> Optional[date]:
    """'2026-03-15', '2026-03' or '2026' -> a date (missing month: July, missing day: the 1st)."""
    p = parse_date_value(str(text or ""))
    if not p:
        return None
    y, m, d = p
    try:
        return date(y, m or 7, d or 1)
    except ValueError:
        return date(y, m or 7, 1)


def news_family(article: dict) -> str:
    return "gdelt" if article.get("via") == "gdelt" else "web"


def news_in_window(news: list, today: Optional[date] = None) -> int:
    """How many articles are from the last 90 days (the ones the news signal can use)."""
    today = today or date.today()
    n = 0
    for a in news:
        try:
            d = date.fromisoformat(str(a.get("date") or "")[:10])
        except ValueError:
            continue
        n += (today - d).days <= NEWS_WINDOW_DAYS
    return n


def is_stale(as_of: str, today: Optional[date] = None) -> bool:
    """True when the source page of a figure is older than 18 months. Unknown date: False (nothing to say)."""
    p = parse_date_value(str(as_of or ""))
    if not p:
        return False
    d = date(p[0], p[1] or 7, p[2] or 1)
    return ((today or date.today()) - d).days > STALE_MONTHS * 30.4375


def figure_is_stale(fig: Optional[Figure], today: Optional[date] = None) -> bool:
    """True for a figure that can go out of date (money, round date, traction) whose source is older than 18 months."""
    return fig is not None and fig.field in STALE_FIELDS and is_stale(fig.as_of, today)


def news_raw(news: list, today: date) -> float:
    """Sum of decay weights of the articles of the last 90 days; only the newest article per domain counts."""
    newest = {}
    for a in news:
        try:
            d = date.fromisoformat(str(a.get("date") or "")[:10])
        except ValueError:
            continue                                    # an article without a date cannot be placed in time
        age = max((today - d).days, 0)
        dom = a.get("domain") or a.get("url") or ""
        if age <= NEWS_WINDOW_DAYS and (dom not in newest or age < newest[dom]):
            newest[dom] = age
    return sum(decay(age, NEWS_HALF_LIFE_DAYS) for age in newest.values())


def compute_signals(records: list, today: Optional[date] = None, use_unconfirmed: bool = True,
                    use_stale: bool = True) -> dict:
    """{company name: {signal: 0..1 or None}}. The pool-wide min-max happens here, once."""
    today = today or date.today()
    raw = {r.name: {} for r in records}
    # Compare companies on the same news sources: if GDELT answered for some companies and not for others,
    # its articles are left out for everybody (a company must not win because GDELT happened to work for it).
    lists = [set(r.news_sources) for r in records if r.news_ok and r.news_sources]
    shared = set.intersection(*lists) if lists else set()
    for r in records:
        fig = {s: _use(r.figures.get(FIGURE_OF[s]), use_unconfirmed, today, use_stale) for s in FIGURE_OF}
        v = raw[r.name]
        v["funding"] = math.log10(fig["funding"].value) if fig["funding"] and fig["funding"].value > 0 else None
        v["valuation"] = math.log10(fig["valuation"].value) if fig["valuation"] and fig["valuation"].value > 0 else None
        d = round_date(fig["recent"].value) if fig["recent"] else None
        v["recent"] = decay(max((today - d).days, 0) / 30.4375, RECENT_HALF_LIFE_MONTHS) if d else None
        news = [a for a in r.news if news_family(a) in shared] if shared else r.news
        v["news"] = math.log1p(news_raw(news, today)) if r.news_ok else None
        v["traction"] = fig["traction"].level if fig["traction"] and fig["traction"].level is not None else None
        v["maturity"] = (min(max(today.year - int(fig["maturity"].value), 0), MATURITY_CAP_YEARS) / MATURITY_CAP_YEARS
                         if fig["maturity"] else None)
    out = {r.name: dict(raw[r.name]) for r in records}
    for s in ("funding", "valuation", "news"):          # these three are scaled across the pool
        names = [n for n in raw if raw[n][s] is not None]
        if not names:                                   # nobody has this signal: nothing to scale
            continue
        for n, x in zip(names, minmax([raw[n][s] for n in names])):
            out[n][s] = x
    return out


# ------------------------------------------------------------------ confidence and ranking
def confidence_label(c: float) -> str:
    return "High" if c >= 0.75 else "Medium" if c >= 0.5 else "Low"


def confidence(rec: CompanyRecord, signals: dict, preset: str, use_unconfirmed: bool = True,
               today: Optional[date] = None, use_stale: bool = True) -> float:
    """A figure counts as trusted only if it is verified AND its source is not older than 18 months."""
    _, coverage = score(signals, preset)
    figs = [_use(rec.figures.get(FIGURE_OF[s]), use_unconfirmed, today, use_stale)
            for s in FIGURE_OF if signals.get(s) is not None]
    figs = [f for f in figs if f is not None]
    verified_share = sum(f.trusted and not figure_is_stale(f, today) for f in figs) / len(figs) if figs else 0.0
    domains = {d for f in figs for d in (f.domains or [f.domain])}
    return 0.5 * coverage + 0.3 * verified_share + 0.2 * min(len(domains), 3) / 3


@dataclass
class Ranked:
    rank: int
    name: str
    score: float
    coverage: float
    confidence: float
    label: str
    signals: dict
    record: CompanyRecord


def rank_signals(records: list, signals: dict, preset: str, use_unconfirmed: bool = True,
                 today: Optional[date] = None, use_stale: bool = True) -> list:
    """Weighted sum on stored signals + sort. This is what the preset dropdown calls."""
    rows = []
    for r in records:
        s = signals[r.name]
        value, coverage = score(s, preset)
        c = confidence(r, s, preset, use_unconfirmed, today, use_stale)
        rows.append(Ranked(0, r.name, value, coverage, c, confidence_label(c), s, r))
    rows.sort(key=lambda x: (-x.score, -x.confidence, x.name.lower()))
    for i, x in enumerate(rows, 1):
        x.rank = i
    return rows


def rank(records: list, preset: str, today: Optional[date] = None, use_unconfirmed: bool = True,
         use_stale: bool = True) -> list:
    return rank_signals(records, compute_signals(records, today, use_unconfirmed, use_stale), preset,
                        use_unconfirmed, today, use_stale)