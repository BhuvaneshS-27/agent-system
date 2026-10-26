"""Verification of extracted figures. No LLM and no network: plain string and number checks.

The model reads a web page and reports facts, each with an evidence_quote. A fact is VERIFIED only if
  1. the quote is really part of the fetched page text (after harmless clean-up: case, spaces, quotes;
     a quote with a small copy error is corrected to the closest real sentence of the page),
  2. the number (or year, or date) the model reported is written in that quote, as a US-dollar amount
     for money, and
  3. the quote is about the right company (its name is in the quote, the page title or the URL).
Money also needs the right kind of word next to it (a valuation word for a valuation, a funding word for
funding, and "to date / in total" for the TOTAL raised, so one round is not mistaken for the total).
A round date may come from the page's publication date when the quote has no date of its own, and a round
date that is later than its own page (or in the future) is never scored. When the model gives an amount but
no usable round date, infer_round_date() dates the round by the article (shown as 'not confirmed'); a "total"
that is really one round is moved to last_round_amount by reclassify_rounds().
Several quotes about the same field are then compared: a figure that a second DOMAIN confirms
(money within 10 percent, same year, same month) is upgraded from "verified" to "high". When verified
values disagree, the one from the NEWEST page wins (Figure.as_of tells how old its source is).

A fact that fails check 1 or 2 is "not confirmed" and FATAL: its number is not supported by the text,
so it never enters the score. A fact that only fails a softer check is "not confirmed" but not fatal:
the score may use it when nothing better exists, and the lower trust shows up in the confidence number
(see research/scoring.py). A few soft failures are marked UNUSABLE because the value is probably the
wrong thing (a round amount reported as the total, a founding year with no 'founded' word, a traction
quote with no customers, users or revenue, a round date that contradicts its page or has no funding word):
those are kept for display only.
"""
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from typing import Any, Optional
from urllib.parse import urlsplit

try:                                            # registered domain, so bbc.co.uk and www.bbc.co.uk agree
    from tld import get_fld
except ImportError:                             # pragma: no cover
    get_fld = None

FIELDS = ("total_raised_usd", "last_round_amount", "last_round_date", "valuation_usd", "founded_year",
          "traction_evidence")
MONEY_FIELDS = ("total_raised_usd", "last_round_amount", "valuation_usd")
MONEY_TOLERANCE = Decimal("0.005")              # rounding of the model's own arithmetic ("1.50B")
AGREE_TOLERANCE = Decimal("0.10")               # "a second domain gives the same number within 10 percent"
CUE_DISTANCE = 80                               # characters between the amount and its funding/valuation word
TOTAL_CUE_DISTANCE = 120                        # ... and its "total" word
MIN_QUOTE_CHARS = 12
SNAP_RATIO = 0.85                               # a quote this similar to a real page sentence is corrected to it
SNAP_MAX_SENTENCE = 600


# ------------------------------------------------------------------ text clean-up
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_TRANS = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-",
                        "\u2014": "-", "\u2212": "-", "\u00a0": " "})


def norm(text) -> str:
    """Lower case, one space between words, curly quotes and long dashes made plain, markdown links
    reduced to their text. Applied to BOTH the page and the quote, so it can only hide formatting."""
    t = unicodedata.normalize("NFKC", str(text or ""))
    t = _MD_LINK.sub(r"\1", t).translate(_TRANS)
    t = re.sub(r"[*_`]", "", t)
    t = re.sub(r"[#>|]", " ", t)
    return " ".join(t.lower().split())


def alnum(text) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", norm(text)).split())


def registrable_domain(url: str) -> str:
    host = (urlsplit(str(url)).hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if get_fld is not None and host:
        try:
            fld = get_fld(f"http://{host}", fail_silently=True)
            if fld:
                return fld.lower()
        except Exception:                       # pragma: no cover
            pass
    parts = host.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


# ------------------------------------------------------------------ amounts
_AMOUNT = re.compile(
    r"(?P<pre>us\s?\$|usd|ca\$|c\$|a\$|au\$|s\$|hk\$|nz\$|\$|€|eur|£|gbp|₹|inr|rs\.?|¥|cny|jpy)?\s?"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?:\s?-?\s?(?P<scale>trillion|billion|million|thousand|bn|mn|mm|[bmk])(?![a-z]))?"
    r"(?:\s?(?P<post>dollars|usd|euros?|pounds|rupees|yuan|yen))?", re.I)
_SCALE = {"trillion": 10 ** 12, "billion": 10 ** 9, "bn": 10 ** 9, "b": 10 ** 9, "million": 10 ** 6,
          "mn": 10 ** 6, "mm": 10 ** 6, "m": 10 ** 6, "thousand": 1000, "k": 1000}
_USD_PRE = {"us$", "us $", "usd", "$"}


@dataclass
class Amount:
    value: Decimal
    usd: bool
    span: tuple


def parse_amounts(text: str) -> list:
    """Every number in the text, scaled ('$1.5 billion' -> 1500000000), with a flag for 'in US dollars'."""
    out = []
    for m in _AMOUNT.finditer(text):
        try:
            value = Decimal(m.group("num").replace(",", "")) * _SCALE.get((m.group("scale") or "").lower(), 1)
        except InvalidOperation:                                    # pragma: no cover
            continue
        pre, post = (m.group("pre") or "").lower().strip(), (m.group("post") or "").lower()
        out.append(Amount(value, pre in _USD_PRE or post in ("dollars", "usd"), m.span()))
    return out


def parse_value_amount(value: str) -> Optional[Decimal]:
    """The amount the model wrote, e.g. '$1.5 billion' or '1500000000'. Currency is ignored here:
    the QUOTE must show it in US dollars."""
    for a in parse_amounts(str(value or "")):
        return a.value
    return None


def close(a: Decimal, b: Decimal, tol: Decimal) -> bool:
    return abs(a - b) <= tol * max(abs(a), abs(b))


# ------------------------------------------------------------------ dates
_MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september",
           "october", "november", "december"]
_MONTH_NUM = {**{m: i for i, m in enumerate(_MONTHS, 1)}, **{m[:3]: i for i, m in enumerate(_MONTHS, 1)},
              "sept": 9}
_YEAR = re.compile(r"\b((?:19|20)\d{2})\b")


def parse_date_value(value: str):
    """'2026-03', '2026-03-15', 'March 2026', 'Mar 15, 2026' or '2026' -> (year, month or None, day or None)."""
    t = norm(value)
    iso = re.search(r"\b((?:19|20)\d{2})-(\d{1,2})(?:-(\d{1,2}))?\b", t)
    if iso and 1 <= int(iso.group(2)) <= 12:
        return int(iso.group(1)), int(iso.group(2)), int(iso.group(3)) if iso.group(3) else None
    y = _YEAR.search(t)
    if not y:
        return None
    month = next((_MONTH_NUM[w] for w in re.findall(r"[a-z]+", t) if w in _MONTH_NUM), None)
    return int(y.group(1)), month, None


def date_text(y: int, m: Optional[int], d: Optional[int]) -> str:
    return f"{y:04d}" + (f"-{m:02d}" if m else "") + (f"-{d:02d}" if m and d else "")


# ------------------------------------------------------------------ cue words
CUES = {
    "funding": re.compile(r"\b(rais\w*|fundrais\w*|funding|financing|invest\w*|capital|backed|round|series [a-z]|seed)\b"),
    "valuation": re.compile(r"\b(valuation|valued|values|valuing|worth)\b"),
    "founded": re.compile(r"\b(founded|established|incorporated|launched|started|since|inception|began|formed)\b"),
    # "TOTAL raised": words that say it is everything so far, not one round
    "total": re.compile(r"\b(total|to date|so far|altogether|overall|cumulative\w*|in all|known capital|"
                        r"(?:has|have) now raised|now (?:has|have)|"
                        r"since (?:its |the |it )?(?:founding|inception|launch\w*|\d{4}))\b"),
    "funding_verb": re.compile(r"\b(rais(?:ed|es|ing)|secur(?:ed|es|ing)|announc(?:ed|es|ing)|clos(?:ed|es|ing)|"
                               r"lands?|landed|bagged?|completes?|completed|nabs?|nabbed|gets?|got|unveils?)\b"),
    # weak = the quote at least talks about customers or revenue. Generic words ("accounts", "businesses",
    # "sales", "deployed") are NOT here: "bank accounts integrated with tools businesses need" is a product
    # description, not traction.
    "traction_weak": re.compile(r"\b(customers?|users?|clients?|merchants?|subscribers?|members?|cardholders?|"
                                r"borrowers?|sellers?|revenue|arr|mrr|partners?|downloads?|profitable)\b"),
}
_TRACTION_STRONG = [
    re.compile(r"\d[\d,.]*\s?(?:k|m|bn|million|billion|thousand)?\+?\s+(?:\w+\s+){0,2}"
               r"(?:customers|users|clients|merchants|businesses|companies|subscribers|downloads|developers|"
               r"accounts|members|cardholders|borrowers|sellers)\b"),
    re.compile(r"[$€£]\s?\d[\d,.]*\s?(?:k|m|bn|million|billion|thousand)?\s+(?:\w+\s+){0,3}"
               r"(?:revenue|arr|mrr|in sales|in revenue)\b"),
    re.compile(r"\b(?:arr|mrr|revenue)\b[^.]{0,40}[$€£]\s?\d"),
]


def traction_level(quote: str) -> float:
    """Rule-based, from the verified quote: 1.0 = numbers (customers, users, members or revenue),
    0.5 = it only mentions customers, users, revenue, partners or 'profitable', 0.0 = nothing like that."""
    q = norm(quote)
    if any(p.search(q) for p in _TRACTION_STRONG):
        return 1.0
    return 0.5 if CUES["traction_weak"].search(q) else 0.0


def _gap(a: tuple, b: tuple) -> int:
    if a[1] <= b[0]:
        return b[0] - a[1]
    if b[1] <= a[0]:
        return a[0] - b[1]
    return 0


def cue_distance(text: str, span: tuple, cue: re.Pattern) -> int:
    return min((_gap(span, m.span()) for m in cue.finditer(text)), default=10 ** 6)


# ------------------------------------------------------------------ data
@dataclass
class Page:
    url: str
    title: str
    text: str
    date: str = ""                                          # publication date 'YYYY-MM-DD' / 'YYYY-MM' / ''
    _norm: Optional[str] = field(default=None, repr=False, compare=False)
    _sents: Optional[list] = field(default=None, repr=False, compare=False)

    @property
    def normalised(self) -> str:
        if self._norm is None:
            self._norm = norm(self.text)
        return self._norm

    @property
    def sentences(self) -> list:
        """[(sentence as written, normalised sentence)] for quote correction."""
        if self._sents is None:
            pieces = (s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", self.text))
            self._sents = [(s, norm(s)) for s in pieces if MIN_QUOTE_CHARS <= len(s) <= SNAP_MAX_SENTENCE]
        return self._sents

    @property
    def domain(self) -> str:
        return registrable_domain(self.url)


@dataclass
class Fact:
    """What the model claimed about one field, from one page."""
    field: str
    value: str
    quote: str
    source_url: str
    title: str = ""
    page_date: str = ""                                     # publication date of the page it came from


@dataclass
class Verdict:
    status: str                       # "verified" or "not_confirmed"
    fatal: bool                       # True: the number is not supported by the text
    reasons: list
    parsed: Any = None                # int dollars, int year, "YYYY[-MM[-DD]]", or the traction text
    level: Optional[float] = None     # traction only
    unusable: bool = False            # soft failure that still disqualifies the value from the score
    note: str = ""                    # something to show next to a verified value
    quote: str = ""                   # the quote as it really is on the page (after any correction)


@dataclass
class Figure:
    """The best value we have for one field of one company, with its proof."""
    field: str
    value: Any
    status: str                       # "high", "verified" or "not_confirmed"
    usable: bool                      # False: the score never uses it
    source_url: str
    evidence_quote: str
    domain: str
    domains: list                     # every domain that gave the same value (2 or more means "high")
    reason: str = ""
    level: Optional[float] = None     # traction only: 0, 0.5 or 1
    as_of: str = ""                   # publication date of the source page ('' = unknown)

    @property
    def trusted(self) -> bool:
        return self.status in ("high", "verified")

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict) -> "Figure":
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__ if k in d})


# ------------------------------------------------------------------ correcting a copy error in a quote
def snap_quote(quote: str, page: Page) -> Optional[str]:
    """The model often copies a sentence with a small slip. If the quote is almost identical (ratio of
    0.85 or more) to ONE real sentence of the page, return that sentence as it is written on the page.
    All number checks then run against the REAL sentence, so a changed number still fails."""
    nq = norm(quote)
    if len(nq) < MIN_QUOTE_CHARS:
        return None
    words = {w for w in re.findall(r"[a-z0-9$]+", nq) if len(w) > 2}
    if not words:
        return None
    best, best_ratio = None, SNAP_RATIO
    for raw, n in page.sentences:
        sw = set(re.findall(r"[a-z0-9$]+", n))
        if len(words & sw) < 0.7 * len(words):
            continue
        sm = SequenceMatcher(None, nq, n, autojunk=False)
        if sm.quick_ratio() < best_ratio:
            continue
        r = sm.ratio()
        if r >= best_ratio:
            best, best_ratio = raw, r
    return best


# ------------------------------------------------------------------ checking one fact
def _word_in(token: str, text: str) -> bool:
    return re.search(rf"\b{re.escape(token)}\b", text) is not None


def _date_problem(y: int, mo: Optional[int], nq: str) -> str:
    """'' if the year (and month) are written in the quote, else the reason."""
    if not _word_in(str(y), nq):
        return f"the year {y} is not in the quote"
    if mo:
        iso_in_quote = re.search(rf"\b{y}-0?{mo}\b", nq)
        month_in_quote = any(_word_in(w, nq) for w, n in _MONTH_NUM.items() if n == mo)
        if not (iso_in_quote or month_in_quote):
            return "the month is not in the quote"
    return ""


def _period(y: int, mo: Optional[int], d: Optional[int]):
    """(first day, last day) a partial date can mean: '2026' = the whole year, '2026-03' = March."""
    first = date(y, mo or 1, d or 1)
    if d:
        return first, first
    if mo:
        return first, date(y + (mo == 12), mo % 12 + 1, 1) - timedelta(days=1)
    return first, date(y, 12, 31)


def date_contradiction(y: int, mo: Optional[int], d: Optional[int], page_date: str, today: Optional[date] = None) -> str:
    """'' if the round date is believable, 'future' if it has not happened yet, or the reason if it is later than
    the page that reports it (a page cannot know about a round that happened after it was published)."""
    today = today or date.today()
    try:
        start, _ = _period(y, mo, d)
        pd = parse_date_value(page_date) if page_date else None
        page_end = _period(*pd)[1] if pd else None
    except ValueError:
        return ""
    if start > today + timedelta(days=1):
        return "future"
    if page_end and start > page_end + timedelta(days=31):
        return (f"the round date {date_text(y, mo, d)} is later than the page's own date {page_date}: "
                "the page cannot know about it")
    return ""


def check_fact(fact: Fact, page: Page, company: str) -> Verdict:
    if fact.field not in FIELDS:
        return Verdict("not_confirmed", True, [f"unknown field {fact.field!r}"])
    nq = norm(fact.quote)
    shown = fact.quote
    fatal, soft, notes = [], [], []
    unusable = False

    # 1. the quote is part of the page (a copy slip is corrected to the real sentence)
    if len(nq) < MIN_QUOTE_CHARS:
        fatal.append("the evidence quote is missing or too short")
    elif nq not in page.normalised:
        real = snap_quote(fact.quote, page)
        if real:
            shown, nq = real, norm(real)
            notes.append("quote corrected to the closest sentence on the page")
        else:
            fatal.append("the evidence quote is not in the page text")

    # 2. the number in the quote equals the number the model reported
    parsed: Any = None
    level = None
    if fact.field in MONEY_FIELDS:
        want = parse_value_amount(fact.value)
        if want is None:
            fatal.append(f"{fact.value!r} is not an amount")
        else:
            hits = [a for a in parse_amounts(nq) if a.usd and close(a.value, want, MONEY_TOLERANCE)]
            if not hits:
                fatal.append(f"{fact.value!r} is not written in the quote as a US-dollar amount")
            else:
                parsed = int(want)
                target, other = ("valuation", "funding") if fact.field == "valuation_usd" else ("funding", "valuation")
                best = min(hits, key=lambda a: cue_distance(nq, a.span, CUES[target]))
                d_t, d_o = cue_distance(nq, best.span, CUES[target]), cue_distance(nq, best.span, CUES[other])
                if d_t > CUE_DISTANCE or d_t > d_o:
                    soft.append("the amount is not clearly " + ("a valuation" if target == "valuation"
                                                                else "a funding amount"))
                if fact.field == "total_raised_usd" and cue_distance(nq, best.span, CUES["total"]) > TOTAL_CUE_DISTANCE:
                    soft.append("the quote does not say this is the TOTAL raised (no 'to date', 'in total', 'so far')")
                    unusable = True                       # it is probably one round, not the total
    elif fact.field == "founded_year":
        y = _YEAR.search(norm(fact.value))
        if not y:
            fatal.append(f"{fact.value!r} is not a year")
        else:
            m = re.search(rf"\b{y.group(1)}\b", nq)
            if not m:
                fatal.append(f"the year {y.group(1)} is not in the quote")
            else:
                parsed = int(y.group(1))
                if cue_distance(nq, m.span(), CUES["founded"]) > CUE_DISTANCE:
                    soft.append("the year is not next to a word like 'founded'")
                    unusable = True                       # a wrong year would distort the maturity signal
    elif fact.field == "last_round_date":
        d = parse_date_value(fact.value)
        if d is None:
            fatal.append(f"{fact.value!r} is not a date")
        else:
            y, mo, day = d
            problem = _date_problem(y, mo, nq)
            if not problem:
                parsed = date_text(y, mo, day)
                if not CUES["funding"].search(nq):
                    soft.append("no funding word in the quote")
                    unusable = True                       # a date with no money in the sentence is often an acquisition or a launch
                bad = date_contradiction(y, mo, day, page.date)
                if bad == "future":
                    fatal.append("the round date is in the future")
                elif bad:
                    soft.append(bad)
                    unusable = True                       # one page with a date that contradicts itself: not scored
            else:
                # An announcement often has its date in the dateline, not in the sentence. Accept the
                # page's publication date when it agrees with the model's date and the quote says that
                # money was raised.
                pd = parse_date_value(page.date) if page.date else None
                if (pd and pd[0] == y and (mo is None or pd[1] == mo) and CUES["funding_verb"].search(nq)
                        and not fatal):
                    parsed = date_text(y, mo, day)
                    notes.append("dated by the page's publication date (the quote has no date)")
                else:
                    fatal.append(problem)
    else:                                                       # traction_evidence
        parsed = " ".join(str(fact.value).split())[:160]
        level = traction_level(shown)
        if level == 0.0:
            soft.append("the quote shows no customers, users, members or revenue")
            unusable = True                                     # no evidence is not the same as zero traction
        elif traction_level(fact.value) == 0.0:                 # the claim itself must be about customers or revenue
            soft.append("what the model reports (" + parsed[:40] + ") is not about customers, users or revenue")
            unusable = True

    # 3. the quote is about this company
    key = alnum(company)
    if key and not any(f" {key} " in f" {alnum(t)} " for t in (shown, page.title, page.url)):
        soft.append("neither the quote, the page title nor the URL names the company")

    note = "; ".join(notes)
    if fatal:
        return Verdict("not_confirmed", True, fatal + soft, parsed, level, True, note, shown)
    if soft:
        return Verdict("not_confirmed", False, soft, parsed, level, unusable, note, shown)
    return Verdict("verified", False, [], parsed, level, False, note, shown)


# ------------------------------------------------------------------ comparing several sources
def agree(field_name: str, a: Any, b: Any) -> bool:
    """Do two verified values say the same thing? Money within 10 percent, year equal, month equal."""
    if field_name in MONEY_FIELDS:
        return close(Decimal(int(a)), Decimal(int(b)), AGREE_TOLERANCE)
    if field_name == "founded_year":
        return int(a) == int(b)
    if field_name == "last_round_date":
        pa, pb = parse_date_value(a), parse_date_value(b)
        return bool(pa and pb and pa[0] == pb[0] and (pa[1] is None or pb[1] is None or pa[1] == pb[1]))
    return False


def _same(field_name: str, v1: Verdict, v2: Verdict) -> bool:
    return v1.level == v2.level if field_name == "traction_evidence" else agree(field_name, v1.parsed, v2.parsed)


def merge_field(field_name: str, checked: list) -> Optional[Figure]:
    """checked = [(Fact, Verdict), ...] for ONE field, in reading order -> the best Figure or None.
    Among verified values the one from the NEWEST page wins (unknown date = oldest); then the one that most
    domains confirm; then the first one read. Two or more agreeing domains make it 'high'.
    Without a verified value: the first not-confirmed value that is not fatal, preferring a usable one,
    then a fatal one (kept only so the dashboard can show it)."""
    if not checked:
        return None
    good = [(f, v) for f, v in checked if v.status == "verified"]
    if good:
        best = None
        for idx, (f, v) in enumerate(good):
            doms = {registrable_domain(f.source_url)}
            doms |= {registrable_domain(f2.source_url) for f2, v2 in good if _same(field_name, v, v2)}
            key = (f.page_date or "", len(doms), -idx)
            if best is None or key > best[0]:
                best = (key, f, v, sorted(doms))
        _, f, v, doms = best
        reasons = [v.note] if v.note else []
        if any(not _same(field_name, v, v2) for _, v2 in good):
            reasons.append("sources disagree; the newest page was used")
        return Figure(field_name, v.parsed, "high" if len(doms) >= 2 else "verified", True, f.source_url,
                      v.quote or f.quote, registrable_domain(f.source_url), doms, "; ".join(reasons), v.level,
                      f.page_date)
    soft = sorted([(f, v) for f, v in checked if not v.fatal], key=lambda fv: fv[1].unusable)
    f, v = (soft or checked)[0]
    return Figure(field_name, v.parsed if v.parsed is not None else f.value, "not_confirmed",
                  bool(soft) and not v.unusable, f.source_url, v.quote or f.quote, registrable_domain(f.source_url),
                  [], "; ".join(v.reasons), v.level, f.page_date)


def reclassify_rounds(checked: list, page: Page, company: str) -> list:
    """The model sometimes files one round as the TOTAL raised ("completed its Series I fundraise of more than
    $6.5 billion"). The total check rightly refuses it, but then the company has no funding figure at all.
    If the same quote passes every check as a LAST ROUND amount, it is added as one (more [Fact, Verdict] pairs)."""
    have = {f.quote for f, _ in checked if f.field == "last_round_amount"}
    extra = []
    for f, v in checked:
        if (f.field == "total_raised_usd" and v.unusable and not v.fatal and f.quote not in have
                and any("TOTAL" in r for r in v.reasons)):
            as_round = Fact("last_round_amount", f.value, f.quote, f.source_url, f.title, f.page_date)
            v2 = check_fact(as_round, page, company)
            if v2.status == "verified" and CUES["funding_verb"].search(norm(v2.quote or f.quote)):
                v2.note = "; ".join(x for x in (v2.note, "filed as the total by the model, but the quote describes one round") if x)
                extra.append((as_round, v2))
                have.add(f.quote)
    return extra


def infer_round_date(figures: dict) -> Optional[Figure]:
    """The model often reports the amount of the latest round but no date, although the article announces it
    ("Ramp said on Thursday that it had raised $1.85 billion"). If the last_round_amount is verified, its quote
    has a funding verb, the quote names no other year, and the page has a date, the round is dated by the
    article: month precision, status not_confirmed (shown with '?'), still usable. A date that the model
    gave and that passed the checks is never replaced."""
    have, amount = figures.get("last_round_date"), figures.get("last_round_amount")
    if (have is not None and have.usable) or amount is None or not amount.trusted or not amount.as_of:
        return None
    q = norm(amount.evidence_quote)
    pd = parse_date_value(amount.as_of)
    if not pd or not pd[1] or not CUES["funding_verb"].search(q):
        return None
    if any(int(y) != pd[0] for y in _YEAR.findall(q)):          # "raised $555 million in 2021" on a 2026 page
        return None
    return Figure("last_round_date", date_text(pd[0], pd[1], None), "not_confirmed", True, amount.source_url,
                  amount.evidence_quote, amount.domain, [],
                  "inferred from the article's date (the model gave no usable round date)", None, amount.as_of)


def merge_all(checked: list) -> dict:
    """[(Fact, Verdict), ...] for all fields -> {field: Figure}."""
    out = {}
    for name in FIELDS:
        fig = merge_field(name, [(f, v) for f, v in checked if f.field == name])
        if fig is not None:
            out[name] = fig
    inferred = infer_round_date(out)
    if inferred is not None:
        out["last_round_date"] = inferred
    return out