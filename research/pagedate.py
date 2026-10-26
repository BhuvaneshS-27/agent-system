"""When was a web page published? Needed so an old figure is not mistaken for a current one.

Two sources: the page's own metadata (trafilatura, Jina 'Published Time') and a date in the URL
(/2026/05/20/some-article or /2021-11-22-some-article). Dates are text: 'YYYY-MM-DD', or 'YYYY-MM' when
there is no day. Text dates sort correctly, so the newest source is simply the biggest string.

Dates written in the page text are used very carefully. News sites show "latest news" lists with today's date
in headers and sidebars (a fintechfutures.com article of June 2026 listed "Oct 23, 2026" around it), so the
newest date in a text is NOT the date of the page. The only text dates that count are those in TABLE ROWS
(lines with '|' separators and a money amount, like Tracxn's "Apr 08, 2024 | $694M | Series I"), because
aggregator profiles are updated without their metadata changing; and such a date can lift the page date by at
most 12 months over its metadata or URL date. A page with no metadata and no URL date stays undated ('')
instead of being guessed.
Nothing here uses the network.
"""
import re
from datetime import date, datetime, timedelta
from typing import Optional

MIN_YEAR = 1995

_URL = re.compile(r"/((?:19|20)\d{2})[/-](0[1-9]|1[0-2])(?:[/-](0[1-9]|[12]\d|3[01]))?(?=[/_.\-?#]|$)")


def parse_any_date(value) -> Optional[date]:
    """'20260905T100000Z', '2026-09-05...', 'Fri, 04 Sep 2026 10:00:00 GMT', 'Sept 4, 2026' -> date, else None."""
    s = str(value or "").strip()
    if not s:
        return None
    try:
        if len(s) >= 8 and s[:8].isdigit():
            return datetime.strptime(s[:8], "%Y%m%d").date()
        return date.fromisoformat(s[:10])
    except ValueError:
        pass
    try:
        from dateutil import parser as dparser
        return dparser.parse(s, fuzzy=True).date()
    except (ValueError, OverflowError, ImportError):
        return None


def plausible(d: Optional[date], today: Optional[date] = None) -> bool:
    today = today or date.today()
    return d is not None and d.year >= MIN_YEAR and d <= today + timedelta(days=1)


def url_date(url: str, today: Optional[date] = None) -> str:
    """'https://x.com/2026/05/20/story' -> '2026-05-20';  '.../2021/11/story' -> '2021-11';  none -> ''."""
    m = _URL.search(str(url or ""))
    if not m:
        return ""
    y, mo, d = int(m.group(1)), int(m.group(2)), m.group(3)
    try:
        if d:
            ok = plausible(date(y, mo, int(d)), today)
            return f"{y:04d}-{mo:02d}-{int(d):02d}" if ok else ""
        ok = plausible(date(y, mo, 1), today)
        return f"{y:04d}-{mo:02d}" if ok else ""
    except ValueError:
        return ""


def best_date(meta_value, url: str, today: Optional[date] = None) -> str:
    """The page's own date if it has a believable one, else the date in the URL, else ''."""
    d = parse_any_date(meta_value)
    if plausible(d, today):
        return d.isoformat()
    return url_date(url, today)


# ------------------------------------------------------------------ dates written in the page text
_MON = (r"(?:january|february|march|april|may|june|july|august|september|october|november|december|"
        r"jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec)")
_MONTH = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_YEAR4 = r"((?:19|20)\d{2})"
_ISO = re.compile(rf"\b{_YEAR4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b")
_MDY = re.compile(rf"\b({_MON})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+{_YEAR4}\b", re.I)         # Apr 08, 2024
_DMY = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MON})\.?,?\s+{_YEAR4}\b", re.I)          # 8 April 2024
_MY = re.compile(rf"\b({_MON})\.?,?\s+{_YEAR4}\b", re.I)                                          # April 2024


def _month_number(word: str) -> int:
    return _MONTH[word.lower()[:3]]


TEXT_DATE_MAX_MONTHS = 12
_MONEY = re.compile(r"[$€£]\s?\d")


def _dates_in(line: str, today: Optional[date]) -> list:
    found = []                                            # (y, m, d or None)
    for m in _ISO.finditer(line):
        found.append((int(m.group(1)), int(m.group(2)), int(m.group(3))))
    for m in _MDY.finditer(line):
        found.append((int(m.group(3)), _month_number(m.group(1)), int(m.group(2))))
    for m in _DMY.finditer(line):
        found.append((int(m.group(3)), _month_number(m.group(2)), int(m.group(1))))
    for m in _MY.finditer(line):
        found.append((int(m.group(2)), _month_number(m.group(1)), None))
    out = []
    for y, mo, d in found:
        try:
            if plausible(date(y, mo, d or 1), today):
                out.append(f"{y:04d}-{mo:02d}" + (f"-{d:02d}" if d else ""))
        except ValueError:
            continue
    return out


def table_row_date(text: str, today: Optional[date] = None) -> str:
    """The newest believable date written in a table row, or ''. A table row is a line with at least two '|'
    separators AND a money amount, like "Apr 08, 2024 | $694M | Series I" (a sidebar line such as
    "Webinar | 18 October 2026 | Quantum computing" has no amount and is not a table row)."""
    best = ""
    for line in text.splitlines():
        if line.count("|") >= 2 and _MONEY.search(line):
            best = max([best, *_dates_in(line, today)])
    return best


def _months_after(earlier: str, later: str) -> float:
    a, b = parse_any_date(earlier + "-01" if len(earlier) == 7 else earlier), \
        parse_any_date(later + "-01" if len(later) == 7 else later)
    return (b - a).days / 30.4375 if a and b else float("inf")


def page_date(meta_value, url: str, text: str = "", today: Optional[date] = None) -> str:
    """The page's metadata date, else the date in its URL; a table-row date in the text may lift it by at most
    12 months. No metadata and no URL date: '' (nothing is guessed from the text)."""
    anchor = best_date(meta_value, url, today)
    if not anchor:
        return ""
    row = table_row_date(text or "", today)
    if row and row > anchor and _months_after(anchor, row) <= TEXT_DATE_MAX_MONTHS:
        return row
    return anchor