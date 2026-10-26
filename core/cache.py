"""SQLite cache for web results, and a small usage counter (credits) kept in usage.json.

The cache key is  tool + normalised query (or URL).  Every row stores the time it was saved,
so a lookup can say "fresh enough" (max_age_s) or "any age" (max_age_s=None: used as demo
insurance when every live provider fails).
"""
import json
import os
import sqlite3
import threading
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit, urlunsplit


def normalise_query(q: str) -> str:
    """'  Top  FinTech Startups ' and 'top fintech startups' are the same cache entry."""
    return " ".join(str(q).lower().split())


def normalise_url(url: str) -> str:
    """Lower-case scheme and host, drop the #fragment and a trailing slash."""
    p = urlsplit(url.strip())
    return urlunsplit((p.scheme.lower(), p.netloc.lower(), p.path.rstrip("/"), p.query, ""))


@dataclass
class CacheHit:
    value: dict
    provider: str
    age_s: float


class WebCache:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._conn()) as db:
            db.execute("CREATE TABLE IF NOT EXISTS web_cache ("
                       "tool TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL, "
                       "provider TEXT NOT NULL, created REAL NOT NULL, PRIMARY KEY (tool, key))")
            db.commit()

    def _conn(self):
        return sqlite3.connect(self.path, timeout=10)        # one short-lived connection per call: thread-safe

    def get(self, tool: str, key: str, max_age_s: Optional[float] = None) -> Optional[CacheHit]:
        with closing(self._conn()) as db:
            row = db.execute("SELECT value, provider, created FROM web_cache WHERE tool=? AND key=?",
                             (tool, key)).fetchone()
        if row is None:
            return None
        age = time.time() - row[2]
        if max_age_s is not None and age > max_age_s:
            return None
        return CacheHit(json.loads(row[0]), row[1], age)

    def put(self, tool: str, key: str, value: dict, provider: str) -> None:
        with closing(self._conn()) as db:
            db.execute("INSERT OR REPLACE INTO web_cache (tool, key, value, provider, created) VALUES (?,?,?,?,?)",
                       (tool, key, json.dumps(value), provider, time.time()))
            db.commit()

    def count(self) -> int:
        with closing(self._conn()) as db:
            return db.execute("SELECT COUNT(*) FROM web_cache").fetchone()[0]


# Free-tier limits shown in the dashboard (Oct 2026, from the guide; verify on the vendor pages).
LIMITS = {"tavily": 1000, "serper": 2500}


class Usage:
    """Counts calls, credits and dollars per provider and per month, saved in usage.json.
    These are LOCAL estimates (Tavily: 1 credit per basic search); compare them with the
    vendor dashboards now and then."""

    def __init__(self, path):
        self.path = Path(path)
        self._lock = threading.Lock()
        try:
            self._data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self._data = {}

    @staticmethod
    def this_month() -> str:
        return time.strftime("%Y-%m")

    def add(self, provider: str, calls: int = 1, credits: float = 0.0, dollars: float = 0.0) -> None:
        with self._lock:
            row = self._data.setdefault(provider, {}).setdefault(
                self.this_month(), {"calls": 0, "credits": 0.0, "dollars": 0.0})
            row["calls"] += calls
            row["credits"] = round(row["credits"] + credits, 4)
            row["dollars"] = round(row["dollars"] + dollars, 6)
            self._save()

    def _save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)

    def month(self, provider: str) -> dict:
        with self._lock:
            row = self._data.get(provider, {}).get(self.this_month(), {})
            return {"calls": row.get("calls", 0), "credits": row.get("credits", 0.0),
                    "dollars": row.get("dollars", 0.0)}

    def total_calls(self, provider: str) -> int:
        with self._lock:
            return sum(r.get("calls", 0) for r in self._data.get(provider, {}).values())

    def lines(self) -> dict:
        """provider -> one readable line for the dashboard."""
        t, e = self.month("tavily"), self.month("exa")
        return {
            "tavily": f"Tavily credits this month: {t['credits']:.1f} / {LIMITS['tavily']} ({t['calls']} calls)",
            "exa": f"Exa this month: {e['calls']} searches, about ${e['dollars']:.3f} spent",
            "serper": f"Serper queries used: {self.total_calls('serper')} / {LIMITS['serper']} (one-time allowance)",
            "jina": f"Jina Reader calls this month: {self.month('jina')['calls']} (keyless limit about 20 per minute)",
            "ddgs": f"ddgs searches this month: {self.month('ddgs')['calls']} (unofficial, unlimited but rate-limited)",
        }