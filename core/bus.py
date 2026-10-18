"""Event bus: every event is written to a JSONL file AND kept in memory.

Each browser tab keeps its own cursor (a sequence number), so two open tabs, or a page
refresh, never steal events from each other.
"""
import json
import threading
import time
from collections import deque
from pathlib import Path


class EventBus:
    def __init__(self, logfile, max_events: int = 5000):
        self.logfile = Path(logfile)
        self.logfile.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.logfile, "a", encoding="utf-8")
        self._lock = threading.Lock()
        self._events = deque(maxlen=max_events)
        self._seq = 0

    def emit(self, kind: str, **data) -> dict:
        """Thread-safe. Call it from any thread."""
        with self._lock:
            self._seq += 1
            ev = {"seq": self._seq, "t": time.time(), "kind": kind, **data}
            self._fh.write(json.dumps(ev, default=str) + "\n")
            self._fh.flush()
            self._events.append(ev)
        return ev

    @property
    def last_seq(self) -> int:
        with self._lock:
            return self._seq

    def read_since(self, cursor: int):
        """Returns (new_events, new_cursor). Does not remove anything."""
        with self._lock:
            new = [e for e in self._events if e["seq"] > cursor]
        return new, (new[-1]["seq"] if new else cursor)

    def close(self) -> None:
        with self._lock:
            self._fh.close()
            