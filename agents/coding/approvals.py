"""Approval gate: the agent thread waits here until the user clicks Approve or Deny.

The agent thread only sleeps on an Event (it never blocks the page). The Approve and Deny
buttons call answer(). Every request and its result is an event, so the log shows exactly
what was asked and what the user decided.
"""
import threading
import time
import uuid

APPROVAL_TIMEOUT = 300      # seconds before an unanswered request counts as denied


class Approvals:
    def __init__(self, bus, timeout: float = APPROVAL_TIMEOUT):
        self.bus = bus
        self.timeout = timeout
        self._pending: dict = {}
        self._lock = threading.Lock()

    def ask(self, job, step: int, command: str, timeout=None) -> str:
        """Blocks. Returns 'approved', 'denied', 'timeout' or 'stopped' (Stop was pressed)."""
        timeout = self.timeout if timeout is None else timeout
        rid = uuid.uuid4().hex[:8]
        rec = {"event": threading.Event(), "ok": False}
        with self._lock:
            self._pending[rid] = rec
        job.status = "waiting_approval"
        self.bus.emit("approval_request", job=job.id, agent=job.agent, step=step, id=rid,
                      command=command, timeout_s=timeout)
        result = "timeout"
        deadline = time.time() + timeout
        try:
            while time.time() < deadline:
                if job.stop_event.is_set():
                    result = "stopped"
                    break
                if rec["event"].wait(0.2):
                    result = "approved" if rec["ok"] else "denied"
                    break
        finally:
            with self._lock:
                self._pending.pop(rid, None)
            if job.status == "waiting_approval":
                job.status = "running"
            self.bus.emit("approval_result", job=job.id, agent=job.agent, step=step, id=rid,
                          command=command, result=result, ok=result == "approved")
        return result

    def answer(self, rid: str, ok: bool) -> bool:
        """Called by the Approve / Deny buttons. False if the request is already closed."""
        with self._lock:
            rec = self._pending.get(rid)
        if rec is None:
            return False
        rec["ok"] = bool(ok)
        rec["event"].set()
        return True

    def is_pending(self, rid: str) -> bool:
        with self._lock:
            return rid in self._pending