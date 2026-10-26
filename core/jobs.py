"""Job record: one row per background run of an agent (shown as a status chip)."""
import threading
import uuid
from dataclasses import dataclass, field
from typing import Optional

# status values: queued, running, waiting_approval, done, failed, stopped
ACTIVE = {"queued", "running", "waiting_approval"}


class JobFailed(Exception):
    """Raised by an agent that ends without success (step budget used up, stuck in a loop).
    The orchestrator shows the job as failed, with this message and no stack trace."""


@dataclass
class JobRecord:
    agent: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    status: str = "queued"
    started: Optional[float] = None
    ended: Optional[float] = None
    attempts: int = 0
    provider_used: str = ""
    error: str = ""
    stop_event: threading.Event = field(default_factory=threading.Event, repr=False)

    @property
    def active(self) -> bool:
        return self.status in ACTIVE