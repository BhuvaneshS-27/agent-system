"""One object shared by every tab."""
from dataclasses import dataclass

from agents.coding.approvals import Approvals
from core.bus import EventBus
from core.orchestrator import Orchestrator
from core.settings import Settings
from research.web import WebTools


@dataclass
class AppContext:
    settings: Settings
    bus: EventBus
    orchestrator: Orchestrator
    approvals: Approvals
    web: WebTools