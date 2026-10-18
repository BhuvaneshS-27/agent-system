"""One object shared by every tab. Approvals will be added in Step 5."""
from dataclasses import dataclass

from core.bus import EventBus
from core.orchestrator import Orchestrator
from core.settings import Settings


@dataclass
class AppContext:
    settings: Settings
    bus: EventBus
    orchestrator: Orchestrator