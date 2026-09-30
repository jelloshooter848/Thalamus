"""Brain regions are functions over one shared cycle context, not chatting personas.

System-1 regions contribute typed questions to a single batched JEV "subcortical sweep" and then
absorb the answers. Regions never call one another; they communicate by posting Signals to the
global workspace and by writing their findings onto the cycle context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from typesafe_sdk import Question

from thalamus.config import Settings
from thalamus.core.homeostasis import Homeostasis
from thalamus.core.memory_store import MemoryStore
from thalamus.core.neuromodulators import Neuromodulators
from thalamus.core.trace import CycleTrace
from thalamus.core.workspace import GlobalWorkspace
from thalamus.providers.base import Decision


@dataclass
class Turn:
    role: str  # "user" | "assistant"
    content: str


@dataclass
class LastAction:
    """What the brain did last turn: the thing the striatal critic learns about."""

    context: str
    action: str
    path: str


@dataclass
class Appraisal:
    valence: float = 0.0  # -1 negative .. +1 positive
    urgency: float = 0.0  # 0..1
    stakes: float = 0.0  # 0..1

    @property
    def arousal(self) -> float:
        return max(self.urgency, self.stakes)


@dataclass
class Arbitration:
    path: str  # "fast" | "slow"
    reasons: list[str]
    demand: float = 0.0
    value: float = 0.0
    cost: float = 0.0
    conflict: float = 0.0


@dataclass
class CycleContext:
    turn: int
    session: str
    message: str
    history: list[Turn]
    settings: Settings
    workspace: GlobalWorkspace
    modulators: Neuromodulators
    homeostasis: Homeostasis
    memory: MemoryStore
    trace: CycleTrace
    last_action: LastAction | None = None
    previous_context: str | None = None

    # Filled in as the cycle runs.
    state: dict[str, Any] = field(default_factory=dict)  # JEV-facing structured percept
    sweep: Decision = field(default_factory=Decision)
    context_label: str = "other"
    context_switched: bool = False
    surprise: float = 0.0
    recall_breadth: float = 0.0
    appraisal: Appraisal = field(default_factory=Appraisal)
    importance: float = 0.0
    action: str = "respond"
    arbitration: Arbitration | None = None
    thought: str | None = None

    @property
    def previous_reply(self) -> str | None:
        for turn in reversed(self.history):
            if turn.role == "assistant":
                return turn.content
        return None


class BrainRegion:
    name: str = "region"

    def questions(self, ctx: CycleContext) -> dict[str, Question]:
        """Questions for the batched System-1 sweep, keyed `<region>.<name>`."""
        return {}

    def absorb(self, ctx: CycleContext) -> None:
        """Read this region's answers from `ctx.sweep` and update the brain."""
