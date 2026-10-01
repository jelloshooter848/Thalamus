"""The shared, typed language every brain region speaks.

Following Sherman & Guillery's driver/modulator split, a signal's `content` is the driver (what it
says) and its modulators (salience, confidence, urgency) only change how strongly it competes.
Regions never exchange free text with each other; they post Signals to the global workspace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

SignalKind = Literal["percept", "memory", "appraisal", "context", "thought", "web", "self", "fact", "mail"]


@dataclass
class Signal:
    source: str  # region that produced it
    kind: SignalKind
    content: str  # the driver
    key: str = ""  # identity for inhibition-of-return and working-memory maintenance
    salience: float = 0.5
    confidence: float = 1.0
    urgency: float = 0.0
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.key:
            self.key = f"{self.source}:{self.kind}:{hash(self.content)}"
