"""Structured record of one cognitive cycle: every gate, judgment and arbitration, with its inputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TraceEvent:
    region: str
    event: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class CycleTrace:
    turn: int
    events: list[TraceEvent] = field(default_factory=list)

    def log(self, region: str, event: str, **data: Any) -> None:
        self.events.append(TraceEvent(region, event, data))

    def find(self, region: str, event: str | None = None) -> list[TraceEvent]:
        return [e for e in self.events if e.region == region and (event is None or e.event == event)]
