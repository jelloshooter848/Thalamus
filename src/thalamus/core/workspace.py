"""Global workspace: a capacity-limited stage where signals compete and winners are broadcast.

Only broadcast content reaches the cortex (Claude's prompt) and is eligible for learning, which is
what keeps System 2's context small and focused. Recently broadcast items are mildly inhibited
(inhibition of return) unless refreshed, so attention doesn't fixate.
"""

from __future__ import annotations

from thalamus.core.signals import Signal

INHIBITION_OF_RETURN = 0.25


class GlobalWorkspace:
    def __init__(self, capacity: int = 7) -> None:
        self.capacity = capacity
        self.candidates: list[Signal] = []
        self.conscious: list[Signal] = []
        self._last_broadcast: set[str] = set()

    def submit(self, signal: Signal) -> None:
        self.candidates.append(signal)

    def compete(self, capacity: int | None = None) -> list[Signal]:
        capacity = capacity or self.capacity
        best: dict[str, tuple[float, Signal]] = {}
        for signal in self.candidates:
            strength = signal.salience
            if signal.key in self._last_broadcast and signal.kind == "memory":
                strength *= 1 - INHIBITION_OF_RETURN
            if signal.key not in best or strength > best[signal.key][0]:
                best[signal.key] = (strength, signal)
        ranked = sorted(best.values(), key=lambda pair: pair[0], reverse=True)
        self.conscious = [signal for _, signal in ranked[:capacity]]
        self._last_broadcast = {signal.key for signal in self.conscious}
        self.candidates = []
        return self.conscious

    def of_kind(self, kind: str) -> list[Signal]:
        return [signal for signal in self.conscious if signal.kind == kind]
