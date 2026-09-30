"""Neuromodulators as a handful of global scalars (after Doya 2002, Aston-Jones & Cohen, Yu & Dayan).

- dopamine: reward prediction error; drives learning of action and arbitration values.
- norepinephrine: arousal / unexpected uncertainty; narrows attention, resets on context switch.
- serotonin: patience / time horizon; makes slow deliberation feel cheaper.
- acetylcholine: expected uncertainty; sets the learning rate and memory encoding strength.

Levels live in [0, 1] and relax toward baseline every cycle; they tune thresholds, they never decide.
"""

from __future__ import annotations

from dataclasses import dataclass, field

BASELINES = {"dopamine": 0.5, "norepinephrine": 0.3, "serotonin": 0.5, "acetylcholine": 0.4}
DECAY = 0.6  # fraction of the deviation from baseline kept each cycle


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass
class Neuromodulators:
    levels: dict[str, float] = field(default_factory=lambda: dict(BASELINES))
    last_rpe: float = 0.0

    def relax(self) -> None:
        for name, baseline in BASELINES.items():
            self.levels[name] = baseline + (self.levels[name] - baseline) * DECAY

    def nudge(self, name: str, delta: float) -> None:
        self.levels[name] = _clamp(self.levels[name] + delta)

    def reward(self, rpe: float) -> None:
        self.last_rpe = rpe
        self.nudge("dopamine", 0.5 * rpe)

    @property
    def dopamine(self) -> float:
        return self.levels["dopamine"]

    @property
    def norepinephrine(self) -> float:
        return self.levels["norepinephrine"]

    @property
    def serotonin(self) -> float:
        return self.levels["serotonin"]

    @property
    def acetylcholine(self) -> float:
        return self.levels["acetylcholine"]

    @property
    def learning_rate(self) -> float:
        return 0.1 + 0.4 * self.acetylcholine

    def workspace_capacity(self, base: int) -> int:
        """High arousal narrows the spotlight (fewer items reach consciousness)."""
        return max(3, round(base * (1.25 - 0.6 * self.norepinephrine)))
