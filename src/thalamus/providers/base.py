"""Provider interfaces. Regions depend on these, never on a vendor SDK directly.

Two kinds of intelligence feed the brain:
- a DecisionProvider (System 1): typed judgments with calibrated probabilities (JEV);
- a LanguageProvider (System 2 / speech): generated language (Claude).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from typesafe_sdk import Question


@dataclass(frozen=True)
class NoulResult:
    p: float  # probability of "yes"; JEV returns no separate confidence for a noul


@dataclass(frozen=True)
class ChoiceResult:
    choice: str
    confidence: float
    probabilities: dict[str, float]

    def p(self, label: str) -> float:
        return self.probabilities.get(label, 0.0)


@dataclass(frozen=True)
class ScoreResult:
    score: float  # probability-weighted level, may fall between levels
    confidence: float
    probabilities: dict[int, float]
    levels: int

    @property
    def normalized(self) -> float:
        return self.score / (self.levels - 1) if self.levels > 1 else 0.0


@dataclass
class Decision:
    nouls: dict[str, NoulResult] = field(default_factory=dict)
    choices: dict[str, ChoiceResult] = field(default_factory=dict)
    scores: dict[str, ScoreResult] = field(default_factory=dict)
    input_tokens: int = 0
    latency_ms: float = 0.0


class DecisionProvider(Protocol):
    async def decide(self, state: Mapping[str, Any], questions: Mapping[str, Question]) -> Decision: ...


@dataclass(frozen=True)
class Generation:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    refused: bool = False


Tier = Literal["fast", "deep"]


TextSink = Callable[[str], Awaitable[None]]  # receives reply text as it streams


class LanguageProvider(Protocol):
    async def generate(
        self,
        *,
        tier: Tier,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        on_text: TextSink | None = None,
    ) -> Generation: ...
