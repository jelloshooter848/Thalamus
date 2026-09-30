"""TypeSafe's recommended patterns for acting on JEV answers, as small pure functions.

Policy lives in code: JEV supplies judgments, these helpers decide what the code may do with them.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

Gate = Literal["act", "confirm", "defer"]


def confidence_gate(confidence: float, act_at: float, confirm_at: float | None = None) -> Gate:
    """Confidence-gated routing: the bar to act rises with the consequences of being wrong."""
    if confidence >= act_at:
        return "act"
    if confirm_at is not None and confidence >= confirm_at:
        return "confirm"
    return "defer"


def noul_gate(p: float, yes_at: float, no_at: float) -> Literal["yes", "no", "unsure"]:
    """Nouls carry no confidence field; distance from 0.5 is the certainty."""
    if p >= yes_at:
        return "yes"
    if p <= no_at:
        return "no"
    return "unsure"


def composite_score(values: Mapping[str, float], weights: Mapping[str, float]) -> float:
    """Composite scoring: weight normalized (0-1) judgments and facts; re-weight without re-asking JEV."""
    total = sum(weights.values())
    if total <= 0:
        return 0.0
    return sum(values.get(name, 0.0) * weight for name, weight in weights.items()) / total


def disagreement(a: float, b: float) -> float:
    """Conflict between two System-1 probes framed differently but meant to agree."""
    return abs(a - b)
