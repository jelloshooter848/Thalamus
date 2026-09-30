"""JEV (TypeSafe AI's System One model) as THALAMUS's fast decision substrate.

All questions a cycle needs are sent in a single `system_one` call ("speculative fan-out"):
JEV answers them in parallel, so extra questions cost tokens but almost no latency.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from typesafe_sdk import (
    AsyncTypeSafeClient,
    ChoiceAnswer,
    NoulAnswer,
    Question,
    ScoreAnswer,
)

from thalamus.config import require_jev_key
from thalamus.providers.base import ChoiceResult, Decision, NoulResult, ScoreResult

# Keep any single text field well inside JEV's 32k state budget.
MAX_FIELD_CHARS = 4000


def clip(text: str, limit: int = MAX_FIELD_CHARS) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


class JevProvider:
    def __init__(self, model: str = "jev-latest", client: AsyncTypeSafeClient | None = None) -> None:
        self._client = client or AsyncTypeSafeClient(api_key=require_jev_key(), model=model)

    async def decide(self, state: Mapping[str, Any], questions: Mapping[str, Question]) -> Decision:
        if not questions:
            return Decision()
        started = time.perf_counter()
        response = await self._client.system_one(state=dict(state), questions=dict(questions))
        decision = Decision(
            input_tokens=response.usage.input_tokens or 0,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        for name, answer in response.answers.items():
            if isinstance(answer, NoulAnswer):
                decision.nouls[name] = NoulResult(p=answer.noul)
            elif isinstance(answer, ChoiceAnswer):
                decision.choices[name] = ChoiceResult(
                    choice=answer.choice,
                    confidence=answer.confidence,
                    probabilities=dict(answer.probabilities),
                )
            elif isinstance(answer, ScoreAnswer):
                decision.scores[name] = ScoreResult(
                    score=answer.score,
                    confidence=answer.confidence,
                    probabilities=dict(answer.probabilities),
                    levels=len(answer.legend),
                )
        return decision

    async def aclose(self) -> None:
        await self._client.aclose()
