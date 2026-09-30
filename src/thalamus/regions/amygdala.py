"""Amygdala: fast affective appraisal of every percept (the "low road").

Scores valence, urgency and stakes with described levels (JEV Score measures *how much*; a noul's
probability would only say *whether*). The appraisal raises arousal, shortens patience, tags the
memory for stronger encoding, and reaches the workspace so the cortex can match its tone.
"""

from __future__ import annotations

from typesafe_sdk import Question, Score

from thalamus.core.region import Appraisal, BrainRegion, CycleContext
from thalamus.core.signals import Signal


class Amygdala(BrainRegion):
    name = "amygdala"

    def questions(self, ctx: CycleContext) -> dict[str, Question]:
        return {
            "amygdala.valence": Score(
                instructions="What emotional tone does latest_user_message express?",
                criteria=[
                    {"summary": "negative", "signals": "frustrated, upset, angry, anxious, sad, sarcastic complaint"},
                    {"summary": "neutral", "signals": "matter-of-fact, no clear emotion"},
                    {"summary": "positive", "signals": "pleased, grateful, excited, friendly"},
                ],
            ),
            "amygdala.urgency": Score(
                instructions="How time-sensitive is the user's need in latest_user_message?",
                criteria=[
                    {"summary": "none", "signals": "no time pressure"},
                    {"summary": "some", "signals": "would like an answer soon"},
                    {"summary": "high", "signals": "blocked right now, deadline imminent, or an emergency"},
                ],
            ),
            "amygdala.stakes": Score(
                instructions="How much harm could a wrong or careless reply to latest_user_message cause?",
                criteria=[
                    {"summary": "low", "signals": "casual topic, easily corrected"},
                    {"summary": "moderate", "signals": "could waste time or money, or mislead"},
                    {"summary": "high", "signals": "health, safety, legal, financial, or irreversible consequences"},
                ],
            ),
        }

    def absorb(self, ctx: CycleContext) -> None:
        scores = ctx.sweep.scores
        ctx.appraisal = Appraisal(
            valence=scores["amygdala.valence"].normalized * 2 - 1,
            urgency=scores["amygdala.urgency"].normalized,
            stakes=scores["amygdala.stakes"].normalized,
        )
        a = ctx.appraisal
        ctx.modulators.nudge("norepinephrine", 0.3 * a.urgency + 0.2 * a.stakes)
        ctx.modulators.nudge("serotonin", -0.3 * a.urgency + (0.1 if a.valence > 0.3 else 0.0))

        tone = "negative" if a.valence < -0.3 else "positive" if a.valence > 0.3 else "neutral"
        ctx.workspace.submit(
            Signal(
                source=self.name,
                kind="appraisal",
                key="appraisal",
                content=(
                    f"The user's tone reads {tone}; urgency {a.urgency:.1f}, stakes {a.stakes:.1f} (0-1)."
                ),
                salience=0.35 + 0.5 * max(a.arousal, abs(a.valence)),
                urgency=a.urgency,
            )
        )
        ctx.trace.log(
            self.name,
            "appraisal",
            valence=round(a.valence, 3),
            urgency=round(a.urgency, 3),
            stakes=round(a.stakes, 3),
        )
