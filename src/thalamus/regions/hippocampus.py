"""Hippocampus: fast episodic encoding and cue-driven recall.

Encoding strength comes from the amygdala (arousal), JEV's importance judgment and acetylcholine.
Recall combines retention (a forgetting curve reinforced by use), importance and relevance to the
cue; when the thalamus says the message leans on the past, the strongest traces also surface even
without a lexical match (pattern completion).
"""

from __future__ import annotations

from typesafe_sdk import Question, Score

from thalamus.core.memory_store import Episode
from thalamus.core.region import BrainRegion, CycleContext

RECALL_BREADTH_WIDE = 0.6


class Hippocampus(BrainRegion):
    name = "hippocampus"

    def questions(self, ctx: CycleContext) -> dict[str, Question]:
        return {
            "hippocampus.importance": Score(
                instructions="How important is latest_user_message to remember in future conversations?",
                criteria=[
                    {"summary": "trivial", "signals": "small talk or a one-off detail"},
                    {"summary": "useful", "signals": "context that may help later"},
                    {
                        "summary": "important",
                        "signals": "a lasting fact about the user: name, preferences, goals, commitments",
                    },
                ],
            )
        }

    def absorb(self, ctx: CycleContext) -> None:
        ctx.importance = ctx.sweep.scores["hippocampus.importance"].normalized

    def recall(self, ctx: CycleContext) -> list[Episode]:
        settings = ctx.settings
        wide = ctx.recall_breadth >= RECALL_BREADTH_WIDE
        k = settings.memory.recall_k * (2 if wide else 1)
        episodes = ctx.memory.recall(
            ctx.message,
            k=k,
            exclude_session=ctx.session,
            exclude_from_turn=max(0, ctx.turn - settings.history_turns // 2),
            include_salient=wide,
            stability_hours=settings.memory.base_stability_hours,
        )
        ctx.trace.log(
            self.name,
            "recall",
            cue_wide=wide,
            candidates=[{"id": e.id, "score": round(e.score, 3), "text": e.text[:80]} for e in episodes],
        )
        return episodes

    def encode(self, ctx: CycleContext, reply: str) -> None:
        a = ctx.appraisal
        # Arousal and acetylcholine strengthen encoding; surprising (dopamine-flagged) turns too.
        strength = min(
            1.0,
            ctx.importance * (0.7 + 0.6 * ctx.modulators.acetylcholine)
            + 0.2 * a.arousal
            + 0.1 * abs(ctx.modulators.last_rpe),
        )
        common = dict(session=ctx.session, turn=ctx.turn, context=ctx.context_label)
        ctx.memory.encode(
            speaker="user", text=ctx.message, importance=strength, valence=a.valence, arousal=a.arousal, **common
        )
        if reply:
            ctx.memory.encode(speaker="assistant", text=reply, importance=0.3 * strength, **common)
        ctx.trace.log(self.name, "encode", strength=round(strength, 3))
