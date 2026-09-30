"""The thalamus: relay, attention gate and context switch.

- Mediodorsal nucleus: infers which task context ("task set") the brain is in. A confident change
  of context is unexpected uncertainty: norepinephrine spikes and working memory is reset.
- Burst vs tonic firing: a message that violates expectations (abrupt topic change, contradiction)
  fires a "burst" that raises arousal; expected input passes in quiet tonic mode.
- Reticular nucleus: decides how widely the hippocampus should search (the recall spotlight).
"""

from __future__ import annotations

from typesafe_sdk import Choice, Noul, Question

from thalamus.core.region import BrainRegion, CycleContext

CONTEXTS = {
    "casual_chat": "Greetings, small talk, thanks, or chit-chat.",
    "factual_question": "Asks for information or an explanation that can be stated from general knowledge.",
    "reasoning": "Math, logic, puzzles, calculations, or analysis that must be worked out in steps.",
    "coding": "Writing, reviewing, explaining, or debugging code.",
    "planning": "Help planning, deciding between options, or organizing something.",
    "personal": "Shares personal information or feelings, or asks for emotional support.",
    "feedback": "Reacts to, corrects, or rates the assistant's previous reply.",
    "other": "Fits none of the other contexts.",
}

CONTEXT_SWITCH_CONFIDENCE = 0.6
BURST_THRESHOLD = 0.6


class Thalamus(BrainRegion):
    name = "thalamus"

    def questions(self, ctx: CycleContext) -> dict[str, Question]:
        return {
            "thalamus.context": Choice(
                instructions="Which kind of conversation context does latest_user_message belong to?",
                criteria=CONTEXTS,
            ),
            "thalamus.recall": Noul(
                instructions=(
                    "Does latest_user_message refer to or depend on something shared earlier, such as "
                    "the user's name, their preferences, or a topic from a past conversation?"
                ),
                criteria={
                    "true": "Answering well needs information from earlier conversations or earlier turns.",
                    "false": "The message is self-contained.",
                },
            ),
            "thalamus.surprise": Noul(
                instructions=(
                    "Does latest_user_message abruptly change topic from, or contradict, "
                    "previous_assistant_reply?"
                ),
                criteria={
                    "true": "It jumps to an unrelated topic or disputes what the assistant said.",
                    "false": (
                        "It continues, follows up on, or acknowledges the previous reply, "
                        "or there is no previous reply."
                    ),
                },
            ),
        }

    def absorb(self, ctx: CycleContext) -> None:
        context = ctx.sweep.choices["thalamus.context"]
        ctx.context_label = context.choice
        ctx.context_switched = (
            ctx.previous_context is not None
            and context.choice != ctx.previous_context
            and context.confidence >= CONTEXT_SWITCH_CONFIDENCE
        )
        if ctx.context_switched:
            ctx.modulators.nudge("norepinephrine", 0.3)

        ctx.surprise = ctx.sweep.nouls["thalamus.surprise"].p if ctx.previous_reply else 0.0
        mode = "burst" if ctx.surprise >= BURST_THRESHOLD else "tonic"
        ctx.modulators.nudge("norepinephrine", 0.2 * ctx.surprise)

        ctx.recall_breadth = ctx.sweep.nouls["thalamus.recall"].p
        # Low-confidence judgments everywhere = known unreliability (expected uncertainty): ACh up.
        if context.confidence < 0.5:
            ctx.modulators.nudge("acetylcholine", 0.15)

        ctx.trace.log(
            self.name,
            "gate",
            context=ctx.context_label,
            context_confidence=round(context.confidence, 3),
            switched=ctx.context_switched,
            firing=mode,
            surprise=round(ctx.surprise, 3),
            recall_breadth=round(ctx.recall_breadth, 3),
        )
