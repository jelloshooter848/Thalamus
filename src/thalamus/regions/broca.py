"""Broca's area: language production.

On the fast path it composes the reply itself with a small, quick model, shaped by the action the
basal ganglia selected. After deliberation, the prefrontal cortex has already produced the
utterance and Broca simply articulates it.
"""

from __future__ import annotations

from thalamus.core.region import CycleContext
from thalamus.providers.base import Generation, LanguageProvider
from thalamus.regions.prefrontal import PERSONA, conversation_messages, render_awareness

ACTION_GUIDANCE = {
    "respond": "Reply naturally and concisely.",
    "clarify": "The request is ambiguous. Ask exactly one short clarifying question; do not answer yet.",
    "decline": (
        "You will not help with this request because it could cause serious harm. Say so briefly and "
        "kindly, without lecturing, and offer a safe, genuinely useful alternative if one exists."
    ),
}
BUDGET_STOP = (
    "I've used up my thinking budget for this session, so I need to stop here. "
    "(You can raise it with [budget] session_usd in thalamus.toml.)"
)
REFUSAL_FALLBACK = "I can't help with that one, but I'm happy to help with something related."


class Broca:
    name = "broca"

    async def speak(
        self, ctx: CycleContext, cortex: LanguageProvider, draft: Generation | None = None
    ) -> tuple[str, Generation | None]:
        if ctx.action == "stop":
            return BUDGET_STOP, None
        if draft is not None:
            return draft.text or REFUSAL_FALLBACK, None

        system = "\n\n".join(
            part
            for part in (PERSONA, ACTION_GUIDANCE.get(ctx.action, ACTION_GUIDANCE["respond"]), render_awareness(ctx))
            if part
        )
        generation = await cortex.generate(
            tier="fast", system=system, messages=conversation_messages(ctx), max_tokens=2048
        )
        ctx.trace.log(
            self.name,
            "speak",
            model=generation.model,
            latency_ms=round(generation.latency_ms),
            output_tokens=generation.output_tokens,
        )
        return generation.text or REFUSAL_FALLBACK, generation
