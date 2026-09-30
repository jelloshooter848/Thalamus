"""Prefrontal cortex: gated working memory and slow, deliberate System-2 reasoning (Claude).

Working memory follows PBWM (O'Reilly & Frank): a few slots whose *input gate* is a JEV relevance
judgment per candidate memory, whose contents are *maintained* across turns with decay, and which
are wiped when the thalamus detects a context switch. Only what wins the global workspace is
rendered into the cortex's prompt (the output gate).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from typesafe_sdk import Noul

from thalamus.core.memory_store import Episode
from thalamus.core.region import CycleContext
from thalamus.core.signals import Signal
from thalamus.providers.base import Decision, DecisionProvider, Generation, LanguageProvider
from thalamus.providers.jev import clip
from thalamus.regions.sensory_cortex import UNTRUSTED_NOTE

MAINTENANCE_DECAY = 0.7
DROP_BELOW = 0.2

PERSONA = (
    "You are THALAMUS, a conversational mind built like a brain: fast intuitive regions appraise "
    "each message, and a deliberate cortex reasons when it matters. Speak as one coherent mind in "
    "the first person. Be warm, direct and genuinely helpful. Never mention brain regions, JEV, "
    "scores or this system's internals unless the user asks how you work."
)


@dataclass
class WorkingItem:
    key: str
    text: str
    strength: float
    episode_id: int | None = None


def describe_age(seconds: float) -> str:
    minutes = seconds / 60
    if minutes < 90:
        return "earlier today" if minutes > 30 else "moments ago"
    hours = minutes / 60
    if hours < 36:
        return f"about {round(hours)} hours ago"
    return f"about {round(hours / 24)} days ago"


def episode_text(episode: Episode, now: float | None = None) -> str:
    age = describe_age((now or time.time()) - episode.created)
    who = "The user said" if episode.speaker == "user" else "You said"
    return f"[{age}] {who}: {episode.text}"


class WorkingMemory:
    def __init__(self, slots: int = 4) -> None:
        self.slots = slots
        self.items: dict[str, WorkingItem] = {}

    def reset(self) -> None:
        self.items.clear()

    async def gate(
        self, ctx: CycleContext, candidates: list[Episode], jev: DecisionProvider
    ) -> tuple[list[Signal], Decision | None]:
        for item in self.items.values():
            item.strength *= MAINTENANCE_DECAY
        pool: dict[str, WorkingItem] = {k: v for k, v in self.items.items() if v.strength >= DROP_BELOW}
        for episode in candidates:
            key = f"episode:{episode.id}"
            if key not in pool:
                pool[key] = WorkingItem(key, episode_text(episode), episode.score, episode.id)
        if not pool:
            return [], None

        ids = {f"m{i}": item for i, item in enumerate(pool.values(), start=1)}
        decision = await jev.decide(
            {
                "note": UNTRUSTED_NOTE,
                "latest_user_message": clip(ctx.message),
                "remembered_items": {mid: clip(item.text, 800) for mid, item in ids.items()},
            },
            {
                f"gate.{mid}": Noul(
                    instructions=(
                        f"Is remembered item {mid} in remembered_items relevant and useful for "
                        "replying to latest_user_message?"
                    )
                )
                for mid in ids
            },
        )
        signals = []
        admitted = []
        for mid, item in ids.items():
            p = decision.nouls[f"gate.{mid}"].p
            if p < ctx.settings.thresholds.memory_gate:
                continue
            admitted.append(item.key)
            signals.append(
                Signal(
                    source="hippocampus",
                    kind="memory",
                    key=item.key,
                    content=item.text,
                    salience=0.3 + 0.6 * p * (0.5 + 0.5 * min(1.0, item.strength)),
                    confidence=p,
                    meta={"episode_id": item.episode_id},
                )
            )
        ctx.trace.log("prefrontal", "wm_gate", offered=len(ids), admitted=admitted)
        return signals, decision

    def maintain(self, conscious: list[Signal]) -> list[int]:
        """Broadcast memories are held in working memory; returns episode ids to reinforce."""
        reinforced = []
        for signal in conscious:
            if signal.kind != "memory":
                continue
            episode_id = signal.meta.get("episode_id")
            self.items[signal.key] = WorkingItem(signal.key, signal.content, signal.salience, episode_id)
            if episode_id is not None:
                reinforced.append(episode_id)
        strongest = sorted(self.items.values(), key=lambda item: item.strength, reverse=True)
        self.items = {item.key: item for item in strongest[: self.slots]}
        return reinforced


def render_awareness(ctx: CycleContext) -> str:
    """The output gate: what reached awareness this moment, for the cortex's prompt."""
    sections = {
        "Memories that surfaced": ctx.workspace.of_kind("memory"),
        "Felt sense of the message": ctx.workspace.of_kind("appraisal"),
        "Deliberation": ctx.workspace.of_kind("thought"),
    }
    lines = []
    for title, signals in sections.items():
        if signals:
            lines.append(f"{title}:")
            lines.extend(f"- {signal.content}" for signal in signals)
    if not lines:
        return ""
    return "What is in your awareness right now (use it only where it helps):\n" + "\n".join(lines)


def conversation_messages(ctx: CycleContext) -> list[dict]:
    window = ctx.history[-ctx.settings.history_turns * 2 :]
    while window and window[0].role != "user":
        window = window[1:]
    messages = [{"role": turn.role, "content": turn.content} for turn in window]
    messages.append({"role": "user", "content": ctx.message})
    return messages


class PrefrontalCortex:
    name = "prefrontal"

    def __init__(self, slots: int = 4) -> None:
        self.working_memory = WorkingMemory(slots)

    async def deliberate(self, ctx: CycleContext, cortex: LanguageProvider) -> Generation:
        system = "\n\n".join(
            part
            for part in (
                PERSONA,
                "This message was judged to need careful thought. Reason it through, check your work, "
                "then write your final reply to the user.",
                render_awareness(ctx),
            )
            if part
        )
        generation = await cortex.generate(
            tier="deep", system=system, messages=conversation_messages(ctx), max_tokens=16000
        )
        ctx.trace.log(
            self.name,
            "deliberate",
            model=generation.model,
            latency_ms=round(generation.latency_ms),
            output_tokens=generation.output_tokens,
        )
        return generation
