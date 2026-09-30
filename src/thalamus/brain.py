"""The Brain: wires the regions together and runs one cognitive cycle per message.

sense → subcortical sweep (one batched JEV call) → learn from the last turn → recall → gate working
memory → global workspace competition → select action → arbitrate fast/slow → deliberate? → speak
→ encode.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass

from thalamus.config import Settings, require_jev_key
from thalamus.core.homeostasis import Homeostasis
from thalamus.core.memory_store import MemoryStore
from thalamus.core.neuromodulators import Neuromodulators
from thalamus.core.region import BrainRegion, CycleContext, LastAction, Turn
from thalamus.core.trace import CycleTrace
from thalamus.core.workspace import GlobalWorkspace
from thalamus.providers.base import Decision, DecisionProvider, Generation, LanguageProvider
from thalamus.providers.claude import ClaudeProvider
from thalamus.providers.jev import JevProvider
from thalamus.providers.search import SearchProvider, TavilySearch, search_key
from thalamus.regions.acc import AnteriorCingulate
from thalamus.regions.amygdala import Amygdala
from thalamus.regions.basal_ganglia import BasalGanglia
from thalamus.regions.broca import Broca
from thalamus.regions.hippocampus import Hippocampus
from thalamus.regions.prefrontal import PrefrontalCortex, buffer_window
from thalamus.regions.self_model import SelfModel
from thalamus.regions.sensory_cortex import SensoryCortex
from thalamus.regions.thalamus import Thalamus
from thalamus.regions.web_sense import WebSense


@dataclass
class Response:
    text: str
    action: str
    path: str
    context: str
    trace: CycleTrace
    cost_usd: float
    modulators: dict[str, float]
    sources: list[dict]


class Brain:
    def __init__(
        self,
        settings: Settings,
        *,
        jev: DecisionProvider,
        cortex: LanguageProvider,
        memory: MemoryStore,
        clock: Callable[[], float] = time.time,
        search: SearchProvider | None = None,
    ) -> None:
        self.settings = settings
        self.jev = jev
        self.cortex = cortex
        self.memory = memory
        self.clock = clock
        self.session = uuid.uuid4().hex[:12]
        self.history: list[Turn] = []
        self.conversation_started = clock()
        self.last_activity: float | None = None
        self.modulators = Neuromodulators()
        self.homeostasis = Homeostasis(settings.budget)
        self.workspace = GlobalWorkspace(settings.workspace_capacity)

        self.sensory = SensoryCortex()
        self.thalamus = Thalamus()
        self.amygdala = Amygdala()
        self.hippocampus = Hippocampus()
        self.acc = AnteriorCingulate()
        self.basal_ganglia = BasalGanglia()
        self.prefrontal = PrefrontalCortex()
        self.broca = Broca()
        self.web = WebSense(search)
        self.self_model = SelfModel()
        self.subcortical: list[BrainRegion] = [
            self.thalamus,
            self.amygdala,
            self.hippocampus,
            self.acc,
            self.basal_ganglia,
            self.web,
            self.self_model,
        ]

        self._turn = 0
        self._last_action: LastAction | None = None
        self._last_context: str | None = None

    def new_conversation(self) -> None:
        """Start a fresh conversation. Long-term memory and learned values are kept, so earlier
        conversations stay reachable, but only through recall, the way a person's are."""
        self.session = uuid.uuid4().hex[:12]
        self.history = []
        self.conversation_started = self.clock()
        self.last_activity = None
        self.prefrontal.working_memory.reset()
        self._turn = 0
        self._last_action = None
        self._last_context = None

    def snapshot(self) -> dict:
        """What the mind currently holds, for the Mind view."""
        idle_limit = self.settings.conversation_idle_minutes * 60
        return {
            "conversation": {
                "id": self.session,
                "started": self.conversation_started,
                "last_activity": self.last_activity,
                "turns": len(self.history) // 2,
                "idle_minutes": self.settings.conversation_idle_minutes,
                "rolls_over_at": self.last_activity + idle_limit if self.last_activity else None,
            },
            "buffer": [
                {"role": turn.role, "content": turn.content}
                for turn in buffer_window(self.history, self.settings.history_turns)
            ],
            "working_memory": [
                asdict(item)
                for item in sorted(
                    self.prefrontal.working_memory.items.values(), key=lambda item: item.strength, reverse=True
                )
            ],
            "recall": self.prefrontal.working_memory.last_gate,
            "web": self.web.last,
            "modulators": dict(self.modulators.levels),
            "budget": {"spent_usd": self.homeostasis.spent_usd, "limit_usd": self.settings.budget.session_usd},
        }

    async def think(self, message: str) -> Response:
        now = self.clock()
        idle = self.settings.conversation_idle_minutes * 60
        rolled_over = self.last_activity is not None and now - self.last_activity > idle
        if rolled_over:
            self.new_conversation()
        self._turn += 1
        trace = CycleTrace(turn=self._turn)
        if rolled_over:
            trace.log("hippocampus", "new_conversation", reason=f"idle for over {idle / 60:.0f} minutes")
        spent_before = self.homeostasis.spent_usd
        self.modulators.relax()
        ctx = CycleContext(
            turn=self._turn,
            session=self.session,
            message=message,
            history=list(self.history),
            settings=self.settings,
            workspace=self.workspace,
            modulators=self.modulators,
            homeostasis=self.homeostasis,
            memory=self.memory,
            trace=trace,
            last_action=self._last_action,
            previous_context=self._last_context,
            now=now,
            conversation_started=self.conversation_started,
            web_enabled=self.web.enabled,
        )

        # Sense, then one parallel System-1 sweep shared by every subcortical region.
        self.sensory.perceive(ctx)
        questions = {}
        for region in self.subcortical:
            questions.update(region.questions(ctx))
        ctx.sweep = await self.jev.decide(ctx.state, questions)
        self._account_jev(ctx.sweep, trace, "sweep", len(questions))

        self.basal_ganglia.learn(ctx)  # dopamine from the reaction to the previous reply
        for region in self.subcortical:
            region.absorb(ctx)
        if ctx.context_switched:
            self.prefrontal.working_memory.reset()

        ctx.action = self.basal_ganglia.select(ctx)

        # Two parallel streams feed awareness: memories recalled from the past, and the web.
        # Each is gated by JEV before it may compete for the global workspace.
        candidates = self.hippocampus.recall(ctx)
        (memory_signals, gate_decision), web = await asyncio.gather(
            self.prefrontal.working_memory.gate(ctx, candidates, self.jev),
            self.web.sense(ctx, self.cortex, self.jev),
        )
        if gate_decision is not None:
            self._account_jev(gate_decision, trace, "wm_gate", len(gate_decision.nouls))
        if web.query_generation is not None:
            self._account_llm(web.query_generation)
        if web.billable_calls:
            self.homeostasis.record_usd("tavily", web.billable_calls * self.settings.web.price_per_call_usd)
        if web.gate_decision is not None:
            self._account_jev(web.gate_decision, trace, "web_gate", len(web.gate_decision.nouls))
        for signal in memory_signals + web.signals:
            self.workspace.submit(signal)
        capacity = self.modulators.workspace_capacity(self.settings.workspace_capacity)
        conscious = self.workspace.compete(capacity)
        self.memory.reinforce(self.prefrontal.working_memory.maintain(conscious))
        trace.log(
            "workspace",
            "broadcast",
            capacity=capacity,
            contents=[f"{s.kind}:{s.content[:60]}" for s in conscious],
        )

        # Arbitrate, think, speak.
        ctx.arbitration = self.acc.arbitrate(ctx)
        draft: Generation | None = None
        if ctx.arbitration.path == "slow":
            draft = await self.prefrontal.deliberate(ctx, self.cortex)
            self._account_llm(draft)
        reply, spoken = await self.broca.speak(ctx, self.cortex, draft)
        if spoken is not None:
            self._account_llm(spoken)

        # Learn: remember the exchange and what was done, for next turn's critic.
        self.hippocampus.encode(ctx, reply)
        cost = self.homeostasis.spent_usd - spent_before
        sources = [signal.meta for signal in conscious if signal.kind == "web" and signal.meta.get("url")]
        meta = {"path": ctx.arbitration.path, "context": ctx.context_label, "cost_usd": cost, "sources": sources}
        self.history += [Turn("user", message), Turn("assistant", reply, meta)]
        self.last_activity = now
        self._last_action = self.basal_ganglia.remember(ctx, ctx.arbitration.path)
        self._last_context = ctx.context_label
        trace.log("neuromodulators", "levels", **{k: round(v, 3) for k, v in self.modulators.levels.items()})
        trace.log("hypothalamus", "budget", spent_usd=round(self.homeostasis.spent_usd, 6))

        return Response(
            text=reply,
            action=ctx.action,
            path=ctx.arbitration.path,
            context=ctx.context_label,
            trace=trace,
            cost_usd=cost,
            modulators=dict(self.modulators.levels),
            sources=sources,
        )

    def _account_jev(self, decision: Decision, trace: CycleTrace, call: str, questions: int) -> None:
        cost = self.homeostasis.record("jev", decision.input_tokens)
        trace.log(
            "jev",
            call,
            questions=questions,
            input_tokens=decision.input_tokens,
            latency_ms=round(decision.latency_ms),
            cost_usd=round(cost, 7),
        )

    def _account_llm(self, generation: Generation) -> None:
        self.homeostasis.record(generation.model, generation.input_tokens, generation.output_tokens)


def create_brain(settings: Settings) -> Brain:
    """Build a brain on the real providers. Raises MissingJevKeyError without a TypeSafe key."""
    require_jev_key()
    search = TavilySearch(depth=settings.web.depth) if settings.web.enabled and search_key() else None
    return Brain(
        settings,
        jev=JevProvider(settings.models.jev),
        cortex=ClaudeProvider(settings.models),
        memory=MemoryStore(settings.memory_path),
        search=search,
    )
