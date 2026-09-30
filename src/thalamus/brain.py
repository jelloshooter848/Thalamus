"""The Brain: wires the regions together and runs one cognitive cycle per message.

sense → subcortical sweep (one batched JEV call) → learn from the last turn → recall → gate working
memory → global workspace competition → select action → arbitrate fast/slow → deliberate? → speak
→ encode.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

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
from thalamus.regions.acc import AnteriorCingulate
from thalamus.regions.amygdala import Amygdala
from thalamus.regions.basal_ganglia import BasalGanglia
from thalamus.regions.broca import Broca
from thalamus.regions.hippocampus import Hippocampus
from thalamus.regions.prefrontal import PrefrontalCortex
from thalamus.regions.sensory_cortex import SensoryCortex
from thalamus.regions.thalamus import Thalamus


@dataclass
class Response:
    text: str
    action: str
    path: str
    context: str
    trace: CycleTrace
    cost_usd: float
    modulators: dict[str, float]


class Brain:
    def __init__(
        self,
        settings: Settings,
        *,
        jev: DecisionProvider,
        cortex: LanguageProvider,
        memory: MemoryStore,
    ) -> None:
        self.settings = settings
        self.jev = jev
        self.cortex = cortex
        self.memory = memory
        self.session = uuid.uuid4().hex[:12]
        self.history: list[Turn] = []
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
        self.subcortical: list[BrainRegion] = [
            self.thalamus,
            self.amygdala,
            self.hippocampus,
            self.acc,
            self.basal_ganglia,
        ]

        self._turn = 0
        self._last_action: LastAction | None = None
        self._last_context: str | None = None

    async def think(self, message: str) -> Response:
        self._turn += 1
        trace = CycleTrace(turn=self._turn)
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

        # Recall and gate memories into working memory, then compete for awareness.
        candidates = self.hippocampus.recall(ctx)
        memory_signals, gate_decision = await self.prefrontal.working_memory.gate(ctx, candidates, self.jev)
        if gate_decision is not None:
            self._account_jev(gate_decision, trace, "wm_gate", len(gate_decision.nouls))
        for signal in memory_signals:
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

        # Select, arbitrate, think, speak.
        ctx.action = self.basal_ganglia.select(ctx)
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
        self.history += [Turn("user", message), Turn("assistant", reply)]
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
            cost_usd=self.homeostasis.spent_usd - spent_before,
            modulators=dict(self.modulators.levels),
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
    return Brain(
        settings,
        jev=JevProvider(settings.models.jev),
        cortex=ClaudeProvider(settings.models),
        memory=MemoryStore(settings.memory_path),
    )
