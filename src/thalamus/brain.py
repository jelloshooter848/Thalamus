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
from thalamus.core.region import BrainRegion, CycleContext, LastAction, StatusSink, Turn
from thalamus.core.trace import CycleTrace
from thalamus.core.workspace import GlobalWorkspace
from thalamus.mail.service import MailService
from thalamus.providers.base import Decision, DecisionProvider, Generation, LanguageProvider, TextSink
from thalamus.providers.claude import ClaudeProvider
from thalamus.providers.jev import JevProvider
from thalamus.providers.search import SearchProvider, TavilySearch, search_key
from thalamus.regions.acc import AnteriorCingulate
from thalamus.regions.amygdala import Amygdala
from thalamus.regions.basal_ganglia import BasalGanglia
from thalamus.regions.broca import Broca
from thalamus.regions.hippocampus import Hippocampus
from thalamus.regions.mail_sense import MailSense
from thalamus.regions.prefrontal import PrefrontalCortex, buffer_window
from thalamus.regions.self_model import SelfModel
from thalamus.regions.semantic import Reconsolidation, SemanticMemory
from thalamus.regions.sensory_cortex import SensoryCortex
from thalamus.regions.sleep import consolidate, pending_episodes
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
    cost_breakdown: dict[str, float]


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
        mail: MailService | None = None,
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
        self.semantic = SemanticMemory()
        self.reconsolidation = Reconsolidation()
        self.mail = mail
        if mail is not None:
            mail._on_costs = lambda decisions, generations: self.record_background_costs("mail", decisions, generations)
        self.mail_sense = MailSense(mail)
        self.subcortical: list[BrainRegion] = [
            self.thalamus,
            self.amygdala,
            self.hippocampus,
            self.acc,
            self.basal_ganglia,
            self.web,
            self.self_model,
            self.reconsolidation,
            self.mail_sense,
        ]

        self._turn = 0
        self._last_action: LastAction | None = None
        self._last_context: str | None = None
        self._resume_or_start()

    def _resume_or_start(self) -> None:
        """Pick the last conversation back up after a restart, unless it has gone quiet."""
        row = self.memory.latest_conversation()
        idle = self.settings.conversation_idle_minutes * 60
        if row is None or self.clock() - row["last_activity"] > idle:
            self.new_conversation()
            return
        self.session = row["id"]
        self.conversation_started = row["started"]
        self.last_activity = row["last_activity"]
        self.history = [Turn(role, content, meta) for role, content, meta in self.memory.load_turns(row["id"])]
        self._turn = len(self.history) // 2

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
        self.memory.start_conversation(self.session, self.conversation_started)

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
            "budget": {
                "spent_usd": self.homeostasis.spent_usd,
                "limit_usd": self.settings.budget.session_usd,
                "by_service": dict(self.homeostasis.by_service),
            },
        }

    async def think(
        self,
        message: str,
        on_text: TextSink | None = None,
        on_status: StatusSink | None = None,
    ) -> Response:
        """Run one cognitive cycle. `on_text` receives the reply as it streams; `on_status` receives
        short progress notes ("Searching the web…") for the UI."""
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
        by_service_before = dict(self.homeostasis.by_service)
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
            mail_accounts=[a.address for a in self.mail.accounts(enabled_only=True)] if self.mail else [],
            on_text=on_text,
            on_status=on_status,
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

        # Reconsolidation first, so anything the user asked to forget can't be recalled below.
        edit = await self.reconsolidation.edit(ctx, self.jev, self.cortex)
        for decision in edit.decisions:
            self._account_jev(decision, trace, "memory_edit", len(decision.nouls))
        for generation in edit.generations:
            self._account_llm(generation)
        self.semantic.recall(ctx)
        await self.mail_sense.sense(ctx)

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
            await ctx.status("Thinking it through…")
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
        self.memory.add_turns(self.session, [("user", message, None), ("assistant", reply, meta)], now)
        self.last_activity = now
        self._last_action = self.basal_ganglia.remember(ctx, ctx.arbitration.path)
        self._last_context = ctx.context_label
        trace.log("neuromodulators", "levels", **{k: round(v, 3) for k, v in self.modulators.levels.items()})
        breakdown = {
            service: total - by_service_before.get(service, 0.0)
            for service, total in self.homeostasis.by_service.items()
            if total - by_service_before.get(service, 0.0) > 0
        }
        self.memory.record_costs(self.session, breakdown, now)
        trace.log(
            "hypothalamus",
            "budget",
            spent_usd=round(self.homeostasis.spent_usd, 6),
            this_turn={service: round(usd, 6) for service, usd in breakdown.items()},
            **({"unpriced_models": sorted(self.homeostasis.unpriced)} if self.homeostasis.unpriced else {}),
        )

        return Response(
            text=reply,
            action=ctx.action,
            path=ctx.arbitration.path,
            context=ctx.context_label,
            trace=trace,
            cost_usd=cost,
            modulators=dict(self.modulators.levels),
            sources=sources,
            cost_breakdown=breakdown,
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

    def sleep_due(self) -> bool:
        """Idle long enough, with new things said since the last sleep."""
        if self.last_activity is None:
            return pending_episodes(self.memory) >= self.settings.memory.sleep_min_new
        idle = self.clock() - self.last_activity
        return (
            idle >= self.settings.memory.sleep_after_idle_minutes * 60
            and pending_episodes(self.memory) >= self.settings.memory.sleep_min_new
        )

    async def sleep(self, trigger: str = "manual", force: bool = False) -> dict:
        """Consolidate recent episodes into long-term facts (see regions/sleep.py)."""
        before = dict(self.homeostasis.by_service)
        result = await consolidate(self.memory, self.settings, self.cortex, self.jev, trigger=trigger, force=force)
        trace = CycleTrace(turn=0)
        for generation in result.generations:
            self._account_llm(generation)
        for decision in result.decisions:
            self._account_jev(decision, trace, "sleep_verify", len(decision.nouls))
        spent = {
            k: v - before.get(k, 0.0) for k, v in self.homeostasis.by_service.items() if v - before.get(k, 0.0) > 0
        }
        self.memory.record_costs("sleep", spent, self.clock())
        report = {**result.report, "cost_usd": round(sum(spent.values()), 6)}
        if result.generations:
            self.memory.log_sleep(report, self.clock())
        return report

    def record_background_costs(self, kind: str, decisions: list[Decision], generations: list[Generation]) -> None:
        """Work done outside a conversation (email triage) goes to the ledger with its own daily budget,
        not to the chat session's budget."""
        body = self.homeostasis
        spent = {
            "jev": sum(body.cost_of("jev", d.input_tokens) for d in decisions),
            "claude": sum(body.cost_of(g.model, g.input_tokens, g.output_tokens) for g in generations),
        }
        self.memory.record_costs(kind, spent, self.clock())

    def _account_llm(self, generation: Generation) -> None:
        self.homeostasis.record(generation.model, generation.input_tokens, generation.output_tokens)


def create_brain(settings: Settings) -> Brain:
    """Build a brain on the real providers. Raises MissingJevKeyError without a TypeSafe key."""
    require_jev_key()
    from thalamus.mail.secrets import default_secrets

    search = TavilySearch(depth=settings.web.depth) if settings.web.enabled and search_key() else None
    jev, cortex = JevProvider(settings.models.jev), ClaudeProvider(settings.models)
    memory = MemoryStore(settings.memory_path)
    mail = MailService(memory, settings, jev, cortex, default_secrets(settings.memory_path.parent))
    return Brain(settings, jev=jev, cortex=cortex, memory=memory, search=search, mail=mail)
