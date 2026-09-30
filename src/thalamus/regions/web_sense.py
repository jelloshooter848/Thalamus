"""The web as a sense: THALAMUS looks things up only when JEV judges it needs to.

1. Orienting: in the batched System-1 sweep, JEV judges whether the message needs current or
   outside information and what kind (general, news or finance).
2. Seeking: if it does, the fast cortex turns the conversation into one search query. A URL
   pasted by the user is read directly instead.
3. Sensory gating: like the thalamic relay, JEV judges each result's relevance. Only results that
   pass reach the global workspace, and from there the cortex. Rejected ones never do.

Web content is untrusted: it is always presented as data, never as instructions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from typesafe_sdk import Choice, Noul, Question

from thalamus.core.region import BrainRegion, CycleContext
from thalamus.core.signals import Signal
from thalamus.health import explain
from thalamus.providers.base import Decision, DecisionProvider, Generation, LanguageProvider
from thalamus.providers.jev import clip
from thalamus.providers.search import SearchOutcome, SearchProvider
from thalamus.regions.prefrontal import conversation_messages
from thalamus.regions.sensory_cortex import UNTRUSTED_NOTE

URL_RE = re.compile(r"https?://[^\s<>\"')\]]+")
TOPICS = {
    "general": "Facts, how-tos, products, places, people, or anything not covered below.",
    "news": "Recent events, current affairs, sports results, announcements, or what happened lately.",
    "finance": "Stock or crypto prices, markets, company financials, or exchange rates.",
}
QUERY_WRITER = (
    "You write web search queries. Given the conversation, reply with one concise search query "
    "(at most 12 words) that would find the information needed to answer the user's latest "
    "message. Include names, places and dates that matter. Reply with the query only."
)


@dataclass
class WebOutcome:
    signals: list[Signal] = field(default_factory=list)
    query_generation: Generation | None = None
    gate_decision: Decision | None = None
    billable_calls: int = 0


class WebSense(BrainRegion):
    name = "web"

    def __init__(self, provider: SearchProvider | None) -> None:
        self.provider = provider
        self.last: dict | None = None  # this turn's web activity, for the Mind view and replies

    @property
    def enabled(self) -> bool:
        return self.provider is not None

    def questions(self, ctx: CycleContext) -> dict[str, Question]:
        if not self.enabled:
            return {}
        return {
            "web.needed": Noul(
                instructions=(
                    "Does replying well to latest_user_message require looking something up on the "
                    "internet: current events, news, weather, prices, schedules, recent releases, or "
                    "specific facts an assistant may not reliably know?"
                ),
                criteria={
                    "true": (
                        "It needs up-to-date or outside information, or the user asks to search or look "
                        "something up."
                    ),
                    "false": (
                        "Conversation, opinion, advice, creative work, reasoning, or well-established "
                        "general knowledge."
                    ),
                },
            ),
            "web.topic": Choice(
                instructions="If latest_user_message were searched on the web, what kind of search would it be?",
                criteria=TOPICS,
            ),
        }

    async def sense(self, ctx: CycleContext, cortex: LanguageProvider, jev: DecisionProvider) -> WebOutcome:
        outcome = WebOutcome()
        if not self.enabled:
            self.last = None
            return outcome
        need = ctx.sweep.nouls["web.needed"].p
        url = next(iter(URL_RE.findall(ctx.message)), None)
        record: dict = {"needed": round(need, 3), "query": None, "results": [], "latency_ms": 0}
        self.last = record

        if ctx.action in ("decline", "stop") or ctx.homeostasis.exhausted:
            record["decision"] = "skipped"
            record["reason"] = f"action is {ctx.action}" if ctx.action in ("decline", "stop") else "budget exhausted"
        elif url:
            record["decision"] = "read"
            record["reason"] = "you shared a link"
        elif need >= ctx.settings.web.threshold:
            record["decision"] = "searched"
            record["reason"] = f"JEV judged outside information is needed (p={need:.2f})"
        else:
            record["decision"] = "skipped"
            record["reason"] = f"not needed (p={need:.2f})"
        if record["decision"] == "skipped":
            ctx.trace.log(self.name, "orient", **{k: record[k] for k in ("needed", "decision", "reason")})
            return outcome

        try:
            if url:
                found = await self.provider.read(url)
            else:
                outcome.query_generation = await cortex.generate(
                    tier="fast", system=QUERY_WRITER, messages=conversation_messages(ctx), max_tokens=60
                )
                query = outcome.query_generation.text.strip().strip('"').splitlines()[0][:200] or ctx.message
                topic = ctx.sweep.choices["web.topic"].choice
                found = await self.provider.search(query, topic=topic, max_results=ctx.settings.web.max_results)
            outcome.billable_calls = 1
        except Exception as error:  # noqa: BLE001 - a failed search must not break the conversation
            problem = explain(error)
            record.update(decision="failed", reason=problem.message)
            ctx.trace.log(self.name, "orient", needed=record["needed"], decision="failed", reason=problem.message)
            return outcome

        record["query"] = found.query
        record["latency_ms"] = round(found.latency_ms)
        outcome.signals, outcome.gate_decision = await self._gate(ctx, found, jev, trusted=bool(url))
        ctx.trace.log(
            self.name,
            "orient",
            needed=record["needed"],
            decision=record["decision"],
            reason=record["reason"],
            query=found.query,
            latency_ms=record["latency_ms"],
            results=[{"title": r["title"], "p": r["p"], "admitted": r["admitted"]} for r in record["results"]],
        )
        return outcome

    async def _gate(
        self, ctx: CycleContext, found: SearchOutcome, jev: DecisionProvider, *, trusted: bool
    ) -> tuple[list[Signal], Decision | None]:
        results = [r for r in found.results if r.content.strip()]
        if not results:
            return [], None
        ids = {f"r{i}": result for i, result in enumerate(results, start=1)}
        decision = None
        if trusted:  # the page the user asked for is let in without a relevance vote
            relevance = {rid: 1.0 for rid in ids}
        else:
            decision = await jev.decide(
                {
                    "note": UNTRUSTED_NOTE,
                    "latest_user_message": clip(ctx.message),
                    "search_query": found.query,
                    "results": {
                        rid: {"title": r.title, "url": r.url, "snippet": clip(r.content, 700)} for rid, r in ids.items()
                    },
                },
                {
                    f"web.{rid}": Noul(
                        instructions=(
                            f"Does result {rid} in results contain information that helps answer "
                            "latest_user_message?"
                        )
                    )
                    for rid in ids
                },
            )
            relevance = {rid: decision.nouls[f"web.{rid}"].p for rid in ids}

        threshold = ctx.settings.thresholds.memory_gate
        admitted = sorted((rid for rid in ids if relevance[rid] >= threshold), key=relevance.get, reverse=True)
        admitted = admitted[: ctx.settings.web.admit]
        limit = 6000 if trusted else 1200
        signals = []
        for rid, result in ids.items():
            ok = rid in admitted
            self.last["results"].append(
                {"title": result.title, "url": result.url, "p": round(relevance[rid], 3), "admitted": ok}
            )
            if ok:
                signals.append(
                    Signal(
                        source=self.name,
                        kind="web",
                        key=f"web:{result.url}",
                        content=f"{result.title} ({result.url}): {clip(result.content, limit)}",
                        salience=0.55 + 0.4 * relevance[rid],
                        confidence=relevance[rid],
                        meta={"title": result.title, "url": result.url},
                    )
                )
        return signals, decision
