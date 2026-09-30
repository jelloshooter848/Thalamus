"""Anterior cingulate cortex: metacognitive arbitration between System 1 and System 2.

Decides whether the slow prefrontal cortex (Claude, deep) is worth recruiting, using the Expected
Value of Control idea (Shenhav, Botvinick & Cohen): escalate when the value of deliberating beats
its cost. Three independent System-1 probes estimate demand; when two framings of the same
question disagree, that conflict itself triggers deliberation (De Neys). Answers that must be
*derived* (math, code, logic) always escalate: that's where fast judgment is known to fail.
The critic's learned values for fast vs slow in this context bias the decision (SOFAI-style).
"""

from __future__ import annotations

from typesafe_sdk import Choice, Noul, Question

from thalamus.core.region import Arbitration, BrainRegion, CycleContext
from thalamus.providers.jev_patterns import composite_score, disagreement

LEARNED_BIAS_LIMIT = 0.3


class AnteriorCingulate(BrainRegion):
    name = "acc"

    def questions(self, ctx: CycleContext) -> dict[str, Question]:
        return {
            "acc.derived": Noul(
                instructions=(
                    "Does answering latest_user_message correctly require working something out, such as "
                    "calculation, code, logic, or multi-step reasoning, rather than recalling or stating it?"
                ),
                criteria={
                    "true": "The answer has to be derived or computed.",
                    "false": "The answer can be read off, recalled, or is conversational.",
                },
            ),
            "acc.simple": Noul(
                instructions="Could a brief, direct reply fully and correctly satisfy latest_user_message?"
            ),
            "acc.effort": Choice(
                instructions="How much thinking does a high-quality reply to latest_user_message require?",
                criteria={
                    "trivial": "A greeting, acknowledgement, or one-line fact.",
                    "moderate": "A clear explanation or a short piece of advice.",
                    "hard": "Careful reasoning, calculation, code, planning, or weighing trade-offs.",
                },
            ),
        }

    def arbitrate(self, ctx: CycleContext) -> Arbitration:
        t = ctx.settings.thresholds
        derived = ctx.sweep.nouls["acc.derived"].p
        simple = ctx.sweep.nouls["acc.simple"].p
        effort = ctx.sweep.choices["acc.effort"]
        heavy = effort.p("hard") + 0.5 * effort.p("moderate")

        demand = composite_score(
            {"derived": derived, "heavy": heavy, "not_simple": 1 - simple},
            {"derived": 0.4, "heavy": 0.35, "not_simple": 0.25},
        )
        conflict = max(disagreement(effort.p("hard"), 1 - simple), disagreement(derived, 1 - simple))
        learned = ctx.memory.value(ctx.context_label, "slow") - ctx.memory.value(ctx.context_label, "fast")
        learned = max(-LEARNED_BIAS_LIMIT, min(LEARNED_BIAS_LIMIT, learned))
        value = demand * (0.6 + 0.8 * ctx.appraisal.stakes) + learned
        # Serotonin is patience: a long time horizon makes deliberation feel cheaper.
        cost = ctx.homeostasis.deliberation_cost * (1.3 - 0.6 * ctx.modulators.serotonin)

        reasons: list[str] = []
        if ctx.homeostasis.exhausted:
            reasons.append("budget exhausted")
        elif ctx.action == "deliberate":
            reasons.append("basal ganglia selected deliberation")
        elif ctx.action in ("clarify", "decline", "stop"):
            pass
        else:
            if derived >= t.derived:
                reasons.append(f"answer must be derived (p={derived:.2f})")
            if conflict >= t.conflict:
                reasons.append(f"System-1 probes disagree (conflict={conflict:.2f})")
            if value - cost >= t.escalate_margin:
                reasons.append(f"value of control {value:.2f} exceeds cost {cost:.2f}")

        slow = bool(reasons) and not ctx.homeostasis.exhausted
        arbitration = Arbitration(
            path="slow" if slow else "fast",
            reasons=reasons or ["System 1 is sufficient"],
            demand=demand,
            value=value,
            cost=cost,
            conflict=conflict,
        )
        ctx.trace.log(
            self.name,
            "arbitrate",
            path=arbitration.path,
            reasons=arbitration.reasons,
            demand=round(demand, 3),
            value=round(value, 3),
            cost=round(cost, 3),
            conflict=round(conflict, 3),
            learned_bias=round(learned, 3),
        )
        return arbitration
