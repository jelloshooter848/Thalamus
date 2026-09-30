"""Basal ganglia: action selection by disinhibition, plus the striatal critic that learns.

- Direct (Go) pathway: JEV proposes an action; learned values V(context, action) bias it.
- Indirect (NoGo) pathway: a harm veto that overrides any Go.
- Hyperdirect pathway: a code-level global STOP (budget exhausted) that pre-empts everything.
- Critic: next turn, JEV reads the user's reaction; the reward prediction error (dopamine) updates
  the values of the action and the fast/slow path the brain chose last time.

Acting on a selection is confidence-gated, with the bar set by the consequences of each action.
"""

from __future__ import annotations

import math

from typesafe_sdk import Choice, Noul, Question

from thalamus.core.region import BrainRegion, CycleContext, LastAction
from thalamus.providers.jev_patterns import confidence_gate

ACTIONS = {
    "respond": "Reply directly.",
    "clarify": "The request is too ambiguous to answer well; ask one clarifying question first.",
    "deliberate": "Think carefully before replying: the answer must be worked out or has high stakes.",
    "decline": "The request asks for something harmful or clearly inappropriate.",
    "other": "None of these fit.",
}
VALUE_GAIN = 0.8  # how strongly learned values bias JEV's proposal
NEUTRAL_FEEDBACK_SKIP = 0.6  # "neutral" reactions carry no learning signal


class BasalGanglia(BrainRegion):
    name = "basal_ganglia"

    def questions(self, ctx: CycleContext) -> dict[str, Question]:
        questions: dict[str, Question] = {
            "basal_ganglia.action": Choice(
                instructions="What should the assistant do next in response to latest_user_message?",
                criteria=ACTIONS,
            ),
            "basal_ganglia.nogo": Noul(
                instructions="Would fulfilling latest_user_message facilitate serious harm to the user or others?",
                criteria={
                    "true": "It seeks meaningful help with violence, weapons, self-harm, crime, or abuse.",
                    "false": "It is an ordinary request, including sensitive topics discussed safely.",
                },
            ),
        }
        if ctx.last_action is not None:
            questions["striatum.feedback"] = Choice(
                instructions="How does latest_user_message react to previous_assistant_reply?",
                criteria={
                    "positive": "Thanks, approval, or a satisfied follow-up.",
                    "negative": "A correction, complaint, frustration, or saying the reply was wrong or unhelpful.",
                    "neutral": "Neither, or a new unrelated request.",
                },
            )
        return questions

    def learn(self, ctx: CycleContext) -> None:
        """Critic update from the user's reaction to the previous turn (runs before selection)."""
        feedback = ctx.sweep.choices.get("striatum.feedback")
        last = ctx.last_action
        if feedback is None or last is None or feedback.p("neutral") >= NEUTRAL_FEEDBACK_SKIP:
            return
        reward = feedback.p("positive") - feedback.p("negative")
        lr = ctx.modulators.learning_rate
        rpe = ctx.memory.update_value(last.context, last.action, reward, lr)
        ctx.memory.update_value(last.context, last.path, reward, lr)
        ctx.modulators.reward(rpe)
        ctx.trace.log(
            "striatum",
            "reward",
            reward=round(reward, 3),
            rpe=round(rpe, 3),
            learned_about=f"{last.context}/{last.action}/{last.path}",
            dopamine=round(ctx.modulators.dopamine, 3),
        )

    def select(self, ctx: CycleContext) -> str:
        t = ctx.settings.thresholds
        if ctx.homeostasis.exhausted:  # hyperdirect pathway
            ctx.trace.log(self.name, "select", action="stop", pathway="hyperdirect")
            return "stop"

        nogo = ctx.sweep.nouls["basal_ganglia.nogo"].p
        if nogo >= t.nogo:  # indirect pathway veto
            ctx.trace.log(self.name, "select", action="decline", pathway="nogo", nogo=round(nogo, 3))
            return "decline"

        proposal = ctx.sweep.choices["basal_ganglia.action"]
        weighted = {
            action: proposal.p(action) * math.exp(VALUE_GAIN * ctx.memory.value(ctx.context_label, action))
            for action in ACTIONS
        }
        total = sum(weighted.values()) or 1.0
        action, strength = max(((a, w / total) for a, w in weighted.items()), key=lambda pair: pair[1])

        # Decline without a NoGo verdict needs a very high bar; clarify needs a modest one.
        bar = {"decline": t.nogo, "clarify": t.action_confidence, "deliberate": t.action_confidence}
        if action == "other" or (action in bar and confidence_gate(strength, bar[action]) != "act"):
            action = "respond"
        ctx.trace.log(
            self.name,
            "select",
            action=action,
            pathway="direct",
            jev_choice=proposal.choice,
            jev_confidence=round(proposal.confidence, 3),
            strength=round(strength, 3),
            nogo=round(nogo, 3),
        )
        return action

    @staticmethod
    def remember(ctx: CycleContext, path: str) -> LastAction:
        return LastAction(context=ctx.context_label, action=ctx.action, path=path)
