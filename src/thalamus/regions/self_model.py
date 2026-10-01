"""Medial prefrontal cortex: self-referential processing.

Language models carry strong trained-in beliefs about themselves ("I can't browse the internet",
"I have no memory", "my knowledge stops at my training cutoff"). A capability sentence in a long
system prompt is often not enough to override them, especially on the fast path. So when JEV
judges that a message is about THALAMUS itself, this region computes the true facts from the
running system and puts them into awareness as the most salient signal of the cycle.
"""

from __future__ import annotations

import time

from typesafe_sdk import Noul, Question

from thalamus.core.region import BrainRegion, CycleContext
from thalamus.core.signals import Signal

ABOUT_ME_THRESHOLD = 0.5


def _when(timestamp: float) -> str:
    return time.strftime("%d %B %Y", time.localtime(timestamp))


class SelfModel(BrainRegion):
    name = "self_model"

    def questions(self, ctx: CycleContext) -> dict[str, Question]:
        return {
            "self.about_me": Noul(
                instructions=(
                    "Is latest_user_message asking about the assistant itself: what it can do, whether "
                    "it can search the internet, what it remembers, what it knows, or how it works?"
                )
            )
        }

    def absorb(self, ctx: CycleContext) -> None:
        p = ctx.sweep.nouls["self.about_me"].p
        if p < ABOUT_ME_THRESHOLD:
            return
        facts = self.facts(ctx)
        ctx.workspace.submit(
            Signal(source=self.name, kind="self", key="self", content="\n- ".join(facts), salience=0.99)
        )
        ctx.trace.log(self.name, "introspect", p=round(p, 3), facts=facts)

    @staticmethod
    def facts(ctx: CycleContext) -> list[str]:
        """True, current facts about this mind, computed from the running system."""
        models = ctx.settings.models
        episodes, first = ctx.memory.stats()
        known = len(ctx.memory.facts())
        facts = []
        if ctx.web_enabled:
            facts.append(
                "Internet: YES. You can search the web right now (via Tavily). You do it automatically "
                "whenever a message needs current or outside information, and you read the pages behind "
                "links the user pastes (Tavily Extract, retrying with an advanced reader). Some sites "
                "block automated readers; when a read fails you're told why. If the user asks you to look "
                "something up, you will."
            )
            facts.append(
                "Knowledge: your built-in knowledge is from training, but that is not a limit here, "
                "because you can look up anything current. Don't describe a training cutoff as a "
                "limitation."
            )
        else:
            facts.append(
                "Internet: NO. Web search isn't set up yet. The user can enable it by adding a free "
                "Tavily API key in Settings."
            )
        if episodes:
            facts.append(
                f"Long-term memory: YES. {episodes} remembered messages stored on this computer, the "
                f"earliest from {_when(first)}, plus {known} distilled facts about the user (consolidated "
                "while you sleep; the user can view, edit or delete them in the Memory window). Recall is "
                "selective, so you may not remember everything."
            )
        else:
            facts.append("Long-term memory: YES, but it's empty so far. Nothing has been stored yet.")
        mail = getattr(ctx, "mail_accounts", None)
        if mail:
            facts.append(
                f"Email: YES, read-only. You can read (never send, move or delete) {len(mail)} account(s): "
                f"{', '.join(mail)}. You check them every {ctx.settings.mail.check_minutes:.0f} minutes, flag "
                "what matters, and can alert the user's phone. Ask-to-check works any time."
            )
        else:
            facts.append(
                "Email: not connected yet. The user can add Gmail, iCloud, Outlook or other accounts in "
                "Settings → Email accounts, read-only."
            )
        facts.append(
            f"This conversation: {ctx.turn} message(s) so far. Earlier conversations are reachable "
            "only through memory."
        )
        facts.append(
            f"Thinking: fast judgments by JEV (TypeSafe), quick replies by {models.fast}, careful "
            f"reasoning by {models.deep} when a message needs it."
        )
        facts.append("The user can watch all of this in the Brain activity and Mind tabs.")
        return facts
