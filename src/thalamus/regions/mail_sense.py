"""Email as a sense in conversation: when the user asks about their mail, THALAMUS looks.

A JEV question in the batched sweep notices "anything important in my email?". If the last check
is more than a few minutes old, new mail is fetched and triaged first. Then the triage summary
(what's worth attention, and why) enters awareness, so the cortex answers from real data.
"""

from __future__ import annotations

from typesafe_sdk import Noul, Question

from thalamus.core.region import BrainRegion, CycleContext
from thalamus.core.signals import Signal

ASKED_THRESHOLD = 0.6
FRESH_SECONDS = 300


class MailSense(BrainRegion):
    name = "mail"

    def __init__(self, service) -> None:
        self.service = service

    @property
    def enabled(self) -> bool:
        return self.service is not None and bool(self.service.accounts(enabled_only=True))

    def questions(self, ctx: CycleContext) -> dict[str, Question]:
        if not self.enabled:
            return {}
        return {
            "mail.asked": Noul(
                instructions="Is latest_user_message asking about the user's email, inbox, or messages they "
                "received?"
            )
        }

    async def sense(self, ctx: CycleContext) -> None:
        if not self.enabled:
            return
        asked = ctx.sweep.nouls.get("mail.asked")
        if asked is None or asked.p < ASKED_THRESHOLD:
            return
        accounts = self.service.accounts(enabled_only=True)
        stale = min((a.last_check or 0) for a in accounts) < ctx.now - FRESH_SECONDS
        summary = None
        if stale:
            await ctx.status("Checking your email…")
            summary = await self.service.check_all()
        ctx.workspace.submit(
            Signal(source=self.name, kind="mail", key="mail", content=self.service.chat_summary(), salience=0.95)
        )
        ctx.trace.log(self.name, "inbox", checked_now=stale, **({"result": summary} if summary else {}))
