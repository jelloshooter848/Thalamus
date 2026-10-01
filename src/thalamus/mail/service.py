"""The mail service: checks accounts, triages new mail with JEV, stores results, notifies the phone.

Read-only by design: IMAP mailboxes are opened read-only and messages are peeked, so nothing is
ever marked read, moved or deleted. Email text is sent to TypeSafe (for JEV) and, for surfaced
emails only, to Anthropic; results are stored locally.
"""

from __future__ import annotations

import asyncio
import secrets as pysecrets
import time
import uuid
from collections.abc import Callable
from typing import Any

from thalamus.config import Settings
from thalamus.core.memory_store import MemoryStore
from thalamus.mail.accounts import PROVIDERS, Account, open_link
from thalamus.mail.imap import ImapSource, MailError
from thalamus.mail.notify import DEFAULT_SERVER, Ntfy
from thalamus.mail.outlook import OutlookAuth
from thalamus.mail.secrets import Secrets
from thalamus.mail.triage import triage
from thalamus.providers.base import Decision, DecisionProvider, Generation, LanguageProvider

CostSink = Callable[[list[Decision], list[Generation]], None]
NOTIFY_LEVELS = {"urgent": 3.6, "important": 2.9}
FIRST_RUN_NOTIFY_HOURS = 2  # on the first check, only alert about mail from the last couple of hours


class MailService:
    def __init__(
        self,
        memory: MemoryStore,
        settings: Settings,
        jev: DecisionProvider,
        cortex: LanguageProvider,
        secrets: Secrets,
        *,
        source: Any = None,
        outlook: OutlookAuth | None = None,
        notifier: Callable[[str, str], Ntfy] = Ntfy,
        on_costs: CostSink | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.memory, self.db, self.settings = memory, memory.db, settings
        self.jev, self.cortex, self.secrets = jev, cortex, secrets
        self.outlook = outlook or OutlookAuth(secrets)
        self.source = source or ImapSource(self._credential)
        self._notifier = notifier
        self._on_costs = on_costs or (lambda decisions, generations: None)
        self.clock = clock
        self.lock = asyncio.Lock()
        self.flows: dict[str, dict] = {}  # Microsoft sign-ins in progress

    # ----- accounts ---------------------------------------------------------------------------
    def accounts(self, enabled_only: bool = False) -> list[Account]:
        rows = self.db.execute("SELECT * FROM mail_accounts ORDER BY id").fetchall()
        accounts = [
            Account(
                id=r["id"], provider=r["provider"], address=r["address"], host=r["host"], port=r["port"],
                username=r["username"], auth=r["auth"], client_id=r["client_id"], folder=r["folder"],
                uidvalidity=r["uidvalidity"], last_uid=r["last_uid"], enabled=bool(r["enabled"]),
                last_check=r["last_check"], last_error=r["last_error"],
            )
            for r in rows
        ]
        return [a for a in accounts if a.enabled] if enabled_only else accounts

    def _credential(self, account: Account) -> str:
        if account.auth == "oauth":
            return self.outlook.token(account.client_id, account.secret_name, account.address)
        secret = self.secrets.get(account.secret_name)
        if not secret:
            raise MailError(f"{account.address}: no password saved. Add the account again in Settings.")
        return secret

    def draft_account(self, provider: str, address: str, host: str = "", port: int = 0, username: str = "",
                      client_id: str = "") -> Account:
        preset = PROVIDERS.get(provider, PROVIDERS["other"])
        return Account(
            id=0, provider=provider, address=address.strip().lower(), host=(host or preset["host"]).strip(),
            port=int(port or preset["port"]), username=(username or address).strip(), auth=preset["auth"],
            client_id=client_id.strip(),
        )

    async def add_password_account(self, draft: Account, password: str) -> Account:
        """Test the login first; only a working account is saved."""
        if not draft.host:
            raise MailError("Enter the IMAP server name (your provider's help pages list it).")
        probe = ImapSource(lambda _: password) if isinstance(self.source, ImapSource) else self.source
        await asyncio.to_thread(probe.test, draft)
        account = self._insert(draft)
        self.secrets.set(account.secret_name, password)
        return account

    def _insert(self, draft: Account) -> Account:
        cursor = self.db.execute(
            "INSERT INTO mail_accounts (provider, address, host, port, username, auth, client_id, created) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (draft.provider, draft.address, draft.host, draft.port, draft.username, draft.auth, draft.client_id,
             self.clock()),
        )
        self.db.commit()
        draft.id = int(cursor.lastrowid)
        return draft

    def remove_account(self, account_id: int) -> None:
        self.secrets.delete(f"mail:{account_id}")
        self.db.execute("DELETE FROM mail_accounts WHERE id = ?", (account_id,))
        self.db.execute("DELETE FROM mail_items WHERE account = ?", (account_id,))
        self.db.execute("DELETE FROM mail_contacts WHERE account = ?", (account_id,))
        self.db.commit()

    # ----- Microsoft sign-in --------------------------------------------------------------------
    async def outlook_start(self, address: str, client_id: str) -> dict:
        draft = self.draft_account("outlook", address, client_id=client_id)
        if "@" not in draft.address:
            raise MailError("Enter your Outlook email address first.")
        flow = await asyncio.to_thread(self.outlook.start, draft.client_id, draft.address)
        flow_id = uuid.uuid4().hex
        state = {"status": "pending", "user_code": flow["user_code"], "verification_uri": flow["verification_uri"],
                 "message": flow.get("message", "")}
        self.flows[flow_id] = state

        async def finish() -> None:
            cache_name = f"outlook:{flow_id}"
            try:
                signed_in = await asyncio.to_thread(self.outlook.finish, draft.client_id, flow, cache_name,
                                                      draft.address)
                draft.username = signed_in or draft.address
                draft.secret_key = cache_name
                await asyncio.to_thread(self.source.test, draft)
                draft.secret_key = ""
                account = self._insert(draft)
                self.secrets.set(account.secret_name, self.secrets.get(cache_name) or "")
                self.secrets.delete(cache_name)
                state.update(status="done", account=account.public())
            except Exception as error:  # noqa: BLE001 - shown to the user on the setup page
                state.update(status="error", error=str(error))

        state["task"] = asyncio.create_task(finish())
        return {"flow_id": flow_id, **{k: v for k, v in state.items() if k != "task"}}

    def outlook_status(self, flow_id: str) -> dict:
        state = self.flows.get(flow_id)
        if state is None:
            return {"status": "error", "error": "This sign-in has expired. Start again."}
        return {k: v for k, v in state.items() if k != "task"}

    # ----- checking ---------------------------------------------------------------------------
    def due(self) -> bool:
        accounts = self.accounts(enabled_only=True)
        if not accounts:
            return False
        last = min((a.last_check or 0) for a in accounts)
        return self.clock() - last >= self.settings.mail.check_minutes * 60

    def spent_today(self) -> float:
        midnight = time.mktime(time.localtime(self.clock())[:3] + (0, 0, 0, 0, 0, -1))
        return sum(self.memory.spending(since=midnight, conversation="mail").values())

    async def check_all(self) -> dict:
        summary = {"checked": 0, "new": 0, "surfaced": 0, "notified": 0, "errors": []}
        async with self.lock:
            for account in self.accounts(enabled_only=True):
                if self.spent_today() >= self.settings.mail.daily_budget_usd:
                    summary["errors"].append("Today's email budget is used up; checking resumes tomorrow.")
                    break
                try:
                    result = await self._check(account)
                    for key in ("new", "surfaced", "notified"):
                        summary[key] += result[key]
                    summary["checked"] += 1
                except Exception as error:  # noqa: BLE001 - one bad account must not stop the others
                    message = str(error)
                    if not isinstance(error, (MailError, RuntimeError)):
                        message = f"{account.address}: {error}"
                    self._update(account.id, last_check=self.clock(), last_error=message)
                    summary["errors"].append(message)
        return summary

    async def _check(self, account: Account) -> dict:
        contacts = self._contacts(account.id)
        if not contacts:
            contacts = await asyncio.to_thread(self.source.sent_contacts, account)
            self.db.executemany(
                "INSERT OR IGNORE INTO mail_contacts (account, address) VALUES (?, ?)",
                [(account.id, a) for a in contacts],
            )
        mine = {a.address for a in self.accounts()} | {account.username.lower()}
        first_run = account.last_uid == 0
        fetched = await asyncio.to_thread(self.source.fetch_new, account, my_addresses=mine, contacts=contacts)
        result = {"new": len(fetched.emails), "surfaced": 0, "notified": 0}
        if fetched.emails:
            about = [row["text"] for row in self.memory.core_facts(12)]
            levels = self.settings.mail
            outcome = await triage(
                fetched.emails, self.jev, self.cortex, about_user=about, surface_at=levels.surface_at,
                notify_at=NOTIFY_LEVELS.get(levels.notify_level, NOTIFY_LEVELS["important"]),
            )
            self._on_costs(outcome.decisions, outcome.generations)
            for verdict in outcome.verdicts:
                e = verdict.email
                recent = self.clock() - e.received <= FIRST_RUN_NOTIFY_HOURS * 3600
                notify = verdict.notify and (recent or not first_run)
                cursor = self.db.execute(
                    "INSERT OR IGNORE INTO mail_items (account, uid, message_id, received, from_name, from_addr, "
                    "subject, snippet, category, importance, needs_reply, scam, known_sender, reason, surfaced, "
                    "link, created) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (account.id, e.uid, e.message_id, e.received, e.from_name, e.from_addr, e.subject,
                     e.text[:400], verdict.category, verdict.importance, verdict.needs_reply, verdict.scam,
                     int(e.facts.get("sender_is_someone_you_have_emailed", False)), verdict.reason,
                     int(verdict.surfaced), open_link(account, e.message_id), self.clock()),
                )
                result["surfaced"] += int(verdict.surfaced)
                if notify and cursor.rowcount and await self._notify(cursor.lastrowid, verdict, account):
                    result["notified"] += 1
        self._update(account.id, last_check=self.clock(), last_error=None, last_uid=fetched.last_uid,
                     uidvalidity=fetched.uidvalidity)
        return result

    def _contacts(self, account_id: int) -> set[str]:
        rows = self.db.execute("SELECT address FROM mail_contacts WHERE account = ?", (account_id,))
        return {r["address"] for r in rows}

    def _update(self, account_id: int, **fields: Any) -> None:
        sets = ", ".join(f"{k} = ?" for k in fields)
        self.db.execute(f"UPDATE mail_accounts SET {sets} WHERE id = ?", (*fields.values(), account_id))
        self.db.commit()

    # ----- notifications ------------------------------------------------------------------------
    def notify_settings(self) -> dict:
        return {
            "enabled": self.memory.get_kv("notify.enabled") == "1",
            "topic": self.secrets.get("notify:topic") or "",
            "server": self.memory.get_kv("notify.server", DEFAULT_SERVER),
            "level": self.settings.mail.notify_level,
        }

    def configure_notify(self, enabled: bool, server: str = "", new_topic: bool = False) -> dict:
        if new_topic or not self.secrets.get("notify:topic"):
            self.secrets.set("notify:topic", "thalamus-" + pysecrets.token_urlsafe(12).replace("_", "x").lower())
        self.memory.set_kv("notify.enabled", "1" if enabled else "0")
        if server.strip():
            self.memory.set_kv("notify.server", server.strip())
        return self.notify_settings()

    async def send_test(self) -> None:
        config = self.notify_settings()
        await self._notifier(config["topic"], config["server"]).send(
            "THALAMUS", "Notifications are working. You'll hear from me when something important arrives.",
            tags="brain",
        )

    async def _notify(self, item_id: int, verdict, account: Account) -> bool:
        config = self.notify_settings()
        if not config["enabled"] or not config["topic"]:
            return False
        e = verdict.email
        urgent = verdict.importance >= NOTIFY_LEVELS["urgent"]
        try:
            await self._notifier(config["topic"], config["server"]).send(
                f"{'Urgent' if urgent else 'Important'}: {e.from_name}",
                f"{e.subject}\n{verdict.reason}".strip(),
                priority=5 if urgent else 4,
                click=open_link(account, e.message_id),
                tags="envelope",
            )
        except Exception:  # noqa: BLE001 - a failed alert must not lose the email
            return False
        self.db.execute("UPDATE mail_items SET notified = 1 WHERE id = ?", (item_id,))
        self.db.commit()
        return True

    # ----- reading results ------------------------------------------------------------------------
    def items(self, *, hours: float = 72, surfaced_only: bool = False, limit: int = 200) -> list[dict]:
        query = "SELECT mail_items.*, mail_accounts.address AS account_address FROM mail_items " \
                "JOIN mail_accounts ON mail_accounts.id = mail_items.account WHERE received >= ? AND dismissed = 0"
        if surfaced_only:
            query += " AND surfaced = 1"
        rows = self.db.execute(query + " ORDER BY surfaced DESC, importance DESC, received DESC LIMIT ?",
                               (self.clock() - hours * 3600, limit)).fetchall()
        return [dict(r) for r in rows]

    def dismiss(self, item_id: int) -> None:
        self.db.execute("UPDATE mail_items SET dismissed = 1 WHERE id = ?", (item_id,))
        self.db.commit()

    def chat_summary(self, hours: float = 48) -> str:
        """What the cortex is told when the user asks about their email."""
        accounts = self.accounts(enabled_only=True)
        if not accounts:
            return "No email accounts are connected yet; the user can add them in Settings → Email accounts."
        last = max((a.last_check or 0) for a in accounts)
        surfaced = self.items(hours=hours, surfaced_only=True, limit=15)
        counts = self.db.execute(
            "SELECT category, COUNT(*) AS n FROM mail_items WHERE received >= ? GROUP BY category ORDER BY n DESC",
            (self.clock() - hours * 3600,),
        ).fetchall()
        lines = [
            f"Accounts: {', '.join(a.address for a in accounts)}. Last checked "
            f"{_ago(self.clock() - last) if last else 'never'}.",
            "Last 48 hours by category: " + (", ".join(f"{r['category']} {r['n']}" for r in counts) or "nothing new"),
        ]
        if surfaced:
            lines.append("Emails worth the user's attention (most important first):")
            for item in surfaced:
                reply = " · expects a reply" if item["needs_reply"] >= 0.7 else ""
                lines.append(
                    f"- [{_ago(self.clock() - item['received'])}] {item['from_name']}: {item['subject']}"
                    f" ({item['category']}{reply}). {item['reason'] or ''}".rstrip()
                )
        else:
            lines.append("Nothing important in that time.")
        errors = [a.last_error for a in accounts if a.last_error]
        if errors:
            lines.append("Problems: " + "; ".join(errors))
        return "\n".join(lines)


def _ago(seconds: float) -> str:
    if seconds < 3600:
        return f"{max(1, round(seconds / 60))} min ago"
    if seconds < 86400:
        return f"{round(seconds / 3600)} h ago"
    return f"{round(seconds / 86400)} days ago"

