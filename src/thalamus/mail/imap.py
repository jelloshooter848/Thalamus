"""Read-only IMAP access. Mailboxes are opened with EXAMINE (read-only) and messages are fetched
with BODY.PEEK, so THALAMUS never marks anything read, moves or deletes anything."""

from __future__ import annotations

import imaplib
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from email.utils import getaddresses

from thalamus.mail.accounts import Account
from thalamus.mail.parse import Email, parse_message

FETCH_BYTES = 60_000  # enough for headers and the readable part of almost any email
FIRST_RUN_DAYS = 2
MAX_PER_CHECK = 200


@dataclass
class FetchResult:
    emails: list[Email] = field(default_factory=list)
    last_uid: int = 0
    uidvalidity: int = 0


class MailError(RuntimeError):
    """A plain-language email problem (bad password, server unreachable...)."""


def explain_imap(error: Exception, account: Account) -> MailError:
    text = str(error)
    if "AUTHENTICATIONFAILED" in text.upper() or "invalid credentials" in text.lower() or "LOGIN failed" in text:
        hint = {
            "gmail": "Use a Google app password (Google Account → Security → 2-Step Verification → App passwords), "
            "not your normal password.",
            "icloud": "Use an app-specific password from appleid.apple.com.",
            "yahoo": "Use an app password from your Yahoo account security settings.",
            "outlook": "Sign in with Microsoft again.",
        }.get(account.provider, "Check the username and password (an app password may be required).")
        return MailError(f"{account.address}: the mail server rejected the login. {hint}")
    if isinstance(error, (OSError, TimeoutError)):
        return MailError(
            f"{account.address}: couldn't reach {account.host}. Check the server name and your connection."
        )
    return MailError(f"{account.address}: {text[:200]}")


class ImapSource:
    """Opens a fresh read-only connection per check (called from a worker thread)."""

    def __init__(self, credential: Callable[[Account], str]) -> None:
        self._credential = credential  # password, or an OAuth access token for "oauth" accounts

    def _connect(self, account: Account) -> imaplib.IMAP4_SSL:
        try:
            conn = imaplib.IMAP4_SSL(account.host, account.port, timeout=30)
            secret = self._credential(account)
            if account.auth == "oauth":
                auth = f"user={account.username}\x01auth=Bearer {secret}\x01\x01".encode()
                conn.authenticate("XOAUTH2", lambda _: auth)
            else:
                conn.login(account.username, secret)
            return conn
        except (imaplib.IMAP4.error, OSError) as error:
            raise explain_imap(error, account) from error

    def test(self, account: Account) -> str:
        conn = self._connect(account)
        try:
            status, data = conn.select(account.folder, readonly=True)
            count = int(data[0]) if status == "OK" and data and data[0] else 0
            return f"connected; {count} messages in {account.folder}"
        finally:
            conn.logout()

    def fetch_new(self, account: Account, *, my_addresses: set[str], contacts: set[str]) -> FetchResult:
        conn = self._connect(account)
        try:
            status, _ = conn.select(account.folder, readonly=True)
            if status != "OK":
                raise MailError(f"{account.address}: couldn't open {account.folder}.")
            uidvalidity = int((conn.response("UIDVALIDITY")[1] or [b"0"])[0] or 0)
            fresh = uidvalidity != account.uidvalidity or account.last_uid == 0
            if fresh:  # first run (or the server renumbered): only look back a couple of days
                since = time.strftime("%d-%b-%Y", time.gmtime(time.time() - FIRST_RUN_DAYS * 86400))
                status, data = conn.uid("SEARCH", None, "SINCE", since)
            else:
                status, data = conn.uid("SEARCH", None, "UID", f"{account.last_uid + 1}:*")
            uids = [int(u) for u in (data[0] or b"").split()] if status == "OK" else []
            uids = [u for u in uids if fresh or u > account.last_uid][-MAX_PER_CHECK:]
            result = FetchResult(last_uid=max([account.last_uid if not fresh else 0, *uids]), uidvalidity=uidvalidity)
            for uid in uids:
                status, parts = conn.uid("FETCH", str(uid), f"(BODY.PEEK[]<0.{FETCH_BYTES}>)")
                raw = next((p[1] for p in parts or [] if isinstance(p, tuple)), None)
                if status == "OK" and raw:
                    result.emails.append(parse_message(raw, uid, my_addresses=my_addresses, contacts=contacts))
            return result
        except imaplib.IMAP4.error as error:
            raise explain_imap(error, account) from error
        finally:
            try:
                conn.logout()
            except (imaplib.IMAP4.error, OSError):
                pass

    def sent_contacts(self, account: Account, limit: int = 500) -> set[str]:
        """Addresses you've written to, from the Sent folder's headers: the people who matter."""
        conn = self._connect(account)
        try:
            status, folders = conn.list()
            sent = next(
                (re.findall(r'"([^"]+)"$|(\S+)$', f.decode())[0] for f in folders or [] if b"\\Sent" in f), None
            )
            name = (sent[0] or sent[1]) if sent else None
            if not name or conn.select(f'"{name}"', readonly=True)[0] != "OK":
                return set()
            status, data = conn.uid("SEARCH", None, "ALL")
            uids = (data[0] or b"").split()[-limit:] if status == "OK" else []
            if not uids:
                return set()
            status, parts = conn.uid("FETCH", b",".join(uids).decode(), "(BODY.PEEK[HEADER.FIELDS (TO CC)])")
            headers = " ".join(p[1].decode(errors="replace") for p in parts or [] if isinstance(p, tuple))
            fields = re.findall(r"(?im)^(?:to|cc):(.*(?:\n[ \t].*)*)", headers)
            return {addr.lower() for _, addr in getaddresses(fields) if addr}
        except (imaplib.IMAP4.error, OSError):
            return set()
        finally:
            try:
                conn.logout()
            except (imaplib.IMAP4.error, OSError):
                pass
