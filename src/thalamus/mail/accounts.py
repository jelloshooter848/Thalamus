"""Email accounts: provider presets and the account record (secrets live elsewhere)."""

from __future__ import annotations

from dataclasses import dataclass

PROVIDERS = {
    "gmail": {"label": "Gmail", "host": "imap.gmail.com", "port": 993, "auth": "password"},
    "icloud": {"label": "iCloud", "host": "imap.mail.me.com", "port": 993, "auth": "password"},
    "yahoo": {"label": "Yahoo", "host": "imap.mail.yahoo.com", "port": 993, "auth": "password"},
    "outlook": {"label": "Outlook", "host": "outlook.office365.com", "port": 993, "auth": "oauth"},
    "other": {"label": "Email", "host": "", "port": 993, "auth": "password"},
}


@dataclass
class Account:
    id: int
    provider: str
    address: str
    host: str
    port: int
    username: str
    auth: str  # "password" or "oauth"
    client_id: str = ""
    folder: str = "INBOX"
    uidvalidity: int = 0
    last_uid: int = 0
    enabled: bool = True
    last_check: float | None = None
    last_error: str | None = None
    secret_key: str = ""  # set only while a new sign-in is being tested, before the account is saved

    @property
    def secret_name(self) -> str:
        return self.secret_key or f"mail:{self.id}"

    @property
    def label(self) -> str:
        return f"{PROVIDERS.get(self.provider, PROVIDERS['other'])['label']} · {self.address}"

    def public(self) -> dict:
        """Safe to show in the UI: no secrets."""
        return {
            "id": self.id,
            "provider": self.provider,
            "address": self.address,
            "label": self.label,
            "host": self.host,
            "enabled": self.enabled,
            "last_check": self.last_check,
            "last_error": self.last_error,
        }


def open_link(account: Account, message_id: str) -> str | None:
    """Where the user can open the email themselves."""
    if account.provider == "gmail" and message_id:
        return f"https://mail.google.com/mail/u/{account.address}/#search/rfc822msgid:{message_id.strip('<>')}"
    if account.provider == "outlook":
        return "https://outlook.live.com/mail/0/inbox"
    if account.provider == "icloud":
        return "https://www.icloud.com/mail"
    if account.provider == "yahoo":
        return "https://mail.yahoo.com"
    return None
