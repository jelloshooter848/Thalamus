"""Turning raw email into the compact, structured view JEV judges.

Code does what code can: decode headers and MIME, pick the readable text, strip quoted replies and
signatures' noise, and pre-compute facts (known sender, addressed directly to you, reply, list mail,
age), because JEV reads text literally and shouldn't have to infer any of that.
"""

from __future__ import annotations

import email
import email.policy
import html
import re
import time
from dataclasses import dataclass, field
from email.utils import getaddresses, parseaddr, parsedate_to_datetime

SNIPPET_CHARS = 1500
QUOTE_START = re.compile(r"^\s*(On .+wrote:|-----Original Message-----|From: .+|Sent from my \w+)", re.I | re.M)


@dataclass
class Email:
    uid: int
    message_id: str
    received: float
    from_name: str
    from_addr: str
    to: list[str]
    cc: list[str]
    subject: str
    text: str
    is_reply: bool
    list_mail: bool
    facts: dict = field(default_factory=dict)


def _html_to_text(markup: str) -> str:
    markup = re.sub(r"(?is)<(script|style|head).*?</\1>", " ", markup)
    markup = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</tr>|</li>", "\n", markup)
    return html.unescape(re.sub(r"<[^>]+>", " ", markup))


def readable_text(message: email.message.EmailMessage) -> str:
    part = message.get_body(preferencelist=("plain", "html")) if message.is_multipart() else message
    if part is None:
        return ""
    try:
        content = part.get_content()
    except (LookupError, UnicodeError, AttributeError, KeyError):
        payload = part.get_payload(decode=True) or b""
        content = payload.decode("utf-8", errors="replace")
    if not isinstance(content, str):
        return ""
    if part.get_content_type() == "text/html":
        content = _html_to_text(content)
    return content


def strip_quoted(text: str) -> str:
    lines = [line for line in text.splitlines() if not line.lstrip().startswith(">")]
    text = "\n".join(lines)
    match = QUOTE_START.search(text)
    if match and text[: match.start()].strip():  # keep the new part, drop the quoted history below it
        text = text[: match.start()]
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t ]+", " ", text)).strip()


def parse_message(
    raw: bytes, uid: int, *, my_addresses: set[str], contacts: set[str], now: float | None = None
) -> Email:
    message = email.message_from_bytes(raw, policy=email.policy.default)
    from_name, from_addr = parseaddr(str(message.get("From", "")))
    to = [a.lower() for _, a in getaddresses([str(v) for v in message.get_all("To", [])]) if a]
    cc = [a.lower() for _, a in getaddresses([str(v) for v in message.get_all("Cc", [])]) if a]
    try:
        received = parsedate_to_datetime(str(message.get("Date"))).timestamp()
    except (TypeError, ValueError):
        received = now or time.time()
    subject = str(message.get("Subject", "")).strip()
    is_reply = bool(message.get("In-Reply-To")) or subject.lower().startswith(("re:", "aw:"))
    list_mail = bool(message.get("List-Unsubscribe") or message.get("List-Id"))
    text = strip_quoted(readable_text(message))[:SNIPPET_CHARS]
    sender = from_addr.lower()
    email_ = Email(
        uid=uid,
        message_id=str(message.get("Message-ID", "")).strip(),
        received=received,
        from_name=from_name or sender,
        from_addr=sender,
        to=to,
        cc=cc,
        subject=subject or "(no subject)",
        text=text,
        is_reply=is_reply,
        list_mail=list_mail,
    )
    email_.facts = {
        "sender_is_someone_you_have_emailed": sender in contacts,
        "addressed_directly_to_you": any(a in my_addresses for a in to),
        "number_of_recipients": len(to) + len(cc),
        "is_a_reply_in_a_thread": is_reply,
        "is_mailing_list_or_bulk_mail": list_mail,
        "hours_ago": round(max(0.0, ((now or time.time()) - received) / 3600), 1),
    }
    return email_
