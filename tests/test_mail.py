"""Phase 3: email parsing, read-only IMAP, JEV triage, notifications, and chat integration."""

import imaplib
import os
import time

import pytest

from fakes import FakeCortex, FakeJev, FakeMailSource, FakeNotifier, raw_email, scenario_rules
from thalamus.brain import Brain
from thalamus.config import Settings
from thalamus.mail.accounts import Account
from thalamus.mail.imap import ImapSource, MailError
from thalamus.mail.parse import parse_message, strip_quoted
from thalamus.mail.secrets import FileSecrets, MemorySecrets, default_secrets
from thalamus.mail.service import MailService

NOW = time.time()


def make_service(store, source, notifier=None, settings=None):
    secrets = MemorySecrets()
    service = MailService(
        store, settings or Settings(), FakeJev(scenario_rules), FakeCortex(), secrets,
        source=source, notifier=notifier or FakeNotifier(),
    )
    return service


async def add_gmail(service, address="me@example.com"):
    return await service.add_password_account(service.draft_account("gmail", address), "app-password")


def test_parse_html_quotes_and_facts():
    raw = raw_email(
        "Re: Dinner Friday?",
        "<p>Sounds great, see you at 7!</p><p>On Mon, Bob wrote:</p><blockquote>&gt; dinner?</blockquote>",
        html=True, extra_headers="In-Reply-To: <x@y>\r\nList-Unsubscribe: <mailto:u@x>\r\n",
    )
    mail = parse_message(raw, 7, my_addresses={"me@example.com"}, contacts={"alex@example.com"}, now=NOW)
    assert mail.text.startswith("Sounds great, see you at 7!") and "Bob wrote" not in mail.text
    assert mail.facts["sender_is_someone_you_have_emailed"] and mail.facts["addressed_directly_to_you"]
    assert mail.facts["is_a_reply_in_a_thread"] and mail.facts["is_mailing_list_or_bulk_mail"]
    assert strip_quoted("Thanks!\n> old line\n> more") == "Thanks!"


def test_secrets_fall_back_to_a_private_file(tmp_path):
    store = default_secrets(tmp_path)  # this sandbox has no OS keychain
    assert isinstance(store, FileSecrets)
    store.set("mail:1", "s3cret")
    assert FileSecrets(tmp_path / "secrets.json").get("mail:1") == "s3cret"
    assert oct(os.stat(tmp_path / "secrets.json").st_mode)[-3:] == "600"
    store.delete("mail:1")
    assert store.get("mail:1") is None


async def test_triage_surfaces_what_matters_and_notifies_once(store):
    source = FakeMailSource(inbox=[
        raw_email("Your electricity bill is due Friday", "Invoice #42: $86.20 due 10/03", sender="PG&E <bill@pge.com>"),
        raw_email("50% off everything!", "Huge sale this weekend", sender="Shop <deals@shop.com>",
                  extra_headers="List-Unsubscribe: <mailto:x@shop.com>\r\n"),
        raw_email("Dinner Friday?", "Hey, can you make dinner Friday at 7?", sender="Alex <alex@example.com>"),
        raw_email("Verify your account now", "Click here to verify your account or it will be closed",
                  sender="Security <no-reply@bank-secure.biz>"),
    ], contacts={"alex@example.com"})
    notifier = FakeNotifier()
    service = make_service(store, source, notifier)
    service.configure_notify(enabled=True)
    await add_gmail(service)

    summary = await service.check_all()
    assert summary == {"checked": 1, "new": 4, "surfaced": 2, "notified": 2, "errors": []}
    items = {i["subject"]: i for i in service.items()}
    assert items["Your electricity bill is due Friday"]["surfaced"] == 1
    assert items["Your electricity bill is due Friday"]["reason"]
    assert items["50% off everything!"]["surfaced"] == 0
    assert items["Verify your account now"]["category"] == "spam"  # scams never surface
    assert items["Dinner Friday?"]["known_sender"] == 1
    assert {s["title"] for s in notifier.sent} == {"Important: PG&E", "Important: Alex"}
    assert "mail.google.com" in items["Dinner Friday?"]["link"]

    again = await service.check_all()  # nothing new, nothing re-sent
    assert again["new"] == 0 and len(notifier.sent) == 2


async def test_first_check_only_alerts_about_recent_mail(store):
    old = raw_email("Overdue invoice", "Your bill is overdue", when=NOW - 30 * 3600)
    service = make_service(store, FakeMailSource(inbox=[old]))
    service.configure_notify(enabled=True)
    await add_gmail(service)
    result = await service.check_all()
    assert result["surfaced"] == 1 and result["notified"] == 0


async def test_daily_budget_stops_checks(store):
    settings = Settings()
    settings.mail.daily_budget_usd = 0.0
    service = make_service(store, FakeMailSource(inbox=[raw_email("hi", "hello")]), settings=settings)
    await add_gmail(service)
    result = await service.check_all()
    assert result["checked"] == 0 and "budget" in result["errors"][0]


async def test_a_failing_account_is_reported_not_fatal(store):
    source = FakeMailSource()
    service = make_service(store, source)
    await add_gmail(service)
    source.fail = MailError("me@example.com: the mail server rejected the login.")
    result = await service.check_all()
    assert result["errors"] == ["me@example.com: the mail server rejected the login."]
    assert service.accounts()[0].last_error.startswith("me@example.com")


async def test_asking_about_email_checks_and_answers_from_triage(store):
    source = FakeMailSource(inbox=[raw_email("Your electricity bill is due Friday", "Invoice $86.20 due 10/03",
                                             sender="PG&E <bill@pge.com>")])
    service = make_service(store, source)
    brain = Brain(Settings(), jev=FakeJev(scenario_rules), cortex=FakeCortex(), memory=store, mail=service)
    await add_gmail(service)
    notes = []

    async def on_status(note):
        notes.append(note)

    response = await brain.think("anything important in my email?", on_status=on_status)
    system = brain.cortex.calls[-1]["system"]
    assert "Checking your email…" in notes
    assert "The user's email (read-only triage" in system and "PG&E: Your electricity bill is due Friday" in system
    assert response.trace.find("mail", "inbox")[0].data["checked_now"] is True
    assert sum(store.spending(conversation="mail").values()) > 0
    assert brain.homeostasis.by_service.get("claude", 0) < 0.01  # triage isn't billed to the chat session


async def test_self_model_knows_about_email(store):
    service = make_service(store, FakeMailSource())
    brain = Brain(Settings(), jev=FakeJev(scenario_rules), cortex=FakeCortex(), memory=store, mail=service)
    await brain.think("can you read my inbox?")
    assert "Email: not connected yet" in brain.cortex.calls[-1]["system"]
    await add_gmail(service)
    await brain.think("can you read my inbox?")
    assert "Email: YES, read-only" in brain.cortex.calls[-1]["system"]


class FakeIMAP:
    """Records how the real ImapSource talks to a server."""

    calls: list = []

    def __init__(self, host, port, timeout=None):
        FakeIMAP.calls = [("connect", host, port)]

    def login(self, user, password):
        if password != "good":
            raise imaplib.IMAP4.error(b"[AUTHENTICATIONFAILED] Invalid credentials")
        FakeIMAP.calls.append(("login", user))

    def select(self, folder, readonly=False):
        FakeIMAP.calls.append(("select", folder, readonly))
        return "OK", [b"2"]

    def response(self, name):
        return name, [b"77"]

    def uid(self, command, *args):
        FakeIMAP.calls.append(("uid", command, *args))
        if command == "SEARCH":
            return "OK", [b"5 6"]
        return "OK", [(b"5 (BODY[] {10}", raw_email("Hello", "Body text")), b")"]

    def logout(self):
        FakeIMAP.calls.append(("logout",))


def test_imap_is_read_only_and_explains_bad_logins(monkeypatch):
    monkeypatch.setattr(imaplib, "IMAP4_SSL", FakeIMAP)
    account = Account(id=1, provider="gmail", address="me@gmail.com", host="imap.gmail.com", port=993,
                      username="me@gmail.com", auth="password")
    result = ImapSource(lambda a: "good").fetch_new(account, my_addresses={"me@gmail.com"}, contacts=set())
    assert ("select", "INBOX", True) in FakeIMAP.calls  # EXAMINE, never SELECT read-write
    fetches = [c for c in FakeIMAP.calls if c[:2] == ("uid", "FETCH")]
    assert fetches and all("BODY.PEEK" in c[3] for c in fetches)  # peek never sets \\Seen
    assert result.last_uid == 6 and result.uidvalidity == 77 and result.emails[0].subject == "Hello"

    with pytest.raises(MailError, match="Google app password"):
        ImapSource(lambda a: "bad").test(account)


class FakeOutlook:
    def __init__(self):
        self.cache = {}

    def start(self, client_id):
        return {"user_code": "ABCD-1234", "verification_uri": "https://microsoft.com/devicelogin", "message": "go"}

    def finish(self, client_id, flow, cache_name):
        self.cache[cache_name] = "token-cache"
        return "me@outlook.com"

    def token(self, client_id, cache_name):
        return "access-token"


async def test_outlook_sign_in_creates_an_account(store):
    import asyncio

    source = FakeMailSource()
    service = MailService(store, Settings(), FakeJev(scenario_rules), FakeCortex(), MemorySecrets(),
                          source=source, outlook=FakeOutlook())
    started = await service.outlook_start("me@outlook.com", "client-123")
    assert started["user_code"] == "ABCD-1234" and started["status"] == "pending"
    for _ in range(50):
        await asyncio.sleep(0.01)
        status = service.outlook_status(started["flow_id"])
        if status["status"] != "pending":
            break
    assert status["status"] == "done", status
    [account] = service.accounts()
    assert account.auth == "oauth" and account.client_id == "client-123" and account.host == "outlook.office365.com"
    assert source.tested == ["me@outlook.com"]
