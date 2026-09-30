"""Phone access: who may connect, passcode login, sessions, and PC-only settings."""

import time

import pytest
from fastapi.testclient import TestClient

from fakes import FakeCortex, FakeJev, scenario_rules
from thalamus.app import create_app
from thalamus.brain import Brain
from thalamus.config import Settings
from thalamus.core.memory_store import MemoryStore
from thalamus.remote import (
    COOKIE,
    LoginThrottle,
    allowed_remote,
    check_passcode,
    hash_passcode,
    make_session,
    valid_session,
)

TOKEN = "test-token"
HEADERS = {"X-Thalamus-Token": TOKEN}
PHONE = ("100.101.102.103", 50000)  # a Tailscale address
LAN = ("192.168.1.20", 50000)


@pytest.fixture(autouse=True)
def env(monkeypatch):
    for key in ("THALAMUS_REMOTE", "THALAMUS_PASSCODE_HASH", "THALAMUS_SECRET", "TAVILY_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("TYPESAFE_API_KEY", "x")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "y")


def make_app(tmp_path, remote_active=True):
    store = MemoryStore(":memory:")
    return create_app(
        Settings(),
        env_path=tmp_path / ".env",
        brain_factory=lambda s: Brain(s, jev=FakeJev(scenario_rules), cortex=FakeCortex(), memory=store),
        token=TOKEN,
        remote_active=remote_active,
    )


def enable_phone(app, passcode="correct horse"):
    pc = TestClient(app)
    result = pc.post("/api/remote", json={"enabled": True, "passcode": passcode}, headers=HEADERS)
    assert result.json()["ok"]
    return pc


def phone(app, address=PHONE):
    return TestClient(app, client=address)


def test_passcode_hash_and_sessions():
    stored = hash_passcode("secret123")
    assert check_passcode("secret123", stored) and not check_passcode("secret124", stored)
    assert "secret123" not in stored
    cookie = make_session("k", days=1)
    assert valid_session(cookie, "k") and not valid_session(cookie, "other-secret")
    assert not valid_session(make_session("k", days=-1), "k")  # expired
    assert allowed_remote("100.64.0.5") and not allowed_remote("8.8.8.8") and not allowed_remote("192.168.1.2")
    assert allowed_remote("192.168.1.2", allow_lan=True)


def test_phone_is_refused_while_phone_access_is_off(tmp_path):
    app = make_app(tmp_path, remote_active=False)
    assert phone(app).get("/").status_code == 403


def test_non_tailscale_devices_are_refused(tmp_path):
    app = make_app(tmp_path)
    enable_phone(app)
    assert phone(app, LAN).get("/").status_code == 403
    assert phone(app, ("8.8.8.8", 1)).get("/").status_code == 403


def test_phone_must_log_in_then_can_chat(tmp_path):
    app = make_app(tmp_path)
    enable_phone(app)
    client = phone(app)
    page = client.get("/")
    assert "Enter the passcode" in page.text and TOKEN not in page.text  # login page, no token leak
    assert client.get("/api/status", headers=HEADERS).status_code == 401
    assert client.get("/manifest.webmanifest").json()["display"] == "standalone"
    assert client.get("/icon-180.png").headers["content-type"] == "image/png"

    assert client.post("/login", json={"passcode": "wrong"}).status_code == 401
    assert client.post("/login", json={"passcode": "correct horse"}).status_code == 200
    assert COOKIE in client.cookies
    assert TOKEN in client.get("/").text
    status = client.get("/api/status", headers=HEADERS).json()
    assert status["local"] is False and status["ready"] is True
    reply = client.post("/api/chat", json={"message": "hello!"}, headers=HEADERS).json()
    assert reply["text"] == "[fast] reply"


def test_settings_can_only_change_on_the_pc(tmp_path):
    app = make_app(tmp_path)
    enable_phone(app)
    client = phone(app)
    client.post("/login", json={"passcode": "correct horse"})
    blocked = client.post("/api/remote", json={"enabled": False}, headers=HEADERS)
    assert blocked.status_code == 403
    assert client.post("/api/setup", json={"typesafe_key": "evil"}, headers=HEADERS).status_code == 403


def test_changing_the_passcode_logs_phones_out(tmp_path):
    app = make_app(tmp_path)
    pc = enable_phone(app)
    client = phone(app)
    client.post("/login", json={"passcode": "correct horse"})
    assert client.get("/api/status", headers=HEADERS).status_code == 200
    pc.post("/api/remote", json={"enabled": True, "passcode": "new passcode"}, headers=HEADERS)
    assert client.get("/api/status", headers=HEADERS).status_code == 401


def test_repeated_wrong_passcodes_lock_out(tmp_path):
    app = make_app(tmp_path)
    enable_phone(app)
    client = phone(app)
    for _ in range(5):
        client.post("/login", json={"passcode": "nope"})
    assert client.post("/login", json={"passcode": "correct horse"}).status_code == 429

    throttle = LoginThrottle(max_failures=1, lockout_seconds=0.01)
    throttle.fail("x")
    time.sleep(0.02)
    assert not throttle.locked("x")


def test_remote_settings_validation_and_restart_hint(tmp_path):
    app = make_app(tmp_path, remote_active=False)
    pc = TestClient(app)
    assert pc.post("/api/remote", json={"enabled": True}, headers=HEADERS).status_code == 400  # no passcode
    assert pc.post("/api/remote", json={"enabled": True, "passcode": "123"}, headers=HEADERS).status_code == 400
    result = pc.post("/api/remote", json={"enabled": True, "passcode": "123456"}, headers=HEADERS).json()
    assert result["restart_required"] is True
    saved = (tmp_path / ".env").read_text()
    assert "THALAMUS_REMOTE=1" in saved and "THALAMUS_PASSCODE_HASH=scrypt$" in saved and "123456" not in saved
    status = pc.get("/api/remote", headers=HEADERS).json()
    assert status["enabled"] and status["passcode_set"] and not status["active"] and status["local"]


def test_qr_code_is_an_svg():
    from thalamus.remote import phone_urls, qr_svg

    urls = phone_urls(8765, {"ip": "100.64.0.9", "name": "my-pc.tail1234.ts.net"})
    assert urls == ["http://my-pc.tail1234.ts.net:8765/", "http://100.64.0.9:8765/"]
    assert qr_svg(urls[0]).lstrip().startswith("<?xml") or "<svg" in qr_svg(urls[0])
