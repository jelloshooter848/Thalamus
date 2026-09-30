"""The browser app: token guard, setup flow, chat, errors and memory."""

import anthropic
import httpx2
import pytest
from fastapi.testclient import TestClient

from fakes import FakeCortex, FakeJev, scenario_rules
from thalamus.app import create_app
from thalamus.brain import Brain
from thalamus.config import Settings
from thalamus.core.memory_store import MemoryStore
from thalamus.health import Check, Problem

TOKEN = "test-token"
HEADERS = {"X-Thalamus-Token": TOKEN}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in ("TYPESAFE_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_WORKSPACE_ID"):
        monkeypatch.delenv(key, raising=False)


def make_client(tmp_path, cortex=None, checks_ok=True):
    store = MemoryStore(":memory:")

    def factory(settings):
        return Brain(settings, jev=FakeJev(scenario_rules), cortex=cortex or FakeCortex(), memory=store)

    async def checker(keys, settings):
        if checks_ok:
            return [Check("JEV", True, "ok"), Check("Claude fast", True, "ok")]
        return [Check("Claude fast", False, problem=Problem("needs workspace", "add it", "workspace"))]

    app = create_app(Settings(), env_path=tmp_path / ".env", brain_factory=factory, checker=checker, token=TOKEN)
    return TestClient(app), tmp_path / ".env"


def test_page_embeds_token_and_api_requires_it(tmp_path):
    client, _ = make_client(tmp_path)
    assert TOKEN in client.get("/").text
    assert client.get("/api/status").status_code == 403
    assert client.get("/api/status", headers={"X-Thalamus-Token": "wrong"}).status_code == 403


def test_foreign_host_is_rejected(tmp_path):
    client, _ = make_client(tmp_path)
    assert client.get("/", headers={"Host": "evil.example"}).status_code == 403


def test_status_reports_missing_keys(tmp_path):
    client, _ = make_client(tmp_path)
    status = client.get("/api/status", headers=HEADERS).json()
    assert status == {"typesafe": False, "anthropic": False, "workspace": False, "ready": False}


def test_setup_saves_keys_only_when_checks_pass(tmp_path):
    client, env = make_client(tmp_path, checks_ok=False)
    body = {"typesafe_key": "ts-1", "anthropic_key": "sk-ant-1", "workspace_id": ""}
    result = client.post("/api/setup", json=body, headers=HEADERS).json()
    assert result["ok"] is False
    assert result["checks"][0]["problem"]["field"] == "workspace"
    assert not env.exists()

    client, env = make_client(tmp_path, checks_ok=True)
    result = client.post("/api/setup", json={**body, "workspace_id": "wrkspc_1"}, headers=HEADERS).json()
    assert result["ok"] is True and result["status"]["ready"] is True
    saved = env.read_text()
    assert "TYPESAFE_API_KEY=ts-1" in saved
    assert "ANTHROPIC_API_KEY=sk-ant-1" in saved
    assert "ANTHROPIC_WORKSPACE_ID=wrkspc_1" in saved


def test_chat_returns_reply_and_trace(tmp_path, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "x")
    client, _ = make_client(tmp_path)
    data = client.post("/api/chat", json={"message": "What is 12 * 34?"}, headers=HEADERS).json()
    assert data["text"] == "[deep] reply"
    assert data["path"] == "slow"
    assert {"acc", "thalamus", "amygdala"} <= {event["region"] for event in data["trace"]}
    assert set(data["modulators"]) == {"dopamine", "norepinephrine", "serotonin", "acetylcholine"}


def test_provider_error_becomes_friendly_json(tmp_path):
    class BrokenCortex(FakeCortex):
        async def generate(self, **kwargs):
            raise anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com"))

    client, _ = make_client(tmp_path, cortex=BrokenCortex())
    response = client.post("/api/chat", json={"message": "hello!"}, headers=HEADERS)
    assert response.status_code == 502
    assert response.json()["error"]["message"] == "Couldn't reach Anthropic (Claude)."
    # The server keeps working afterwards.
    assert client.get("/api/status", headers=HEADERS).status_code == 200


def test_memory_list_and_clear(tmp_path):
    client, _ = make_client(tmp_path)
    client.post("/api/chat", json={"message": "Hi, my name is Riley"}, headers=HEADERS)
    memory = client.get("/api/memory", headers=HEADERS).json()
    assert any(e["text"] == "Hi, my name is Riley" for e in memory["episodes"])
    client.post("/api/memory/clear", json={}, headers=HEADERS)
    assert client.get("/api/memory", headers=HEADERS).json()["episodes"] == []


def test_history_mind_and_new_conversation(tmp_path):
    client, _ = make_client(tmp_path)
    assert client.get("/api/history", headers=HEADERS).json() == {"turns": []}
    assert client.get("/api/mind", headers=HEADERS).json() == {}

    client.post("/api/chat", json={"message": "hello!"}, headers=HEADERS)
    turns = client.get("/api/history", headers=HEADERS).json()["turns"]
    assert [t["role"] for t in turns] == ["user", "assistant"]
    assert turns[1]["meta"]["path"] == "fast"
    assert len(client.get("/api/mind", headers=HEADERS).json()["buffer"]) == 2

    client.post("/api/conversation/new", json={}, headers=HEADERS)
    assert client.get("/api/history", headers=HEADERS).json() == {"turns": []}
    assert client.get("/api/mind", headers=HEADERS).json()["buffer"] == []
