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
    for key in ("TYPESAFE_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_WORKSPACE_ID", "TAVILY_API_KEY"):
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
    assert status == {
        "typesafe": False, "anthropic": False, "workspace": False, "web": False, "ready": False, "local": True
    }


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


def test_setup_saves_optional_search_key(tmp_path):
    client, env = make_client(tmp_path)
    body = {"typesafe_key": "ts-1", "anthropic_key": "sk-ant-1", "tavily_key": "tvly-1"}
    result = client.post("/api/setup", json=body, headers=HEADERS).json()
    assert result["status"]["web"] is True
    assert "TAVILY_API_KEY=tvly-1" in env.read_text()


def test_blank_fields_keep_saved_values(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_saved")
    client, env = make_client(tmp_path)
    client.post("/api/setup", json={"tavily_key": "tvly-2"}, headers=HEADERS)
    saved = env.read_text()
    assert "TAVILY_API_KEY=tvly-2" in saved
    assert "ANTHROPIC_WORKSPACE_ID" not in saved  # untouched, not blanked
    assert client.get("/api/status", headers=HEADERS).json()["workspace"] is True


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


def test_chat_streams_text_then_the_full_result(tmp_path):
    import json as _json

    client, _ = make_client(tmp_path)
    with client.stream("POST", "/api/chat/stream", json={"message": "What is 12 * 34?"}, headers=HEADERS) as r:
        events = [_json.loads(line) for line in r.iter_lines() if line]
    kinds = [e["type"] for e in events]
    assert kinds[0] == "status" and events[0]["text"] == "Thinking it through…"
    assert "".join(e["delta"] for e in events if e["type"] == "text") == "[deep] reply"
    assert kinds[-1] == "done" and events[-1]["path"] == "slow" and events[-1]["trace"]


def test_stream_reports_provider_errors(tmp_path):
    import json as _json

    class BrokenCortex(FakeCortex):
        async def generate(self, **kwargs):
            raise anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com"))

    client, _ = make_client(tmp_path, cortex=BrokenCortex())
    with client.stream("POST", "/api/chat/stream", json={"message": "hello!"}, headers=HEADERS) as r:
        events = [_json.loads(line) for line in r.iter_lines() if line]
    assert events[-1] == {"type": "error", "error": {"message": "Couldn't reach Anthropic (Claude).", "hint": "Check your internet connection.", "field": None}}


def test_facts_can_be_viewed_edited_pinned_and_deleted(tmp_path):
    client, _ = make_client(tmp_path)
    client.post("/api/chat", json={"message": "Hi, my name is Riley"}, headers=HEADERS)
    client.post("/api/chat", json={"message": "I live in Gilroy"}, headers=HEADERS)
    report = client.post("/api/sleep", json={}, headers=HEADERS).json()
    assert len(report["added"]) == 2

    data = client.get("/api/facts", headers=HEADERS).json()
    assert data["reports"][0]["trigger"] == "manual" and data["pending"] == 0
    fact = next(f for f in data["facts"] if "Gilroy" in f["text"])
    client.patch(f"/api/facts/{fact['id']}", json={"text": "The user lives in Gilroy, CA.", "pinned": True},
                 headers=HEADERS)
    manual = client.post("/api/facts", json={"text": "The user prefers short answers.", "category": "preference"},
                         headers=HEADERS).json()
    facts = client.get("/api/facts", headers=HEADERS).json()["facts"]
    assert facts[0]["text"] == "The user lives in Gilroy, CA." and facts[0]["pinned"] == 1
    assert any(f["origin"] == "you" for f in facts)
    client.delete(f"/api/facts/{manual['id']}", headers=HEADERS)
    assert len(client.get("/api/facts", headers=HEADERS).json()["facts"]) == 2
