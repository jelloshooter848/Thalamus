import asyncio
import json
import os

import anthropic
import httpx2

from thalamus.config import Models, save_env
from thalamus.health import explain
from thalamus.providers.claude import ClaudeProvider, make_client


def test_save_env_updates_in_place_and_keeps_other_lines(tmp_path, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("NEW_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("# my comment\nTYPESAFE_API_KEY=old\nOTHER=keep\n")
    save_env({"TYPESAFE_API_KEY": "new", "NEW_KEY": "v"}, env)
    assert env.read_text() == "# my comment\nTYPESAFE_API_KEY=new\nOTHER=keep\nNEW_KEY=v\n"
    assert os.environ["TYPESAFE_API_KEY"] == "new"


def test_workspace_id_is_sent_with_claude_requests(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_123")
    assert make_client(api_key="k").default_headers["anthropic-workspace-id"] == "wrkspc_123"

    seen = []

    def handler(request):
        seen.append(request)
        return httpx2.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-haiku-4-5",
                "content": [{"type": "text", "text": "ready"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 5, "output_tokens": 1},
            },
        )

    client = anthropic.AsyncAnthropic(
        api_key="k",
        default_headers={"anthropic-workspace-id": "wrkspc_123"},
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    generation = asyncio.run(
        ClaudeProvider(Models(), client=client).generate(
            tier="fast", system="s", messages=[{"role": "user", "content": "hi"}], max_tokens=10
        )
    )
    assert generation.text == "ready"
    assert seen[0].headers["anthropic-workspace-id"] == "wrkspc_123"
    assert json.loads(seen[0].content)["model"] == "claude-haiku-4-5"


def test_workspace_error_is_explained():
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx2.Response(400, request=request, json={"error": {"message": "workspace"}})
    error = anthropic.BadRequestError(
        "This API key is not scoped to a workspace", response=response, body=None
    )
    problem = explain(error)
    assert problem.field == "workspace"
    assert "wrkspc_" in problem.hint
