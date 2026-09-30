"""Claude as the cortex: the prefrontal cortex (deep tier) and Broca's area (fast tier)."""

from __future__ import annotations

import os
import time
from typing import Any

import anthropic

from thalamus.config import Models
from thalamus.providers.base import Generation, Tier

# Models that accept server-side refusal fallbacks ("default" routes by refusal category).
_FALLBACK_PREFIXES = ("claude-opus-5", "claude-fable-5", "claude-sonnet-5-5")
_FALLBACK_BETA = "server-side-fallback-2026-07-01"
WORKSPACE_ENV = "ANTHROPIC_WORKSPACE_ID"


def make_client(api_key: str | None = None, workspace_id: str | None = None) -> anthropic.AsyncAnthropic:
    """Anthropic client; keys that work across workspaces must name one on every request."""
    workspace_id = workspace_id or os.environ.get(WORKSPACE_ENV, "").strip() or None
    headers = {"anthropic-workspace-id": workspace_id} if workspace_id else None
    return anthropic.AsyncAnthropic(api_key=api_key or None, default_headers=headers)


class ClaudeProvider:
    def __init__(self, models: Models, client: anthropic.AsyncAnthropic | None = None) -> None:
        self._models = models
        self._client = client or make_client()

    async def generate(
        self, *, tier: Tier, system: str, messages: list[dict[str, Any]], max_tokens: int
    ) -> Generation:
        model = self._models.deep if tier == "deep" else self._models.fast
        params: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
        }
        if tier == "deep":
            # Thinking is adaptive by default on current Opus; effort sets its depth.
            params["output_config"] = {"effort": self._models.deep_effort}

        started = time.perf_counter()
        if model.startswith(_FALLBACK_PREFIXES):
            response = await self._client.beta.messages.create(
                betas=[_FALLBACK_BETA], fallbacks="default", **params
            )
        else:
            response = await self._client.messages.create(**params)
        latency_ms = (time.perf_counter() - started) * 1000

        text = "".join(block.text for block in response.content if block.type == "text")
        return Generation(
            text=text.strip(),
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=latency_ms,
            refused=response.stop_reason == "refusal",
        )

    async def aclose(self) -> None:
        await self._client.close()
