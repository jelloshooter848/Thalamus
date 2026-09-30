"""Connectivity checks for JEV and Claude, and plain-language explanations of provider errors.

Shared by `thalamus doctor` and the browser app's setup page.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import anthropic
from tavily import errors as tavily_errors
from typesafe_sdk import (
    Noul,
    TypeSafeAPIConnectionError,
    TypeSafeAPIError,
    TypeSafeAuthenticationError,
    TypeSafeError,
    TypeSafePermissionDeniedError,
    TypeSafeRateLimitError,
)

from thalamus.config import MissingJevKeyError
from thalamus.providers.base import DecisionProvider, LanguageProvider
from thalamus.providers.search import SearchProvider


@dataclass
class Problem:
    message: str
    hint: str = ""
    field: str | None = None  # which setup field to highlight: typesafe | anthropic | workspace


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    problem: Problem | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def explain(error: Exception) -> Problem:
    """Turn a provider exception into something a person can act on."""
    if isinstance(error, MissingJevKeyError):
        return Problem("A TypeSafe API key is required.", "Add it on the setup page.", "typesafe")
    if isinstance(error, TypeSafeAuthenticationError | TypeSafePermissionDeniedError):
        return Problem("TypeSafe rejected the API key.", "Check the key at typesafe.ai and paste it again.", "typesafe")
    if isinstance(error, TypeSafeRateLimitError):
        return Problem("JEV is rate limiting requests.", "Wait a moment and try again.")
    if isinstance(error, TypeSafeAPIConnectionError):
        return Problem("Couldn't reach TypeSafe (JEV).", "Check your internet connection.")
    if isinstance(error, TypeSafeAPIError | TypeSafeError):
        return Problem(f"JEV returned an error: {error}")

    if isinstance(error, tavily_errors.InvalidAPIKeyError | tavily_errors.MissingAPIKeyError):
        return Problem("Tavily rejected the API key.", "Check the key at app.tavily.com and paste it again.", "tavily")
    if isinstance(error, tavily_errors.UsageLimitExceededError):
        return Problem("The Tavily plan's search limit is used up.", "Check your usage at app.tavily.com.", "tavily")
    if isinstance(error, tavily_errors.ForbiddenError):
        return Problem("Tavily refused the request.", "Check the key's permissions at app.tavily.com.", "tavily")
    if isinstance(error, tavily_errors.BadRequestError | tavily_errors.TimeoutError):
        return Problem(f"The web search failed: {error}")

    if isinstance(error, anthropic.AuthenticationError):
        return Problem(
            "Anthropic rejected the API key.", "Create a key at console.anthropic.com and paste it again.", "anthropic"
        )
    if isinstance(error, anthropic.BadRequestError) and "workspace" in str(error).lower():
        return Problem(
            "This Anthropic key works across workspaces, so it needs a Workspace ID.",
            "In the Anthropic Console open Settings → Workspaces, copy the workspace ID "
            "(it starts with wrkspc_) and paste it in the Workspace ID field.",
            "workspace",
        )
    if isinstance(error, anthropic.PermissionDeniedError):
        return Problem(
            "The Anthropic key isn't allowed to do this.", "Check the key's workspace and permissions.", "anthropic"
        )
    if isinstance(error, anthropic.NotFoundError):
        return Problem("A Claude model isn't available to this key.", "Check the model names in thalamus.toml.")
    if isinstance(error, anthropic.RateLimitError):
        return Problem("Claude is rate limiting requests.", "Wait a moment and try again.")
    if isinstance(error, anthropic.APIConnectionError):
        return Problem("Couldn't reach Anthropic (Claude).", "Check your internet connection.")
    if isinstance(error, anthropic.APIStatusError):
        return Problem(f"Claude returned an error: {error.message}")
    return Problem(f"Unexpected error: {error}")


async def run_checks(
    jev: DecisionProvider, cortex: LanguageProvider, jev_model: str, search: SearchProvider | None = None
) -> list[Check]:
    checks = []
    try:
        decision = await jev.decide({"text": "Hello there!"}, {"greeting": Noul(instructions="Is text a greeting?")})
        checks.append(
            Check(
                f"JEV ({jev_model})",
                True,
                f"answered in {decision.latency_ms:.0f} ms (P(greeting)={decision.nouls['greeting'].p:.2f})",
            )
        )
    except Exception as error:  # noqa: BLE001 - every failure becomes a readable check result
        checks.append(Check(f"JEV ({jev_model})", False, problem=explain(error)))

    for tier in ("fast", "deep"):
        try:
            generation = await cortex.generate(
                tier=tier,
                system="Reply with the single word: ready",
                messages=[{"role": "user", "content": "Status?"}],
                max_tokens=64 if tier == "fast" else 2048,
            )
            detail = f"{generation.model} replied in {generation.latency_ms:.0f} ms"
            checks.append(Check(f"Claude {tier}", True, detail))
        except Exception as error:  # noqa: BLE001
            checks.append(Check(f"Claude {tier}", False, problem=explain(error)))
            if explain(error).field:  # same key problem would repeat for the other tier
                break

    if search is not None:
        try:
            outcome = await search.search("THALAMUS connectivity check", topic="general", max_results=1)
            checks.append(Check("Web search (Tavily)", True, f"answered in {outcome.latency_ms:.0f} ms"))
        except Exception as error:  # noqa: BLE001
            checks.append(Check("Web search (Tavily)", False, problem=explain(error)))
    return checks
