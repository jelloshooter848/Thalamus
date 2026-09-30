"""Cost tracking: versioned model names are priced, and costs are split by service."""

from fakes import FakeCortex, FakeJev, FakeSearch, scenario_rules
from thalamus.brain import Brain
from thalamus.config import Budget, Settings
from thalamus.core.homeostasis import Homeostasis
from thalamus.providers.base import Generation
from thalamus.providers.search import WebResult


def test_versioned_model_names_are_priced_by_family():
    body = Homeostasis(Budget())
    assert body.record("claude-haiku-4-5-20251001", 1_000_000, 0) == 1.0
    assert body.record("claude-opus-5-5-20260915", 0, 1_000_000) == 20.0
    # Longest family wins: opus-5-5 is not priced as opus-5.
    assert body.price("claude-opus-5-5") == (4.0, 20.0)
    assert body.price("claude-opus-5-20260101") == (5.0, 25.0)
    assert not body.unpriced


def test_unknown_models_are_flagged_not_silently_free():
    body = Homeostasis(Budget())
    assert body.record("mystery-model", 1000, 1000) == 0.0
    assert body.unpriced == {"mystery-model"}


class VersionedCortex(FakeCortex):
    async def generate(self, **kwargs):
        await super().generate(**kwargs)
        return Generation(text="reply", model="claude-haiku-4-5-20251001", input_tokens=2000, output_tokens=200)


async def test_turn_cost_includes_claude_and_is_broken_down(store):
    search = FakeSearch(results=[WebResult("Weather in Boston", "https://w.example", "Boston weather: rain")])
    brain = Brain(Settings(), jev=FakeJev(scenario_rules), cortex=VersionedCortex(), memory=store, search=search)
    response = await brain.think("what's the weather in Boston?")
    parts = response.cost_breakdown
    # Two Haiku calls (query writer + reply), each 2000 in / 200 out at $1/$5 per million.
    assert abs(parts["claude"] - 2 * (2000 * 1 + 200 * 5) / 1e6) < 1e-9
    assert parts["tavily"] == 0.008
    assert parts["jev"] > 0
    assert abs(sum(parts.values()) - response.cost_usd) < 1e-6
    assert brain.snapshot()["budget"]["by_service"]["claude"] == parts["claude"]
