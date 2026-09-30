from thalamus.config import Budget
from thalamus.core.homeostasis import Homeostasis
from thalamus.core.neuromodulators import BASELINES, Neuromodulators
from thalamus.core.signals import Signal
from thalamus.core.workspace import GlobalWorkspace
from thalamus.providers.jev_patterns import composite_score, confidence_gate, noul_gate


def test_workspace_broadcasts_top_k_by_salience():
    ws = GlobalWorkspace(capacity=2)
    for i, salience in enumerate([0.2, 0.9, 0.5]):
        ws.submit(Signal("test", "memory", f"item {i}", key=f"k{i}", salience=salience))
    assert [s.key for s in ws.compete()] == ["k1", "k2"]
    assert ws.candidates == []


def test_inhibition_of_return_lets_new_memories_through():
    ws = GlobalWorkspace(capacity=1)
    ws.submit(Signal("test", "memory", "old", key="old", salience=0.6))
    ws.compete()
    ws.submit(Signal("test", "memory", "old", key="old", salience=0.6))
    ws.submit(Signal("test", "memory", "new", key="new", salience=0.5))
    assert [s.key for s in ws.compete()] == ["new"]


def test_neuromodulators_relax_toward_baseline():
    mods = Neuromodulators()
    mods.nudge("norepinephrine", 0.6)
    before = mods.norepinephrine
    mods.relax()
    assert BASELINES["norepinephrine"] < mods.norepinephrine < before
    mods.nudge("dopamine", 5)
    assert mods.dopamine == 1.0


def test_arousal_narrows_the_spotlight():
    mods = Neuromodulators()
    calm = mods.workspace_capacity(7)
    mods.nudge("norepinephrine", 0.7)
    assert mods.workspace_capacity(7) < calm


def test_homeostasis_prices_and_exhaustion():
    body = Homeostasis(Budget(session_usd=0.01))
    cost = body.record("claude-opus-5-5", 1000, 100)
    assert abs(cost - (1000 * 4 + 100 * 20) / 1e6) < 1e-12
    assert body.record("jev-latest", 1_000_000) == 0.042
    assert body.exhausted
    assert body.deliberation_cost == 1.0


def test_jev_patterns():
    assert confidence_gate(0.95, act_at=0.9, confirm_at=0.6) == "act"
    assert confidence_gate(0.7, act_at=0.9, confirm_at=0.6) == "confirm"
    assert confidence_gate(0.5, act_at=0.9, confirm_at=0.6) == "defer"
    assert noul_gate(0.5, yes_at=0.8, no_at=0.2) == "unsure"
    assert composite_score({"a": 1.0, "b": 0.0}, {"a": 3, "b": 1}) == 0.75
