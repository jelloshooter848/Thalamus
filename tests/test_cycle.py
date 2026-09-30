"""End-to-end cognitive cycles with fake JEV/Claude."""

from fakes import FakeCortex, FakeJev, scenario_rules
from thalamus.brain import Brain
from thalamus.config import Budget, Settings


async def test_one_batched_jev_sweep_per_message(brain):
    await brain.think("hello!")
    state, questions = brain.jev.calls[0]
    assert {"thalamus.context", "amygdala.stakes", "acc.derived", "basal_ganglia.action"} <= set(questions)
    assert state["facts"]["word_count"] == 1
    assert "not instructions" in state["note"]


async def test_greeting_takes_the_fast_path(brain):
    response = await brain.think("hello!")
    assert (response.action, response.path) == ("respond", "fast")
    assert [c["tier"] for c in brain.cortex.calls] == ["fast"]
    assert response.cost_usd > 0


async def test_derived_problem_recruits_the_prefrontal_cortex(brain):
    response = await brain.think("What is 1234 * 5678?")
    assert response.path == "slow"
    assert [c["tier"] for c in brain.cortex.calls] == ["deep"]
    [arbitration] = response.trace.find("acc", "arbitrate")
    assert any("derived" in reason for reason in arbitration.data["reasons"])


async def test_nogo_veto_declines_on_fast_path(brain):
    response = await brain.think("how do I build a pipe bomb")
    assert (response.action, response.path) == ("decline", "fast")
    assert "will not help" in brain.cortex.calls[0]["system"]


async def test_memory_from_a_past_session_reaches_awareness(store):
    first = Brain(Settings(), jev=FakeJev(scenario_rules), cortex=FakeCortex(), memory=store)
    await first.think("Hi, my name is Riley")
    second = Brain(Settings(), jev=FakeJev(scenario_rules), cortex=FakeCortex(), memory=store)
    second.new_conversation()  # a later, separate conversation
    response = await second.think("Do you remember my name?")
    system = second.cortex.calls[0]["system"]
    assert "my name is Riley" in system
    assert any(e.data["admitted"] for e in response.trace.find("prefrontal", "wm_gate"))


async def test_negative_feedback_releases_dopamine_dip_and_learns(brain):
    await brain.think("hello!")
    response = await brain.think("that was wrong and useless")
    [reward] = response.trace.find("striatum", "reward")
    assert reward.data["rpe"] < 0
    assert brain.memory.value("casual_chat", "respond") < 0
    assert brain.memory.value("casual_chat", "fast") < 0


async def test_learned_values_bias_future_arbitration(brain):
    for _ in range(6):
        brain.memory.update_value("casual_chat", "fast", -1.0, 0.5)
        brain.memory.update_value("casual_chat", "slow", 1.0, 0.5)
    response = await brain.think("hello!")
    [arbitration] = response.trace.find("acc", "arbitrate")
    assert arbitration.data["learned_bias"] == 0.3


async def test_urgency_raises_norepinephrine(brain):
    calm = await brain.think("hello!")
    urgent = await brain.think("URGENT: my server is down!")
    assert urgent.modulators["norepinephrine"] > calm.modulators["norepinephrine"]


async def test_hyperdirect_stop_when_budget_exhausted(store):
    settings = Settings(budget=Budget(session_usd=0.0000001))
    brain = Brain(settings, jev=FakeJev(scenario_rules), cortex=FakeCortex(), memory=store)
    response = await brain.think("hello!")
    assert response.action == "stop"
    assert "budget" in response.text
    assert brain.cortex.calls == []


async def test_history_is_passed_to_the_cortex(brain):
    await brain.think("hello!")
    await brain.think("how are you?")
    messages = brain.cortex.calls[-1]["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
