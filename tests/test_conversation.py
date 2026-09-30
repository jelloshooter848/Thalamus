"""Conversation boundaries, the memory self-model, and the Mind snapshot."""

from fakes import FakeCortex, FakeJev, scenario_rules
from thalamus.brain import Brain
from thalamus.config import Settings


class Clock:
    def __init__(self) -> None:
        self.now = 1_900_000_000.0

    def __call__(self) -> float:
        return self.now


def make_brain(store, clock=None):
    return Brain(Settings(), jev=FakeJev(scenario_rules), cortex=FakeCortex(), memory=store, clock=clock or Clock())


async def test_prompt_carries_an_accurate_memory_self_model(store):
    brain = make_brain(store)
    await brain.think("hello!")
    system = brain.cortex.calls[0]["system"]
    assert "long-term episodic memory" in system
    assert "never claim you have no memory" in system
    assert "It is now" in system


async def test_new_conversation_forgets_the_thread_but_not_the_past(store):
    brain = make_brain(store)
    await brain.think("Hi, my name is Riley")
    old_session = brain.session
    brain.new_conversation()
    assert brain.history == [] and brain.session != old_session

    await brain.think("Do you remember my name?")
    call = brain.cortex.calls[-1]
    assert [m["role"] for m in call["messages"]] == ["user"]  # nothing carried in the thread...
    assert "Memories from earlier conversations" in call["system"]  # ...but it is remembered
    assert "my name is Riley" in call["system"]


async def test_long_idle_gap_starts_a_new_conversation(store):
    clock = Clock()
    brain = make_brain(store, clock)
    await brain.think("hello!")
    session = brain.session
    clock.now += 10 * 60
    await brain.think("still here")
    assert brain.session == session and len(brain.history) == 4

    clock.now += 31 * 60
    response = await brain.think("back again")
    assert brain.session != session and len(brain.history) == 2
    assert response.trace.find("hippocampus", "new_conversation")


async def test_snapshot_shows_buffer_working_memory_and_recall(store):
    brain = make_brain(store)
    await brain.think("Hi, my name is Riley")
    brain.new_conversation()
    await brain.think("Do you remember my name?")
    mind = brain.snapshot()

    assert [t["role"] for t in mind["buffer"]] == ["user", "assistant"]
    assert mind["conversation"]["turns"] == 1
    assert any("my name is Riley" in slot["text"] and slot["held"] == 1 for slot in mind["working_memory"])
    verdicts = {r["admitted"] for r in mind["recall"]}
    assert verdicts == {True, False}  # the name was let in, an unrelated memory was turned away
    assert set(mind["modulators"]) == {"dopamine", "norepinephrine", "serotonin", "acetylcholine"}


async def test_assistant_turns_keep_how_they_were_produced(store):
    brain = make_brain(store)
    await brain.think("What is 12 * 34?")
    assert brain.history[-1].meta["path"] == "slow"
