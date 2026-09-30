"""Self-awareness: true capability facts reach the cortex when the user asks about THALAMUS."""

from fakes import FakeCortex, FakeJev, FakeSearch, scenario_rules
from thalamus.brain import Brain
from thalamus.config import Settings

HEADING = "Facts about yourself right now"


def make_brain(store, search=None):
    return Brain(Settings(), jev=FakeJev(scenario_rules), cortex=FakeCortex(), memory=store, search=search)


async def test_capability_question_brings_true_facts_into_awareness(store):
    brain = make_brain(store, FakeSearch())
    await brain.think("Hi, my name is Riley")
    response = await brain.think("can you search the internet now?")
    system = brain.cortex.calls[-1]["system"]
    assert HEADING in system
    assert "Internet: YES" in system
    assert "Long-term memory: YES. 2 remembered messages" in system
    assert response.trace.find("self_model", "introspect")


async def test_without_search_it_says_how_to_enable_it(store):
    brain = make_brain(store)
    await brain.think("can you search the internet now?")
    system = brain.cortex.calls[-1]["system"]
    assert "Internet: NO" in system and "Tavily API key in Settings" in system


async def test_ordinary_messages_skip_introspection(store):
    brain = make_brain(store, FakeSearch())
    response = await brain.think("hello!")
    assert HEADING not in brain.cortex.calls[-1]["system"]
    assert not response.trace.find("self_model", "introspect")
