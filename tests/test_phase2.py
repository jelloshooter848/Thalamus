"""Phase 2: sleep consolidation, verified facts, semantic recall, and forgetting/correcting."""

from fakes import FakeCortex, FakeJev, scenario_rules
from thalamus.brain import Brain
from thalamus.config import Settings

FACTS_HEADING = "What you know about the user"


def make_brain(store):
    return Brain(Settings(), jev=FakeJev(scenario_rules), cortex=FakeCortex(), memory=store)


async def test_sleep_distills_verified_facts_and_rejects_overreach(store):
    brain = make_brain(store)
    await brain.think("Hi, my name is Riley")
    await brain.think("I live in Gilroy and I love hiking")
    report = await brain.sleep(trigger="manual")

    facts = [row["text"] for row in store.facts()]
    assert set(facts) == {"The user's name is Riley.", "The user lives in Gilroy."}
    assert report["rejected"][0]["text"] == "The user is a professional mountaineer."  # JEV caught it
    assert report["cost_usd"] > 0 and store.sleep_reports(1)[0]["trigger"] == "manual"
    sleep_call = next(c for c in brain.cortex.calls if c.get("json"))
    assert sleep_call["tier"] == "memory"

    again = await brain.sleep(trigger="idle")  # nothing new since the watermark
    assert again["note"] == "nothing new to consolidate"


async def test_facts_reach_awareness_in_later_conversations(store):
    brain = make_brain(store)
    await brain.think("Hi, my name is Riley")
    await brain.think("I live in Gilroy")
    await brain.sleep()
    brain.new_conversation()
    await brain.think("hello!")
    system = brain.cortex.calls[-1]["system"]
    assert FACTS_HEADING in system and "The user's name is Riley." in system


async def test_sleep_updates_a_fact_that_changed(store):
    brain = make_brain(store)
    await brain.think("I live in Gilroy")
    await brain.think("hello!")
    await brain.sleep()
    await brain.think("big news, I moved to Portland")
    await brain.think("hello!")
    report = await brain.sleep()
    assert report["updated"] == [{"from": "The user lives in Gilroy.", "to": "The user lives in Portland."}]
    assert [row["text"] for row in store.facts()] == ["The user lives in Portland."]


async def test_forget_that_removes_facts_and_memories(store):
    brain = make_brain(store)
    await brain.think("I live in Gilroy")
    await brain.think("hello!")
    await brain.sleep()
    response = await brain.think("please forget where I live, Gilroy")
    assert store.facts() == []
    system = brain.cortex.calls[-1]["system"]
    assert "You just forgot, at the user's request" in system and "Gilroy" in system
    assert response.trace.find("reconsolidation", "forget")
    assert not [e for e in store.recall("Gilroy", k=5) if "I live in Gilroy" in e.text]


async def test_a_correction_replaces_the_wrong_fact(store):
    brain = make_brain(store)
    await brain.think("Hi, my name is Riley")
    await brain.think("hello!")
    await brain.sleep()
    await brain.think("that's not right, my name is actually Sam")
    facts = [row["text"] for row in store.facts()]
    assert facts == ["The user's name is Sam."]
    assert "You just updated your memory" in brain.cortex.calls[-1]["system"]


async def test_self_model_counts_known_facts(store):
    brain = make_brain(store)
    await brain.think("Hi, my name is Riley")
    await brain.think("I live in Gilroy")
    await brain.sleep()
    await brain.think("can you remember things about me?")
    assert "plus 2 distilled facts about the user" in brain.cortex.calls[-1]["system"]


async def test_sleep_is_due_only_after_idle_with_new_material(store):
    brain = make_brain(store)
    await brain.think("Hi, my name is Riley")
    await brain.think("I live in Gilroy")
    assert not brain.sleep_due()  # just talked
    brain.last_activity -= 31 * 60
    assert brain.sleep_due()
    await brain.sleep(trigger="idle")
    assert not brain.sleep_due()  # nothing new
