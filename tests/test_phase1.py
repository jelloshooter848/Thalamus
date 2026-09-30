"""Phase 1: conversations survive restarts, a cost ledger, streaming, and grounded lookups."""

from fakes import FakeCortex, FakeJev, FakeSearch, scenario_rules
from thalamus.brain import Brain
from thalamus.config import Settings
from thalamus.providers.search import WebResult


class Clock:
    def __init__(self) -> None:
        self.now = 1_900_000_000.0

    def __call__(self) -> float:
        return self.now


def make_brain(store, clock=None, search=None):
    return Brain(
        Settings(), jev=FakeJev(scenario_rules), cortex=FakeCortex(), memory=store, clock=clock or Clock(), search=search
    )


async def test_a_restart_resumes_the_conversation(store):
    clock = Clock()
    first = make_brain(store, clock)
    await first.think("Hi, my name is Riley")
    clock.now += 5 * 60
    restarted = make_brain(store, clock)  # same memory file, new process
    assert restarted.session == first.session
    assert [t.content for t in restarted.history] == ["Hi, my name is Riley", "[fast] reply"]
    assert restarted.history[1].meta["path"] == "fast"
    await restarted.think("how are you?")
    assert [m["role"] for m in restarted.cortex.calls[-1]["messages"]] == ["user", "assistant", "user"]


async def test_a_restart_after_a_long_gap_starts_fresh(store):
    clock = Clock()
    first = make_brain(store, clock)
    await first.think("hello!")
    clock.now += 2 * 3600
    later = make_brain(store, clock)
    assert later.session != first.session and later.history == []


async def test_every_turn_is_written_to_the_cost_ledger(store):
    brain = make_brain(store)
    response = await brain.think("hello!")
    spent = store.spending()
    assert abs(sum(spent.values()) - response.cost_usd) < 1e-9
    assert set(spent) == {"jev", "claude"}


async def test_replies_stream_and_status_notes_are_sent(store):
    brain = make_brain(store, search=FakeSearch(results=[WebResult("Weather", "https://w.example", "weather rain")]))
    chunks, notes = [], []

    async def on_text(delta):
        chunks.append(delta)

    async def on_status(note):
        notes.append(note)

    response = await brain.think("what's the weather?", on_text=on_text, on_status=on_status)
    assert "".join(chunks) == response.text
    assert notes == ["Searching the web…"]
    assert brain.cortex.calls[-1]["streamed"] is True
    assert not brain.cortex.calls[0]["streamed"]  # the query writer isn't shown to the user

    notes.clear()
    await brain.think("What is 1234 * 5678?", on_text=on_text, on_status=on_status)
    assert notes == ["Thinking it through…"]


async def test_grounded_lookups_stay_on_the_fast_path(store):
    brain = make_brain(store, search=FakeSearch(results=[WebResult("Weather", "https://w.example", "weather rain")]))
    brain.jev.rules = lambda state, questions: {
        **scenario_rules(state, questions),
        **(
            {
                "acc.effort": __import__("fakes").choice("moderate", ["trivial", "moderate", "hard"]),
                "acc.simple": __import__("fakes").NoulResult(0.2),  # probes disagree
            }
            if "results" not in state and "remembered_items" not in state
            else {}
        ),
    }
    response = await brain.think("what's the weather?")
    [arbitration] = response.trace.find("acc", "arbitrate")
    assert arbitration.data["grounded"] is True
    assert response.path == "fast"
