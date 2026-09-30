import time

from thalamus.core.memory_store import query_terms


def test_recall_ranks_relevant_and_important(store):
    now = time.time()
    store.encode(session="s1", turn=1, speaker="user", text="My name is Riley", context="personal", importance=0.9, now=now)
    store.encode(session="s1", turn=2, speaker="user", text="The weather is nice", context="casual_chat", importance=0.1, now=now)
    hits = store.recall("what is my name?", k=3, now=now)
    assert [h.text for h in hits] == ["My name is Riley"]


def test_forgetting_curve_and_reinforcement(store):
    old = time.time() - 30 * 24 * 3600
    faded = store.encode(session="s", turn=1, speaker="user", text="favourite colour teal", context="personal", importance=0.1, now=old)
    [hit] = store.recall("favourite colour", k=1)
    store.reinforce([faded])
    [again] = store.recall("favourite colour", k=1)
    assert again.score > hit.score
    assert again.recall_count == 1


def test_current_conversation_window_is_excluded(store):
    store.encode(session="now", turn=5, speaker="user", text="pizza toppings", context="casual_chat", importance=0.5)
    assert store.recall("pizza", k=3, exclude_session="now", exclude_from_turn=3) == []
    assert store.recall("pizza", k=3, exclude_session="now", exclude_from_turn=6)


def test_salient_memories_surface_without_a_lexical_cue(store):
    store.encode(session="s", turn=1, speaker="user", text="I am allergic to peanuts", context="personal", importance=1.0)
    assert store.recall("what should I cook?", k=3) == []
    assert store.recall("what should I cook?", k=3, include_salient=True)


def test_query_terms_are_safe_for_fts():
    assert query_terms('What\'s "my" name? OR DROP; *') == ["name", "drop"]


def test_value_learning_returns_prediction_error(store):
    rpe = store.update_value("coding", "fast", reward=-1.0, learning_rate=0.5)
    assert rpe == -1.0
    assert store.value("coding", "fast") == -0.5
    assert store.update_value("coding", "fast", reward=-1.0, learning_rate=0.5) == -0.5
