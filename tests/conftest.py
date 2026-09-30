import pytest

from fakes import FakeCortex, FakeJev, scenario_rules
from thalamus.brain import Brain
from thalamus.config import Settings
from thalamus.core.memory_store import MemoryStore


@pytest.fixture
def store():
    store = MemoryStore(":memory:")
    yield store
    store.close()


@pytest.fixture
def brain(store):
    return Brain(Settings(), jev=FakeJev(scenario_rules), cortex=FakeCortex(), memory=store)
