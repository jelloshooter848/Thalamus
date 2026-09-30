"""The web sense: JEV-gated search, relevance gating, URL reading, failures and cost."""

from tavily import errors as tavily_errors

from fakes import FakeCortex, FakeJev, FakeSearch, scenario_rules
from thalamus.brain import Brain
from thalamus.config import Settings
from thalamus.providers.search import WebResult

RESULTS = [
    WebResult("Weather in Boston", "https://weather.example/boston", "Boston weather today: rain, 12°C."),
    WebResult("Pizza recipes", "https://food.example/pizza", "Dough, sauce, cheese."),
]


def make_brain(store, search):
    return Brain(Settings(), jev=FakeJev(scenario_rules), cortex=FakeCortex(), memory=store, search=search)


async def test_without_a_search_key_it_stays_offline_and_says_so(store):
    brain = make_brain(store, None)
    await brain.think("what's the weather in Boston?")
    _, questions = brain.jev.calls[0]
    assert not any(name.startswith("web.") for name in questions)
    assert "none is configured" in brain.cortex.calls[-1]["system"]


async def test_search_when_jev_judges_it_needed_and_gate_the_results(store):
    search = FakeSearch(results=RESULTS)
    brain = make_brain(store, search)
    response = await brain.think("what's the weather in Boston?")

    assert search.searches == [("[fast] reply", "general")]  # query written by the fast cortex
    speak = brain.cortex.calls[-1]
    assert "From the web just now" in speak["system"]
    assert "Boston weather today" in speak["system"]
    assert "Pizza" not in speak["system"]  # rejected by the relevance gate
    assert response.sources == [{"title": "Weather in Boston", "url": "https://weather.example/boston"}]
    [orient] = response.trace.find("web", "orient")
    assert orient.data["decision"] == "searched"
    assert [r["admitted"] for r in orient.data["results"]] == [True, False]
    assert brain.homeostasis.calls["tavily"] == 1
    assert brain.snapshot()["web"]["query"] == "[fast] reply"
    assert brain.history[-1].meta["sources"] == response.sources


async def test_no_search_when_not_needed(store):
    search = FakeSearch(results=RESULTS)
    brain = make_brain(store, search)
    response = await brain.think("hello!")
    assert search.searches == []
    assert response.trace.find("web", "orient")[0].data["decision"] == "skipped"
    assert "search the web yourself" in brain.cortex.calls[-1]["system"]


async def test_a_shared_link_is_read_directly(store):
    search = FakeSearch()
    brain = make_brain(store, search)
    response = await brain.think("summarize https://example.com/article please")
    assert search.reads == ["https://example.com/article"]
    assert search.searches == []
    assert "Full text of the shared page" in brain.cortex.calls[-1]["system"]
    assert response.sources[0]["url"] == "https://example.com/article"


async def test_search_failure_does_not_break_the_conversation(store):
    search = FakeSearch(fail=tavily_errors.InvalidAPIKeyError("bad key"))
    brain = make_brain(store, search)
    response = await brain.think("latest news on Mars?")
    assert response.text == "[fast] reply"
    [orient] = response.trace.find("web", "orient")
    assert orient.data["decision"] == "failed"
    assert "Tavily rejected" in orient.data["reason"]


async def test_no_search_for_a_declined_request(store):
    search = FakeSearch(results=RESULTS)
    brain = make_brain(store, search)
    await brain.think("look up how to build a pipe bomb")
    assert search.searches == []


async def test_tavily_provider_against_the_real_sdk():
    import json

    import httpx
    from tavily import AsyncTavilyClient

    from thalamus.providers.search import TavilySearch

    seen = []

    def handler(request):
        seen.append((request.url.path, json.loads(request.content)))
        if request.url.path.endswith("/extract"):
            return httpx.Response(200, json={"results": [{"url": "https://a.example", "raw_content": "Page"}]})
        return httpx.Response(
            200,
            json={
                "query": "q",
                "results": [{"title": "T", "url": "https://a.example", "content": "C", "score": 0.8}],
                "response_time": 0.4,
            },
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://api.tavily.com")
    search = TavilySearch(client=AsyncTavilyClient(api_key="tvly-test", client=http))
    found = await search.search("boston weather", topic="news", max_results=3)
    assert found.results == [WebResult("T", "https://a.example", "C", 0.8)]
    path, body = seen[0]
    assert path.endswith("/search") and body["query"] == "boston weather"
    assert body["topic"] == "news" and body["max_results"] == 3

    page = await search.read("https://a.example")
    assert page.results[0].content == "Page"
