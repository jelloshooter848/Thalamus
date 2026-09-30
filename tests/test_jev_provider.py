"""The real TypeSafe SDK wired through THALAMUS's provider, against a mocked HTTP transport."""

import json

import httpx2
from typesafe_sdk import AsyncTypeSafeClient, Choice, Noul, Score

from thalamus.providers.jev import JevProvider


async def test_single_batched_request_is_parsed():
    seen = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "usage": {"input_tokens": 321, "output_tokens": 0},
                "answers": {
                    "yes": {"type": "noul", "noul": 0.93},
                    "pick": {
                        "type": "choice",
                        "choice": "b",
                        "confidence": 0.8,
                        "probabilities": {"a": 0.1, "b": 0.9},
                    },
                    "rate": {
                        "type": "score",
                        "score": 1.5,
                        "confidence": 0.7,
                        "legend": {"0": "low", "1": "mid", "2": "high"},
                        "probabilities": {"0": 0.1, "1": 0.3, "2": 0.6},
                    },
                },
            },
        )

    client = AsyncTypeSafeClient(api_key="test", transport=httpx2.MockTransport(handler))
    jev = JevProvider(client=client)
    decision = await jev.decide(
        {"text": "hi"},
        {
            "yes": Noul(instructions="Is text a greeting?"),
            "pick": Choice(criteria={"a": None, "b": None}),
            "rate": Score(criteria=["low", "mid", "high"]),
        },
    )
    await jev.aclose()

    assert len(seen) == 1
    assert seen[0].url.path == "/v1/systemone"
    body = json.loads(seen[0].content)
    assert set(body["questions"]) == {"yes", "pick", "rate"}
    assert decision.nouls["yes"].p == 0.93
    assert decision.choices["pick"].p("b") == 0.9
    assert decision.scores["rate"].normalized == 0.75
    assert decision.input_tokens == 321
