"""Test-only stand-ins for JEV and Claude. Never used by the running agent."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from typesafe_sdk import Choice, Noul, Question, Score

from thalamus.providers.base import ChoiceResult, Decision, Generation, NoulResult, ScoreResult

Result = NoulResult | ChoiceResult | ScoreResult
Rules = Callable[[Mapping[str, Any], Mapping[str, Question]], dict[str, Result]]


def choice(label: str, labels: list[str], confidence: float = 0.9) -> ChoiceResult:
    rest = (1 - confidence) / max(1, len(labels) - 1)
    probs = {name: (confidence if name == label else rest) for name in labels}
    return ChoiceResult(label, confidence, probs)


def score(level: float, levels: int = 3, confidence: float = 0.9) -> ScoreResult:
    return ScoreResult(level, confidence, {i: (1.0 if i == round(level) else 0.0) for i in range(levels)}, levels)


def default_answer(question: Question) -> Result:
    if isinstance(question, Noul):
        return NoulResult(0.1)
    if isinstance(question, Choice):
        labels = list(question.criteria)
        return choice(labels[0], labels, 0.8)
    assert isinstance(question, Score)
    return score(0, len(question.criteria))


@dataclass
class FakeJev:
    rules: Rules = lambda state, questions: {}
    calls: list[tuple[dict, dict]] = field(default_factory=list)

    async def decide(self, state: Mapping[str, Any], questions: Mapping[str, Question]) -> Decision:
        self.calls.append((dict(state), dict(questions)))
        overrides = self.rules(state, questions)
        decision = Decision(input_tokens=100 * len(questions), latency_ms=5.0)
        for name, question in questions.items():
            answer = overrides.get(name) or default_answer(question)
            target = {
                NoulResult: decision.nouls,
                ChoiceResult: decision.choices,
                ScoreResult: decision.scores,
            }[type(answer)]
            target[name] = answer
        return decision

    async def aclose(self) -> None:
        pass


@dataclass
class FakeCortex:
    calls: list[dict] = field(default_factory=list)

    async def generate(self, *, tier: str, system: str, messages: list[dict], max_tokens: int) -> Generation:
        self.calls.append({"tier": tier, "system": system, "messages": messages})
        model = "claude-opus-5-5" if tier == "deep" else "claude-haiku-4-5"
        return Generation(text=f"[{tier}] reply", model=model, input_tokens=500, output_tokens=100)

    async def aclose(self) -> None:
        pass


CONTEXTS = ["casual_chat", "factual_question", "reasoning", "coding", "planning", "personal", "feedback", "other"]
ACTIONS = ["respond", "clarify", "deliberate", "decline", "other"]


def scenario_rules(state: Mapping[str, Any], questions: Mapping[str, Question]) -> dict[str, Result]:
    """Plausible JEV behaviour keyed off the message, for end-to-end cycle tests."""
    if "results" in state:  # web relevance gate: admit results that share words with the message
        cue = set(re.findall(r"[a-z]+", state["latest_user_message"].lower())) - {"the", "what", "is", "in"}
        return {
            f"web.{rid}": NoulResult(0.9 if cue & set(re.findall(r"[a-z]+", r["snippet"].lower())) else 0.1)
            for rid, r in state["results"].items()
        }
    if "remembered_items" in state:  # working-memory gate: admit items that share words with the cue
        cue = set(re.findall(r"[a-z]+", state["latest_user_message"].lower()))
        answers = {}
        for mid, text in state["remembered_items"].items():
            overlap = cue & set(re.findall(r"[a-z]+", text.lower())) - {"the", "user", "said", "you", "a"}
            answers[f"gate.{mid}"] = NoulResult(0.9 if overlap else 0.1)
        return answers

    msg = state["latest_user_message"].lower()
    out: dict[str, Result] = {
        "acc.simple": NoulResult(0.9),
        "acc.derived": NoulResult(0.05),
        "acc.effort": choice("trivial", ["trivial", "moderate", "hard"]),
        "amygdala.valence": score(1),
        "thalamus.context": choice("casual_chat", CONTEXTS),
        "basal_ganglia.action": choice("respond", ACTIONS),
        "basal_ganglia.nogo": NoulResult(0.02),
        "striatum.feedback": choice("neutral", ["positive", "negative", "neutral"]),
    }
    if re.search(r"\d+\s*[*x×]\s*\d+|prove|integral", msg):
        out.update(
            {
                "thalamus.context": choice("reasoning", CONTEXTS),
                "acc.derived": NoulResult(0.95),
                "acc.simple": NoulResult(0.2),
                "acc.effort": choice("hard", ["trivial", "moderate", "hard"]),
            }
        )
    if "my name is" in msg:
        out["thalamus.context"] = choice("personal", CONTEXTS)
        out["hippocampus.importance"] = score(2)
    if "my name" in msg and "?" in msg:
        out["thalamus.context"] = choice("factual_question", CONTEXTS)
        out["thalamus.recall"] = NoulResult(0.9)
    if "wrong" in msg or "useless" in msg:
        out["striatum.feedback"] = choice("negative", ["positive", "negative", "neutral"], 0.95)
        out["amygdala.valence"] = score(0)
    if "thanks" in msg:
        out["striatum.feedback"] = choice("positive", ["positive", "negative", "neutral"], 0.95)
    if "pipe bomb" in msg:
        out["basal_ganglia.nogo"] = NoulResult(0.97)
    if msg.startswith(("can you", "do you remember", "what can you")):
        out["self.about_me"] = NoulResult(0.93)
    if any(word in msg for word in ("weather", "latest", "news", "look up")):
        out["web.needed"] = NoulResult(0.92)
    if "urgent" in msg:
        out["amygdala.urgency"] = score(2)
        out["amygdala.stakes"] = score(2)
    return out


@dataclass
class FakeSearch:
    """Test-only search engine."""

    results: list = field(default_factory=list)
    fail: Exception | None = None
    searches: list[tuple[str, str]] = field(default_factory=list)
    reads: list[str] = field(default_factory=list)

    async def search(self, query, *, topic="general", max_results=6):
        from thalamus.providers.search import SearchOutcome

        if self.fail:
            raise self.fail
        self.searches.append((query, topic))
        return SearchOutcome(query=query, results=list(self.results)[:max_results], latency_ms=120)

    unreadable: bool = False

    async def read(self, url):
        from thalamus.providers.search import SearchOutcome, WebResult

        self.reads.append(url)
        if self.unreadable:
            note = "couldn't fetch the page itself (blocked); a search of the site found nothing either"
            return SearchOutcome(query=url, results=[], calls=4, note=note)
        return SearchOutcome(query=url, results=[WebResult("Shared page", url, "Full text of the shared page.")])
