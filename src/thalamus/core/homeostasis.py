"""Hypothalamus: interoception of the agent's own resources.

Spending is tracked against a session budget (the set point). As reserves deplete, slow thinking
gets costlier for the anterior cingulate; at exhaustion the hyperdirect STOP fires.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from thalamus.config import Budget


@dataclass
class Homeostasis:
    budget: Budget
    spent_usd: float = 0.0
    calls: dict[str, int] = field(default_factory=dict)
    by_service: dict[str, float] = field(default_factory=dict)  # jev / claude / tavily
    unpriced: set[str] = field(default_factory=set)

    def price(self, model: str) -> tuple[float, float] | None:
        """Per-million-token prices. APIs report versioned names (claude-haiku-4-5-20251001), so
        fall back to the longest configured name the model starts with."""
        if model.startswith("jev"):
            return self.budget.prices.get("jev")
        if model in self.budget.prices:
            return self.budget.prices[model]
        family = max((name for name in self.budget.prices if model.startswith(name)), key=len, default=None)
        return self.budget.prices[family] if family else None

    def record(self, model: str, input_tokens: int, output_tokens: int = 0) -> float:
        prices = self.price(model)
        if prices is None:
            self.unpriced.add(model)
            prices = (0.0, 0.0)
        cost = (input_tokens * prices[0] + output_tokens * prices[1]) / 1_000_000
        service = "jev" if model.startswith("jev") else "claude"
        self._add(service, model, cost)
        return cost

    def record_usd(self, name: str, usd: float) -> float:
        """For services priced per call rather than per token (e.g. web search)."""
        self._add(name, name, usd)
        return usd

    def _add(self, service: str, name: str, usd: float) -> None:
        self.spent_usd += usd
        self.by_service[service] = self.by_service.get(service, 0.0) + usd
        self.calls[name] = self.calls.get(name, 0) + 1

    @property
    def depletion(self) -> float:
        if self.budget.session_usd <= 0:
            return 1.0
        return min(1.0, self.spent_usd / self.budget.session_usd)

    @property
    def exhausted(self) -> bool:
        return self.depletion >= 1.0

    @property
    def deliberation_cost(self) -> float:
        """Baseline cost of recruiting System 2, rising steeply as the budget runs out."""
        return 0.2 + 0.8 * self.depletion**2
