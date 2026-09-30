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

    def record(self, model: str, input_tokens: int, output_tokens: int = 0) -> float:
        key = "jev" if model.startswith("jev") else model
        price_in, price_out = self.budget.prices.get(key, (0.0, 0.0))
        cost = (input_tokens * price_in + output_tokens * price_out) / 1_000_000
        self.spent_usd += cost
        self.calls[key] = self.calls.get(key, 0) + 1
        return cost

    def record_usd(self, name: str, usd: float) -> float:
        """For services priced per call rather than per token (e.g. web search)."""
        self.spent_usd += usd
        self.calls[name] = self.calls.get(name, 0) + 1
        return usd

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
