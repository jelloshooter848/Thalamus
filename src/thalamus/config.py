"""Configuration: model choices, thresholds, budgets and API-key checks.

JEV is not optional. Every System-1 region runs on it, and there is deliberately no
stand-in, so a missing TypeSafe key stops THALAMUS before it thinks at all.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

JEV_KEY_ENV = "TYPESAFE_API_KEY"


class MissingJevKeyError(RuntimeError):
    def __init__(self) -> None:
        super().__init__(
            f"THALAMUS requires a TypeSafe API key: set {JEV_KEY_ENV} (see .env.example). "
            "JEV powers the thalamus, amygdala, basal ganglia and anterior cingulate; "
            "there is no fallback model."
        )


@dataclass
class Models:
    jev: str = "jev-latest"
    # Broca's fast path: short, well-grounded replies.
    fast: str = "claude-haiku-4-5"
    # Prefrontal cortex: slow, deliberate System-2 reasoning.
    deep: str = "claude-opus-5-5"
    deep_effort: str = "high"


@dataclass
class Thresholds:
    # ACC: escalate to the prefrontal cortex when value of control exceeds its cost by this margin.
    escalate_margin: float = 0.1
    # "Derived" answers (math, code, logic) are where JEV and fast models fail; escalate above this.
    derived: float = 0.7
    # Two System-1 framings disagreeing by more than this counts as conflict.
    conflict: float = 0.5
    # Basal ganglia NoGo veto (high bar: a wrong refusal is costly too).
    nogo: float = 0.85
    # Minimum Choice confidence before the basal ganglia act on JEV's selected action.
    action_confidence: float = 0.55
    # Hippocampal input gate into working memory.
    memory_gate: float = 0.5


@dataclass
class Budget:
    session_usd: float = 1.00
    # Prices per million tokens: (input, output).
    prices: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "jev": (0.042, 0.0),
            "claude-haiku-4-5": (1.0, 5.0),
            "claude-sonnet-5": (2.0, 10.0),
            "claude-sonnet-5-5": (2.0, 10.0),
            "claude-opus-5": (5.0, 25.0),
            "claude-opus-5-5": (4.0, 20.0),
            "claude-opus-4-8": (5.0, 25.0),
            "claude-fable-5": (10.0, 50.0),
            "claude-fable-5-1": (10.0, 50.0),
        }
    )


@dataclass
class Memory:
    path: str = "~/.thalamus/memory.db"
    recall_k: int = 5
    # Hours for an unrehearsed, unimportant memory to fall to ~37% retention.
    base_stability_hours: float = 72.0


@dataclass
class Web:
    enabled: bool = True  # also needs TAVILY_API_KEY; without it THALAMUS stays offline
    threshold: float = 0.6  # JEV's P(needs outside information) required to search
    max_results: int = 6
    admit: int = 4  # most results allowed into awareness
    depth: str = "basic"  # Tavily search depth: basic (1 credit) or advanced (2 credits)
    price_per_call_usd: float = 0.008


@dataclass
class Remote:
    port: int = 8765  # fixed, so the phone's home-screen icon keeps working
    session_days: float = 90.0  # how long a phone stays logged in
    keep_awake: bool = True  # stop the PC from sleeping while phone access is on
    allow_lan: bool = False  # also accept devices on the home network (Tailscale is always allowed)


@dataclass
class Settings:
    models: Models = field(default_factory=Models)
    thresholds: Thresholds = field(default_factory=Thresholds)
    budget: Budget = field(default_factory=Budget)
    memory: Memory = field(default_factory=Memory)
    web: Web = field(default_factory=Web)
    remote: Remote = field(default_factory=Remote)
    workspace_capacity: int = 7
    history_turns: int = 6
    # After this long without a message, the next one starts a new conversation.
    conversation_idle_minutes: float = 30.0

    @property
    def memory_path(self) -> Path:
        return Path(self.memory.path).expanduser()


def _merge(target: Any, values: dict[str, Any]) -> None:
    known = {f.name for f in fields(target)}
    for key, value in values.items():
        if key not in known:
            raise ValueError(f"Unknown config key: {key}")
        current = getattr(target, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge(current, value)
        else:
            setattr(target, key, value)


def load_dotenv(path: Path = Path(".env")) -> None:
    """Load KEY=VALUE lines from .env without overriding the real environment."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip("'\"")
        if value:
            os.environ.setdefault(key.strip(), value)


def save_env(values: dict[str, str], path: Path = Path(".env")) -> None:
    """Set KEY=VALUE pairs in .env (updating in place, keeping other lines) and in this process."""
    lines = path.read_text().splitlines() if path.is_file() else []
    pending = {key: value.strip() for key, value in values.items()}
    for i, line in enumerate(lines):
        key = line.split("=", 1)[0].strip()
        if "=" in line and not line.lstrip().startswith("#") and key in pending:
            lines[i] = f"{key}={pending.pop(key)}"
    lines += [f"{key}={value}" for key, value in pending.items()]
    path.write_text("\n".join(lines) + "\n")
    for key, value in values.items():
        if value.strip():
            os.environ[key] = value.strip()
        else:
            os.environ.pop(key, None)


def load_settings(path: Path | None = None) -> Settings:
    settings = Settings()
    path = path or Path("thalamus.toml")
    if path.is_file():
        _merge(settings, tomllib.loads(path.read_text()))
    return settings


def require_jev_key() -> str:
    key = os.environ.get(JEV_KEY_ENV, "").strip()
    if not key:
        raise MissingJevKeyError()
    return key
