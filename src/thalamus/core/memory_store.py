"""Persistent substrate for long-term memory and learned values (SQLite + FTS5).

- episodes: append-only episodic traces with time, context and affect (pattern separation), never
  overwritten, with an Ebbinghaus-style retention curve that is reinforced whenever recalled.
- values: the striatal critic's learned value V(context, option) for actions and arbitration paths.
"""

from __future__ import annotations

import math
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY,
    created REAL NOT NULL,
    last_access REAL NOT NULL,
    session TEXT NOT NULL,
    turn INTEGER NOT NULL,
    speaker TEXT NOT NULL,
    text TEXT NOT NULL,
    context TEXT NOT NULL,
    importance REAL NOT NULL,
    valence REAL NOT NULL,
    arousal REAL NOT NULL,
    recall_count INTEGER NOT NULL DEFAULT 0
);
CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(text, content='episodes', content_rowid='id');
CREATE TRIGGER IF NOT EXISTS episodes_ai AFTER INSERT ON episodes BEGIN
    INSERT INTO episodes_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TABLE IF NOT EXISTS vals (
    context TEXT NOT NULL,
    option TEXT NOT NULL,
    value REAL NOT NULL,
    visits INTEGER NOT NULL,
    PRIMARY KEY (context, option)
);
"""

_STOPWORDS = frozenset(
    "the and for are but not you your yours with this that what whats have has had was were will "
    "would could should can cant about from they them their there then than just like into also "
    "how why who when where which does did doing done its it's i'm im me my mine our ours please "
    "tell know".split()
)


@dataclass
class Episode:
    id: int
    created: float
    session: str
    turn: int
    speaker: str
    text: str
    context: str
    importance: float
    valence: float
    arousal: float
    recall_count: int
    score: float = 0.0


def query_terms(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return list(dict.fromkeys(w for w in words if len(w) > 2 and w not in _STOPWORDS))


class MemoryStore:
    def __init__(self, path: Path | str) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(_SCHEMA)

    # ----- episodic memory -------------------------------------------------------------------
    def encode(
        self,
        *,
        session: str,
        turn: int,
        speaker: str,
        text: str,
        context: str,
        importance: float,
        valence: float = 0.0,
        arousal: float = 0.0,
        now: float | None = None,
    ) -> int:
        now = now or time.time()
        cursor = self._db.execute(
            "INSERT INTO episodes (created, last_access, session, turn, speaker, text, context, "
            "importance, valence, arousal) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (now, now, session, turn, speaker, text, context, importance, valence, arousal),
        )
        self._db.commit()
        return int(cursor.lastrowid)

    def recall(
        self,
        cue: str,
        *,
        k: int,
        exclude_session: str | None = None,
        exclude_from_turn: int = 0,
        include_salient: bool = False,
        stability_hours: float = 72.0,
        now: float | None = None,
    ) -> list[Episode]:
        """Retrieve by recency (retention) + importance + relevance to the cue."""
        now = now or time.time()
        relevance: dict[int, float] = {}
        terms = query_terms(cue)
        if terms:
            rows = self._db.execute(
                "SELECT rowid, bm25(episodes_fts) AS rank FROM episodes_fts WHERE episodes_fts MATCH ? "
                "ORDER BY rank LIMIT 50",
                (" OR ".join(f'"{t}"' for t in terms),),
            ).fetchall()
            if rows:
                best, worst = rows[0]["rank"], rows[-1]["rank"]
                spread = (worst - best) or 1.0
                relevance = {r["rowid"]: 1.0 - 0.7 * (r["rank"] - best) / spread for r in rows}
        ids = set(relevance)
        if include_salient:  # pattern completion without a lexical cue: strongest traces surface
            ids |= {
                r["id"]
                for r in self._db.execute(
                    "SELECT id FROM episodes ORDER BY importance DESC, created DESC LIMIT 10"
                )
            }
        if not ids:
            return []

        placeholders = ",".join("?" * len(ids))
        episodes: list[Episode] = []
        for row in self._db.execute(f"SELECT * FROM episodes WHERE id IN ({placeholders})", tuple(ids)):
            if row["session"] == exclude_session and row["turn"] >= exclude_from_turn:
                continue  # already in the conversation window; no need to remember it
            age_hours = max(0.0, (now - row["last_access"]) / 3600)
            stability = stability_hours * (1 + row["recall_count"]) * (1 + 2 * row["importance"])
            retention = math.exp(-age_hours / stability)
            score = 0.3 * retention + 0.3 * row["importance"] + 0.4 * relevance.get(row["id"], 0.0)
            episodes.append(
                Episode(
                    id=row["id"],
                    created=row["created"],
                    session=row["session"],
                    turn=row["turn"],
                    speaker=row["speaker"],
                    text=row["text"],
                    context=row["context"],
                    importance=row["importance"],
                    valence=row["valence"],
                    arousal=row["arousal"],
                    recall_count=row["recall_count"],
                    score=score,
                )
            )
        episodes.sort(key=lambda e: e.score, reverse=True)
        return episodes[:k]

    def reinforce(self, episode_ids: list[int], now: float | None = None) -> None:
        """Recall strengthens a memory and resets its forgetting clock."""
        if not episode_ids:
            return
        now = now or time.time()
        self._db.executemany(
            "UPDATE episodes SET recall_count = recall_count + 1, last_access = ? WHERE id = ?",
            [(now, i) for i in episode_ids],
        )
        self._db.commit()

    def episodes(self, limit: int = 50) -> list[sqlite3.Row]:
        return self._db.execute("SELECT * FROM episodes ORDER BY created DESC LIMIT ?", (limit,)).fetchall()

    def stats(self) -> tuple[int, float | None]:
        """How many episodes are stored, and when the oldest was formed."""
        row = self._db.execute("SELECT COUNT(*) AS n, MIN(created) AS first FROM episodes").fetchone()
        return int(row["n"]), row["first"]

    def clear(self) -> None:
        self._db.executescript("DELETE FROM episodes; DELETE FROM vals;")
        self._db.execute("INSERT INTO episodes_fts(episodes_fts) VALUES ('rebuild')")
        self._db.commit()

    # ----- learned values (striatal critic) --------------------------------------------------
    def value(self, context: str, option: str) -> float:
        row = self._db.execute(
            "SELECT value FROM vals WHERE context = ? AND option = ?", (context, option)
        ).fetchone()
        return float(row["value"]) if row else 0.0

    def update_value(self, context: str, option: str, reward: float, learning_rate: float) -> float:
        """Rescorla-Wagner / TD(0) update. Returns the reward prediction error."""
        old = self.value(context, option)
        rpe = reward - old
        self._db.execute(
            "INSERT INTO vals (context, option, value, visits) VALUES (?, ?, ?, 1) "
            "ON CONFLICT(context, option) DO UPDATE SET value = excluded.value, visits = visits + 1",
            (context, option, old + learning_rate * rpe),
        )
        self._db.commit()
        return rpe

    def values(self) -> list[sqlite3.Row]:
        return self._db.execute("SELECT * FROM vals ORDER BY context, option").fetchall()

    def close(self) -> None:
        self._db.close()
