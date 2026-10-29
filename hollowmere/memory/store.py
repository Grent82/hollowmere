"""SQLite persistence for memories and simulation snapshots.

Replaces the original's tree of JSON files that were rewritten wholesale on
every save. Memories are appended as they happen; agent/world state is a small
JSON snapshot written periodically, so a run can be resumed at any point.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id            INTEGER PRIMARY KEY,
    agent_id      TEXT NOT NULL,
    kind          TEXT NOT NULL,
    text          TEXT NOT NULL,
    created       TEXT NOT NULL,
    last_accessed TEXT NOT NULL,
    importance    INTEGER NOT NULL,
    evidence      TEXT NOT NULL DEFAULT '[]',
    embedding     BLOB
);
CREATE INDEX IF NOT EXISTS memories_agent ON memories(agent_id);
CREATE TABLE IF NOT EXISTS events (
    id     INTEGER PRIMARY KEY,
    time   TEXT NOT NULL,
    kind   TEXT NOT NULL,
    agents TEXT NOT NULL,
    text   TEXT NOT NULL,
    extra  TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS samples (
    time     TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    x        INTEGER NOT NULL,
    y        INTEGER NOT NULL,
    mode     TEXT NOT NULL,
    place    TEXT,
    area     TEXT
);
CREATE TABLE IF NOT EXISTS kv (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class Store:
    def __init__(self, path: str | Path):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        have = {r["name"] for r in self.db.execute("PRAGMA table_info(memories)")}
        for column, ddl in (("source", "TEXT NOT NULL DEFAULT ''"), ("about", "TEXT NOT NULL DEFAULT '[]'")):
            if column not in have:  # a save from before memories said where they came from
                self.db.execute(f"ALTER TABLE memories ADD COLUMN {column} {ddl}")
        self.db.commit()
        self.db.execute("PRAGMA journal_mode=WAL")

    def close(self) -> None:
        self.db.commit()
        self.db.close()

    # ---------------------------------------------------------------- memories

    def insert_memory(self, agent_id: str, kind: str, text: str, created: datetime,
                      importance: int, evidence: list[int], embedding: np.ndarray, source: str = "",
                      about: list[str] | None = None) -> int:
        cur = self.db.execute(
            "INSERT INTO memories(agent_id, kind, text, created, last_accessed, importance, evidence, embedding,"
            " source, about) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (agent_id, kind, text, created.isoformat(), created.isoformat(), importance,
             json.dumps(evidence), embedding.astype(np.float32).tobytes(), source, json.dumps(about or [])),
        )
        self.db.commit()
        return int(cur.lastrowid)

    def touch_memories(self, ids: list[int], when: datetime) -> None:
        self.db.executemany("UPDATE memories SET last_accessed=? WHERE id=?", [(when.isoformat(), i) for i in ids])
        self.db.commit()

    def update_embeddings(self, rows: list[tuple[int, np.ndarray]]) -> None:
        self.db.executemany("UPDATE memories SET embedding=? WHERE id=?",
                            [(e.astype(np.float32).tobytes(), i) for i, e in rows])
        self.db.commit()

    def match_embedder(self, signature: str) -> bool:
        """Forget stored vectors made by a different embedder, so they are rebuilt on load.

        Vectors from two models cannot be compared even when they have the same length.
        Returns True if vectors were dropped."""
        if self.get("embedder") == signature:
            return False
        dropped = self.has_memories()
        self.db.execute("UPDATE memories SET embedding=NULL")
        self.put("embedder", signature)
        return dropped

    def load_memories(self, agent_id: str) -> list[sqlite3.Row]:
        return self.db.execute("SELECT * FROM memories WHERE agent_id=? ORDER BY id", (agent_id,)).fetchall()

    def has_memories(self) -> bool:
        return self.db.execute("SELECT 1 FROM memories LIMIT 1").fetchone() is not None

    # ------------------------------------------------------------ run record
    # Mostly for `hollowmere report` to judge a run after the fact, but events_since() below is
    # also read back by the simulation itself (the town review), unlike the rest of this table.

    def add_event(self, time: datetime, kind: str, agents: list[str], text: str, extra: dict) -> None:
        self.db.execute("INSERT INTO events(time, kind, agents, text, extra) VALUES (?,?,?,?,?)",
                        (time.isoformat(), kind, json.dumps(agents), text, json.dumps(extra, default=str)))
        self.db.commit()

    def events_since(self, after: datetime, kinds: set[str]) -> list[str]:
        """Event text for the given kinds, strictly after `after` - durable across restarts and not
        bounded the way the in-memory UI feed (Engine.events) is."""
        placeholders = ",".join("?" * len(kinds))
        rows = self.db.execute(
            f"SELECT text FROM events WHERE time > ? AND kind IN ({placeholders}) ORDER BY id",
            (after.isoformat(), *kinds)).fetchall()
        return [r["text"] for r in rows]

    def events_of(self, kind: str, after: datetime) -> list[dict]:
        """Events of one kind strictly after `after`, oldest first, with their extra fields."""
        rows = self.db.execute("SELECT time, agents, text, extra FROM events WHERE kind = ? AND time > ? ORDER BY id",
                               (kind, after.isoformat())).fetchall()
        return [{"time": r["time"], "text": r["text"], "agents": json.loads(r["agents"]), **json.loads(r["extra"])}
                for r in rows]

    def chat_log_for(self, agent_id: str, limit: int = 200) -> list[dict]:
        """This agent's chat lines and conversation endings, oldest first, each carrying `convo` so
        the caller can group consecutive lines into one conversation and mark where it ended."""
        rows = self.db.execute(
            "SELECT time, kind, agents, text, extra FROM events WHERE kind IN ('chat','chat_end') "
            "ORDER BY id DESC LIMIT 5000").fetchall()
        out: list[dict] = []
        for r in rows:
            agents = json.loads(r["agents"])
            if agent_id not in agents:
                continue
            extra = json.loads(r["extra"])
            out.append({"time": r["time"], "kind": r["kind"], "text": r["text"],
                        "speaker": extra.get("speaker"), "convo": extra.get("convo"),
                        "with": next((a for a in agents if a != agent_id), None)})
            if len(out) >= limit:
                break
        out.reverse()
        return out

    def prayers_for(self, agent_id: str, limit: int = 10) -> list[dict]:
        """This agent's most recent prayers and the god's answer to each, newest first."""
        rows = self.db.execute(
            "SELECT time, agents, text, extra FROM events WHERE kind = 'prayer' ORDER BY id DESC LIMIT 2000").fetchall()
        out: list[dict] = []
        for r in rows:
            if agent_id not in json.loads(r["agents"]):
                continue
            extra = json.loads(r["extra"])
            out.append({"time": r["time"], "god": extra.get("god"), "form": extra.get("form"),
                        "clarity": extra.get("clarity", "none"), "text": r["text"].split(": ", 1)[-1]})
            if len(out) >= limit:
                break
        return out

    def prayers_to(self, god_id: str, limit: int = 30) -> list[dict]:
        """The latest prayers this god answered and how, oldest first."""
        rows = self.db.execute(
            "SELECT time, agents, text, extra FROM events WHERE kind = 'prayer' ORDER BY id DESC LIMIT 2000").fetchall()
        out: list[dict] = []
        for r in rows:
            extra = json.loads(r["extra"])
            if extra.get("god") != god_id:
                continue
            out.append({"time": r["time"], "agent": json.loads(r["agents"])[0], "form": extra.get("form"),
                        "text": r["text"].split(": ", 1)[-1]})
            if len(out) >= limit:
                break
        return out[::-1]

    def add_samples(self, time: datetime, rows: list[tuple]) -> None:
        """rows: (agent_id, x, y, mode, task place, area under the agent)."""
        self.db.executemany("INSERT INTO samples VALUES (?,?,?,?,?,?,?)", [(time.isoformat(), *r) for r in rows])
        self.db.commit()

    def trim_record(self, after: datetime) -> None:
        """Forget events and samples past `after`: a resumed run replays that time from the last autosave."""
        self.db.execute("DELETE FROM events WHERE time > ?", (after.isoformat(),))
        self.db.execute("DELETE FROM samples WHERE time > ?", (after.isoformat(),))
        self.db.commit()

    # ---------------------------------------------------------------------- kv

    def put(self, key: str, value: object) -> None:
        self.db.execute("INSERT OR REPLACE INTO kv(key, value) VALUES (?, ?)", (key, json.dumps(value)))
        self.db.commit()

    def get(self, key: str, default: object = None) -> object:
        row = self.db.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default
