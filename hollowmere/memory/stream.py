"""The memory stream: every observation, conversation, plan and thought an agent has.

Retrieval follows the paper: score = recency + importance + relevance, each
min-max normalised over the candidates. Differences from the original code:
recency decays with *game time* since last access (the original decayed by
position in a sorted list, so a burst of trivial observations aged everything
else), and all vectors live in one numpy matrix so retrieval is a single
matrix-vector product instead of a Python loop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from typing import Iterable, Literal

import numpy as np

from hollowmere.memory.embed import Embedder
from hollowmere.memory.store import Store

Kind = Literal["observation", "chat", "thought", "reflection", "plan"]

RECENCY_HALF_LIFE_H = 8.0
W_RECENCY, W_IMPORTANCE, W_RELEVANCE = 1.0, 1.0, 1.5


@dataclass
class Memory:
    id: int
    kind: Kind
    text: str
    created: datetime
    last_accessed: datetime
    importance: int
    evidence: list[int] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"id": self.id, "kind": self.kind, "text": self.text, "created": self.created.isoformat(),
                "importance": self.importance, "evidence": self.evidence}


class MemoryStream:
    def __init__(self, agent_id: str, store: Store, embedder: Embedder):
        self.agent_id = agent_id
        self.store = store
        self.embedder = embedder
        self.items: list[Memory] = []
        self._vecs: np.ndarray | None = None  # (n, dim), row i matches items[i]
        self._index: dict[int, int] = {}

    # ----------------------------------------------------------------- loading

    async def load(self) -> None:
        rows = self.store.load_memories(self.agent_id)
        self.items = [
            Memory(r["id"], r["kind"], r["text"], datetime.fromisoformat(r["created"]),
                   datetime.fromisoformat(r["last_accessed"]), r["importance"], json.loads(r["evidence"]))
            for r in rows
        ]
        self._index = {m.id: i for i, m in enumerate(self.items)}
        vecs = [np.frombuffer(r["embedding"], dtype=np.float32) if r["embedding"] else None for r in rows]
        dim = (await self.embedder.embed(["x"])).shape[1]
        stale = [i for i, v in enumerate(vecs) if v is None or v.shape[0] != dim]
        if stale:  # embedder changed since the save: re-embed those rows
            fresh = await self.embedder.embed([self.items[i].text for i in stale])
            for j, i in enumerate(stale):
                vecs[i] = fresh[j]
            self.store.update_embeddings([(self.items[i].id, vecs[i]) for i in stale])
        self._vecs = np.stack(vecs) if vecs else None

    # ------------------------------------------------------------------ adding

    async def add(self, kind: Kind, text: str, now: datetime, importance: int,
                  evidence: list[int] | None = None) -> Memory:
        importance = max(1, min(10, int(importance)))
        vec = (await self.embedder.embed([text]))[0]
        mid = self.store.insert_memory(self.agent_id, kind, text, now, importance, evidence or [], vec)
        mem = Memory(mid, kind, text, now, now, importance, evidence or [])
        self._index[mid] = len(self.items)
        self.items.append(mem)
        self._vecs = vec[None, :] if self._vecs is None else np.vstack([self._vecs, vec])
        return mem

    def get(self, mid: int) -> Memory | None:
        i = self._index.get(mid)
        return self.items[i] if i is not None else None

    def recent(self, n: int, kinds: set[str] | None = None) -> list[Memory]:
        out = [m for m in reversed(self.items) if kinds is None or m.kind in kinds]
        return list(reversed(out[:n]))

    def seen_recently(self, text: str, now: datetime, minutes: float) -> bool:
        for m in reversed(self.items):
            if (now - m.created).total_seconds() > minutes * 60:
                return False
            if m.text == text:
                return True
        return False

    # --------------------------------------------------------------- retrieval

    async def retrieve(self, queries: list[str], now: datetime, k: int = 12,
                       exclude_kinds: set[str] | None = None) -> list[Memory]:
        """Top-k memories for the queries, merged and returned in chronological order."""
        if not self.items or self._vecs is None:
            return []
        # Embed first: over the network this await lets other coroutines add memories, and
        # everything below must see one consistent list of items and vectors.
        qvecs = await self.embedder.embed(queries)
        items, vecs = self.items[:], self._vecs
        mask = np.array([exclude_kinds is None or m.kind not in exclude_kinds for m in items])
        if not mask.any():
            return []
        hours = np.array([(now - m.last_accessed).total_seconds() / 3600 for m in items])
        recency = _norm(0.5 ** (hours / RECENCY_HALF_LIFE_H), mask)
        importance = _norm(np.array([m.importance for m in items], dtype=float), mask)
        chosen: dict[int, float] = {}
        for qv in qvecs:
            relevance = _norm(vecs @ qv, mask)
            score = W_RECENCY * recency + W_IMPORTANCE * importance + W_RELEVANCE * relevance
            score[~mask] = -np.inf
            for i in np.argsort(-score)[:k]:
                if np.isfinite(score[i]):
                    chosen[int(i)] = max(chosen.get(int(i), -np.inf), float(score[i]))
        top = sorted(chosen, key=chosen.get, reverse=True)[: max(k, len(queries) * k // 2)]
        result = sorted((items[i] for i in top), key=lambda m: m.id)
        for m in result:
            m.last_accessed = now
        self.store.touch_memories([m.id for m in result], now)
        return result


def repeats(text: str, earlier: Iterable[str]) -> bool:
    """True if the text is (nearly) one of the earlier ones."""
    norm = " ".join(text.lower().split())
    for e in earlier:
        e = " ".join(e.lower().split())
        if SequenceMatcher(None, norm, e).ratio() > 0.85 or (len(norm) > 30 and norm in e):
            return True  # a near copy, or one sentence lifted from an earlier text
    return False


def _norm(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    vals = x[mask]
    lo, hi = float(vals.min()), float(vals.max())
    if hi - lo < 1e-9:
        return np.full_like(x, 0.5, dtype=float)
    return (x - lo) / (hi - lo)
