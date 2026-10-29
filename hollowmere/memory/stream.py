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
# What a mind concluded itself, as against what it saw and heard. Conclusions carry importance 8-10 against 2-4
# for sights, so in one pool they filled every retrieval (10 reflections, 0 observations) and, being built from
# one another, drifted into beliefs the world never supported. The original draws on both in one pool too, but
# with k=30 and a handful of reflections a day; with a k of 6-12 the mix has to be kept.
OWN_KINDS = frozenset({"reflection", "thought"})
OWN_SHARE = 0.4  # at most this share of a retrieval may be the mind's own conclusions


@dataclass
class Memory:
    id: int
    kind: Kind
    text: str
    created: datetime
    last_accessed: datetime
    importance: int
    evidence: list[int] = field(default_factory=list)
    source: str = ""  # "saw", "lived", "told:<id>", "word", "own" or "divine"; empty for memories from before it was kept
    about: list[str] = field(default_factory=list)  # whom it is about, as agent ids

    def as_dict(self) -> dict:
        return {"id": self.id, "kind": self.kind, "text": self.text, "created": self.created.isoformat(),
                "importance": self.importance, "evidence": self.evidence, "source": self.source,
                "about": self.about}


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
                   datetime.fromisoformat(r["last_accessed"]), r["importance"], json.loads(r["evidence"]),
                   r["source"], json.loads(r["about"]))
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
                  evidence: list[int] | None = None, source: str = "", about: list[str] | None = None) -> Memory:
        importance = max(1, min(10, int(importance)))
        source = source or ("own" if kind in OWN_KINDS else "")  # a thought or a reflection is one's own conclusion
        vec = (await self.embedder.embed([text]))[0]
        mid = self.store.insert_memory(self.agent_id, kind, text, now, importance, evidence or [], vec, source,
                                       about or [])
        mem = Memory(mid, kind, text, now, now, importance, evidence or [], source, list(about or []))
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
        recency_raw = 0.5 ** (hours / RECENCY_HALF_LIFE_H)
        importance_raw = np.array([m.importance for m in items], dtype=float)
        own = np.array([m.kind in OWN_KINDS for m in items])
        seen_pool, own_pool = mask & ~own, mask & own
        pools = [p for p in (seen_pool, own_pool) if p.any()]
        # each pool is normalised on its own: conclusions compete with conclusions, sights with sights
        recency = {id(p): _norm(recency_raw, p) for p in pools}
        importance = {id(p): _norm(importance_raw, p) for p in pools}
        chosen: dict[int, float] = {}
        for qv in qvecs:
            sim = vecs @ qv
            ranked = {}
            for p in pools:
                score = (W_RECENCY * recency[id(p)] + W_IMPORTANCE * importance[id(p)]
                         + W_RELEVANCE * _norm(sim, p))
                score[~p] = -np.inf
                ranked[id(p)] = [int(i) for i in np.argsort(-score) if p[i]], score
            for i in self._quota(k, seen_pool, own_pool, ranked):
                chosen[i] = max(chosen.get(i, -np.inf), float(ranked[id(seen_pool if not own[i] else own_pool)][1][i]))
        limit = max(k, len(queries) * k // 2)
        top = self._cap(chosen, own, limit)
        result = sorted((items[i] for i in top), key=lambda m: m.id)
        refreshed = [m for m in result if m.kind not in OWN_KINDS]  # a recalled conclusion must not stay "fresh"
        for m in refreshed:
            m.last_accessed = now
        self.store.touch_memories([m.id for m in refreshed], now)
        return result

    @staticmethod
    def _quota(k: int, seen_pool: np.ndarray, own_pool: np.ndarray, ranked: dict) -> list[int]:
        """The top k indices: at most OWN_SHARE of them conclusions, a short pool made up from the other."""
        own_n = min(int(k * OWN_SHARE), int(own_pool.sum()))
        seen_n = min(k - own_n, int(seen_pool.sum()))
        own_n = min(k - seen_n, int(own_pool.sum()))
        out: list[int] = []
        if seen_pool.any():
            out += ranked[id(seen_pool)][0][:seen_n]
        if own_pool.any():
            out += ranked[id(own_pool)][0][:own_n]
        return out

    @staticmethod
    def _cap(chosen: dict[int, float], own: np.ndarray, limit: int) -> list[int]:
        """Merge the picks of several queries down to `limit`, keeping the same share of conclusions."""
        by_score = sorted(chosen, key=chosen.get, reverse=True)
        mine = [i for i in by_score if own[i]]
        seen = [i for i in by_score if not own[i]]
        own_n = min(int(limit * OWN_SHARE), len(mine))
        seen_n = min(limit - own_n, len(seen))
        own_n = min(limit - seen_n, len(mine))
        return seen[:seen_n] + mine[:own_n]


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
