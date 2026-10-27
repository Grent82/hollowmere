"""Text embeddings for memory relevance.

`ServerEmbedder` uses the LLM server's /embeddings endpoint. `HashingEmbedder`
is a dependency-free fallback (signed feature hashing of word unigrams and
bigrams) that is crude but deterministic and instant, so the simulation still
works when the server has no embedding model loaded.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from collections import OrderedDict
from typing import Protocol

import httpx
import numpy as np

from hollowmere.config import EmbeddingConfig, LLMConfig

log = logging.getLogger(__name__)

EMBED_BATCH = 64
EMBED_CACHE = 5000
EMBED_ATTEMPTS = 4

_WORD_RE = re.compile(r"[a-z0-9']+")
_STOP = frozenset(
    "a an the and or of to in on at for with by from is are was were be been it its this that "
    "he she they them his her their i me my you your we our as into about over is am do does did "
    "has have had not no so but if then than there here".split()
)


class Embedder(Protocol):
    signature: str

    async def embed(self, texts: list[str]) -> np.ndarray: ...


def _tokens(text: str) -> list[str]:
    words = [w[:-2] if w.endswith("'s") else w.strip("'") for w in _WORD_RE.findall(text.lower())]
    return [w[:-1] if len(w) > 4 and w.endswith("s") else w for w in words if w not in _STOP]


class HashingEmbedder:
    def __init__(self, dim: int = 768):
        self.dim = dim
        self.signature = f"hashing-{dim}"

    def _bucket(self, feature: str) -> tuple[int, float]:
        h = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "little")
        return h % self.dim, (1.0 if (h >> 63) & 1 else -1.0)

    def embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        toks = _tokens(text)
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        for f in feats:
            i, sign = self._bucket(f)
            vec[i] += sign * (0.6 if "_" in f else 1.0)
        n = np.linalg.norm(vec)
        return vec / n if n else vec

    async def embed(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.embed_one(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)


class ServerEmbedder:
    def __init__(self, llm_cfg: LLMConfig, model: str):
        self.model = model
        self.signature = f"server-{model}"
        self._http = httpx.AsyncClient(
            base_url=llm_cfg.base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {llm_cfg.key}"},
            timeout=60,
        )
        # Observations repeat ("X is at the forge, busy: ..."), and over a network every
        # request costs a round trip, so identical texts are embedded once.
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

    async def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), np.float32)
        missing =list(dict.fromkeys(t for t in texts if t not in self._cache))
        for i in range(0, len(missing), EMBED_BATCH):  # re-embedding a whole save at once is too big for one request
            chunk = missing[i:i + EMBED_BATCH]
            rows = await self._request(chunk)
            arr = np.asarray([r["embedding"] for r in rows], dtype=np.float32)
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            for text, vec in zip(chunk, arr / np.where(norms == 0, 1, norms)):
                self._cache[text] = vec
        for t in texts:
            self._cache.move_to_end(t)
        while len(self._cache) > EMBED_CACHE:
            self._cache.popitem(last=False)
        return np.stack([self._cache[t] for t in texts])

    async def _request(self, chunk: list[str]) -> list[dict]:
        """One /embeddings call, retried: a network hiccup must not take a memory (or the tick) with it."""
        for attempt in range(EMBED_ATTEMPTS):
            try:
                resp = await self._http.post("/embeddings", json={"model": self.model, "input": chunk})
                resp.raise_for_status()
                rows = sorted(resp.json()["data"], key=lambda r: r["index"])
                if len(rows) == len(chunk):
                    return rows
                error: Exception = ValueError(f"{len(rows)} embeddings for {len(chunk)} texts")
            except (httpx.HTTPError, ValueError, KeyError, TypeError) as e:
                error = e
                status = e.response.status_code if isinstance(e, httpx.HTTPStatusError) else 0
                if 400 <= status < 500 and status not in (408, 429):  # e.g. 404: the server has no embeddings
                    raise RuntimeError(f"embeddings rejected: {error!r}") from error
            if attempt == EMBED_ATTEMPTS - 1:
                raise RuntimeError(f"embeddings failed after {EMBED_ATTEMPTS} attempts: {error!r}") from error
            log.warning("embedding request failed (%r); retrying", error)
            await asyncio.sleep(2 ** attempt)
        raise AssertionError("unreachable")


async def make_embedder(cfg: EmbeddingConfig, llm_cfg: LLMConfig) -> Embedder:
    if cfg.provider == "hashing":
        return HashingEmbedder(cfg.hashing_dim)
    server = ServerEmbedder(llm_cfg, cfg.model)
    try:
        await server.embed(["probe"])
        log.info("using server embeddings (%s)", cfg.model)
        return server
    except Exception as e:  # noqa: BLE001 - any failure means "not available"
        if cfg.provider == "server":
            raise
        log.warning("server embeddings unavailable (%s); using hashing embedder", e)
        return HashingEmbedder(cfg.hashing_dim)
