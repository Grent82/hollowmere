"""Assemble a runnable simulation from config."""

from __future__ import annotations

import logging

from hollowmere.agents.agent import Agent, Ctx
from hollowmere.agents.persona import Persona, load_personas
from hollowmere.config import Config
from hollowmere.llm.client import LLM, OpenAICompatLLM
from hollowmere.memory.embed import Embedder, HashingEmbedder, make_embedder
from hollowmere.memory.store import Store
from hollowmere.memory.stream import MemoryStream
from hollowmere.sim.engine import Engine
from hollowmere.world.town import build_town

log = logging.getLogger(__name__)


async def seed_memories(agent: Agent, persona: Persona, names: dict[str, str], engine_now) -> None:
    """The original seeded memories by splitting a semicolon list; here each seed is a full sentence."""
    await agent.memory.add("thought", persona.background, engine_now, 6)
    await agent.memory.add("thought", persona.currently, engine_now, 8)
    for other, rel in persona.relationships.items():
        await agent.memory.add("thought", f"{names.get(other, other)} is {rel}.", engine_now, 5)


async def build_engine(cfg: Config, *, fake: bool = False, fresh: bool = False,
                       llm: LLM | None = None, embedder: Embedder | None = None) -> Engine:
    if llm is None:
        if fake:
            from hollowmere.llm.fake import FakeLLM
            llm = FakeLLM()
        else:
            llm = OpenAICompatLLM(cfg.llm)
    if embedder is None:
        embedder = HashingEmbedder(cfg.embeddings.hashing_dim) if fake else await make_embedder(cfg.embeddings, cfg.llm)

    save_path = cfg.sim.save_path.replace(".db", "-fake.db") if fake and cfg.sim.save_path != ":memory:" else cfg.sim.save_path
    if fresh and save_path != ":memory:":
        from pathlib import Path
        for suffix in ("", "-wal", "-shm"):
            Path(save_path + suffix).unlink(missing_ok=True)
    store = Store(save_path)
    if store.match_embedder(embedder.signature):
        log.info("memories were embedded by another model; re-embedding them with %s", embedder.signature)

    world = build_town()
    personas = load_personas()
    if cfg.sim.agents:
        personas = [p for p in personas if p.id in cfg.sim.agents]
    names = {p.id: p.name for p in personas}
    agents = [Agent(p, MemoryStream(p.id, store, embedder), world) for p in personas]
    known_as = {p.id: p.known_as for p in personas if p.known_as}
    engine = Engine(cfg.sim, world, agents, Ctx(llm, world, names, known_as), store)

    resumed = store.has_memories() and engine.load()
    for a in agents:
        await a.memory.load()
    if not resumed:
        for a, p in zip(agents, personas):
            await seed_memories(a, p, names, engine.now)
        engine.save()
    log.info("%s simulation at %s with %d agents", "resumed" if resumed else "new", engine.now, len(agents))
    return engine
