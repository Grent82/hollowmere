"""A mind's own conclusions must not crowd out what it actually saw and heard.

Measured on a real run: reflections carry importance 8-10 against 2-4 for observations, so retrieval
returned ~10 reflections and 0 observations for every query, and reflections built on reflections
drifted into beliefs nothing in the world supported ("Silas hunts my burrow") while the observations that
contradicted them never reached the prompt."""

from datetime import datetime, timedelta

from hollowmere.config import Config
from hollowmere.llm.fake import FakeLLM
from hollowmere.memory.embed import HashingEmbedder
from hollowmere.memory.stream import MemoryStream
from hollowmere.memory.store import Store
from hollowmere.sim.setup import build_engine

NOW = datetime(1347, 11, 11, 12)
OWN = {"reflection", "thought"}


async def _stream(reflections: int, observations: int, thoughts: int = 0):
    mem = MemoryStream("t", Store(":memory:"), HashingEmbedder(256))
    for i in range(reflections):
        await mem.add("reflection", f"Silas is hunting my burrow, I am sure of it {i}", NOW - timedelta(hours=i), 10)
    for i in range(thoughts):
        await mem.add("thought", f"Silas is hunting my burrow, I fear {i}", NOW - timedelta(hours=i), 5)
    for i in range(observations):
        await mem.add("observation", f"Silas is at the graveyard digging a grave {i}", NOW - timedelta(hours=i), 2)
    return mem


async def test_reflections_cannot_crowd_out_what_was_seen():
    mem = await _stream(reflections=20, observations=20)
    got = await mem.retrieve(["Silas hunting"], NOW, k=8)
    kinds = [m.kind for m in got]
    assert len(got) == 8
    assert kinds.count("reflection") <= 3 and kinds.count("observation") >= 5


async def test_thoughts_count_as_own_conclusions_and_share_the_quota():
    mem = await _stream(reflections=10, observations=20, thoughts=10)
    got = await mem.retrieve(["Silas hunting"], NOW, k=8)
    assert sum(m.kind in OWN for m in got) <= 3


async def test_a_short_pool_is_filled_from_the_other_one():
    mem = await _stream(reflections=20, observations=2)
    got = await mem.retrieve(["Silas"], NOW, k=8)
    assert len(got) == 8  # nobody is starved of memories because the quota was not met
    assert sum(m.kind == "observation" for m in got) == 2


async def test_with_no_conclusions_at_all_everything_comes_from_what_was_seen():
    mem = await _stream(reflections=0, observations=12)
    assert len(await mem.retrieve(["Silas"], NOW, k=8)) == 8


async def test_conclusions_still_compete_with_each_other_on_importance():
    mem = MemoryStream("t", Store(":memory:"), HashingEmbedder(256))
    await mem.add("reflection", "the well water is foul", NOW, 3)
    deadly = await mem.add("reflection", "the well water is foul", NOW, 9)  # as relevant, but weighs more
    await mem.add("observation", "a crow on the wall", NOW, 2)
    got = await mem.retrieve(["the well water"], NOW, k=2)
    assert deadly.id in [m.id for m in got]


async def test_retrieving_a_conclusion_does_not_refresh_it():
    """Refreshed on every recall, a conclusion stays 'recent' for ever and keeps coming back."""
    mem = await _stream(reflections=2, observations=2)
    before = {m.id: m.last_accessed for m in mem.items}
    later = NOW + timedelta(hours=30)
    got = await mem.retrieve(["Silas"], later, k=4)
    assert got
    for m in got:
        if m.kind in OWN:
            assert m.last_accessed == before[m.id], m.kind
        else:
            assert m.last_accessed == later


async def _engine(tmp_path):
    cfg = Config()
    cfg.sim.save_path = str(tmp_path / "run.db")
    return await build_engine(cfg, llm=FakeLLM(latency=0), embedder=HashingEmbedder(128))


async def test_only_what_was_experienced_brings_the_next_reflection_closer(tmp_path):
    """The original counts perceived events only: thoughts and reflections never trigger a reflection."""
    engine = await _engine(tmp_path)
    nib = engine.agents["nib"]
    nib.importance_acc = 0
    await nib._remember("thought", "Silas is hunting me", NOW, 8)
    await nib._remember("reflection", "he has abandoned his duty", NOW, 10)
    await nib._remember("plan", "I will hide", NOW, 5)
    assert nib.importance_acc == 0
    await nib.observe("Silas is digging", NOW, 3)
    await nib._remember("chat", "I spoke with Isolde", NOW, 6)
    assert nib.importance_acc == 9


async def test_reflecting_reads_only_what_was_seen_and_heard(tmp_path):
    engine = await _engine(tmp_path)
    nib = engine.agents["nib"]
    for i in range(8):
        await nib.observe(f"Silas is digging {i}", NOW, 3)
    seen = {}
    real_recent, real_retrieve = nib.memory.recent, nib.memory.retrieve

    def recent(n, kinds=None):
        seen["recent"] = kinds
        return real_recent(n, kinds=kinds)

    async def retrieve(queries, now, k=12, exclude_kinds=None):
        seen["excluded"] = exclude_kinds
        return await real_retrieve(queries, now, k, exclude_kinds)

    nib.memory.recent, nib.memory.retrieve = recent, retrieve
    await nib.reflect(engine.ctx, NOW)
    assert seen["recent"] == {"observation", "chat"}
    assert {"reflection", "thought"} <= set(seen["excluded"])
