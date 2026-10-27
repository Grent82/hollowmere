"""Absence: villagers who set out to leave and turn back, and villagers held in the cell."""

from datetime import datetime

from hollowmere.config import Config
from hollowmere.llm.fake import FakeLLM
from hollowmere.memory.embed import HashingEmbedder
from hollowmere.sim.setup import build_engine

NOW = datetime(1347, 11, 5, 12)


async def _engine(tmp_path):
    cfg = Config()
    cfg.sim.save_path = str(tmp_path / "run.db")
    return await build_engine(cfg, llm=FakeLLM(latency=0), embedder=HashingEmbedder(128))


async def test_agent_without_absence_is_neither_away_nor_jailed(tmp_path):
    a = (await _engine(tmp_path)).agents["tobin"]
    assert a.absence is None and not a.is_away and not a.is_jailed


async def test_absence_survives_snapshot_and_restore(tmp_path):
    engine = await _engine(tmp_path)
    a, b = engine.agents["tobin"], engine.agents["isolde"]
    a.absence = {"kind": "away", "since": NOW.isoformat(), "until": "1347-11-07T17:00:00", "reason": "wolves"}
    b.restore(a.snapshot() | {"x": 1, "y": 1})
    assert b.absence == a.absence and b.is_away


def test_absence_days_are_configured():
    sim = Config().sim
    assert (sim.absence_min_days, sim.absence_max_days) == (1, 3)
    assert (sim.detention_min_days, sim.detention_max_days, sim.max_jailed) == (1, 3, 2)


def _send_away(agent, days=2):
    agent.absence = {"kind": "away", "since": NOW.isoformat(), "until": "1347-11-07T17:00:00", "reason": "wolves"}
    agent.mode, agent.path, agent.task = "away", [], None


async def test_an_away_agent_is_not_seen_and_does_not_see(tmp_path):
    engine = await _engine(tmp_path)
    tobin, other = engine.agents["tobin"], engine.agents["brannoc"]
    other.x, other.y, tobin.x, tobin.y = 20, 20, 21, 20
    assert engine._visible(other, tobin)
    _send_away(tobin)
    assert not tobin.awake
    assert not engine._visible(other, tobin) and not engine._visible(tobin, other)
    engine.now = NOW
    await engine._perceive(NOW)
    assert not any("Tobin" in m.text for m in other.memory.items if m.kind == "observation")


async def test_an_away_agent_is_not_planned_reviewed_or_sampled(tmp_path):
    engine = await _engine(tmp_path)
    tobin = engine.agents["tobin"]
    _send_away(tobin)
    tobin.importance_acc = 10_000
    engine.now = NOW
    await engine._schedule(NOW)
    engine._background(NOW)
    assert tobin.plan == [] and tobin.mode == "away"
    assert "reflect:tobin" not in engine._bg
    engine._sample(NOW)
    assert not engine.store.db.execute("select 1 from samples where agent_id='tobin'").fetchall()
    assert engine.store.db.execute("select 1 from samples where agent_id='brannoc'").fetchall()
