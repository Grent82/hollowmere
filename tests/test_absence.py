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
