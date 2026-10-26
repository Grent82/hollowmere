"""What a bystander stores about someone else's activity must be something that could be seen."""

from datetime import datetime

from hollowmere.agents.agent import Task
from hollowmere.config import Config
from hollowmere.llm.fake import FakeLLM
from hollowmere.memory.embed import HashingEmbedder
from hollowmere.sim.setup import build_engine

NOW = datetime(1347, 11, 5, 19, 0)


async def _engine(tmp_path):
    cfg = Config()
    cfg.sim.save_path = str(tmp_path / "run.db")
    return await build_engine(cfg, llm=FakeLLM(latency=0), embedder=HashingEmbedder(128))


def _doing(engine, agent_id, activity, others_at):
    """Put `agent_id` mid-task at (10, 10) with `activity`; `others_at` maps agent id -> distance in tiles."""
    agent = engine.agents[agent_id]
    agent.x, agent.y = 10, 10
    agent.mode = "acting"
    agent.task = Task(NOW, NOW, activity, agent.persona.home, "")
    for other_id, dist in others_at.items():
        engine.agents[other_id].x, engine.agents[other_id].y = 10 + dist, 10
    return agent


async def test_activity_naming_someone_far_away_is_not_stored_as_seen(tmp_path):
    """Regression: 'spots Isolde and moves to her side' was stored as an observation while Isolde was 38
    tiles off, because the text is the planned step, not what happened. A watcher then built a jealous
    reflection on an event that never took place."""
    engine = await _engine(tmp_path)
    tobin = _doing(engine, "tobin", "spots Isolde and moves to her side", {"isolde": 30})
    seen = engine._seen_doing(tobin)
    assert "Isolde" not in seen and "busy" not in seen
    assert seen.startswith("is at ")


async def test_activity_naming_someone_close_is_kept(tmp_path):
    engine = await _engine(tmp_path)
    tobin = _doing(engine, "tobin", "spots Isolde and moves to her side", {"isolde": 3})
    assert "spots Isolde and moves to her side" in engine._seen_doing(tobin)


async def test_activity_with_nobody_named_is_kept_as_before(tmp_path):
    engine = await _engine(tmp_path)
    tobin = _doing(engine, "tobin", "hammers a spearhead at the anvil", {})
    assert engine._seen_doing(tobin) == tobin.activity_phrase(engine.world)


async def test_a_surname_or_title_counts_as_naming_the_person(tmp_path):
    engine = await _engine(tmp_path)
    tobin = _doing(engine, "tobin", "dreads Captain Graves finding him out", {"roderic": 25})
    assert "Graves" not in engine._seen_doing(tobin)


async def test_a_persons_own_name_never_hides_their_activity(tmp_path):
    engine = await _engine(tmp_path)
    tobin = _doing(engine, "tobin", "Tobin sharpens the chisel", {})
    assert "Tobin sharpens the chisel" in engine._seen_doing(tobin)
