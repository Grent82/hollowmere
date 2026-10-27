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


# ------------------------------------------------------------------ setting out and turning back

from datetime import timedelta  # noqa: E402

from hollowmere.agents.agent import Digest  # noqa: E402
from hollowmere.sim.engine import Conversation  # noqa: E402

EVENING = datetime(1347, 11, 5, 22, 0)


async def _with_baby(engine):
    preg = {"mother_id": "isolde", "father_id": "tobin", "start_day": "1347-11-01", "due_day": "1347-11-05",
            "resolved": False}
    engine.pregnancies.append(preg)
    await engine._give_birth(preg, NOW)
    return next(a for a in engine.agents.values() if "isolde" in a.persona.parents)


def _talk(engine, a, b):
    return Conversation("c1", engine.agents[a], engine.agents[b], "to plan", NOW)


async def test_both_digests_saying_leaving_schedules_a_departure(tmp_path):
    engine = await _engine(tmp_path)
    engine.now = NOW
    engine._settle(_talk(engine, "isolde", "tobin"),
                   {"isolde": Digest(leaving=True), "tobin": Digest(leaving=True)})
    assert [(q["do"], q["ids"], q["at"]) for q in engine._queue] == [
        ("depart", ["isolde", "tobin"], "1347-11-05T22:00:00")]


async def test_one_digest_saying_leaving_does_nothing(tmp_path):
    engine = await _engine(tmp_path)
    engine.now = NOW
    engine._settle(_talk(engine, "isolde", "tobin"), {"isolde": Digest(leaving=True), "tobin": Digest()})
    assert engine._queue == []


async def test_a_departure_takes_the_baby_along_and_hides_everyone(tmp_path):
    engine = await _engine(tmp_path)
    child = await _with_baby(engine)
    engine.now = NOW
    engine.schedule_departure(engine.agents["isolde"], engine.agents["tobin"], NOW)
    await engine._absences(EVENING)
    for who in ("isolde", "tobin", child.id):
        a = engine.agents[who]
        assert a.is_away and a.mode == "away"
        assert datetime.fromisoformat(a.absence["until"]) - EVENING >= timedelta(hours=19)
    assert engine._queue and engine._queue[0]["do"] == "vanish"


async def test_turning_back_writes_the_reason_into_both_memories(tmp_path):
    engine = await _engine(tmp_path)
    engine.now = NOW
    engine.schedule_departure(engine.agents["isolde"], engine.agents["tobin"], NOW)
    await engine._absences(EVENING)
    isolde, tobin = engine.agents["isolde"], engine.agents["tobin"]
    reason = isolde.absence["reason"]
    back = datetime.fromisoformat(isolde.absence["until"])
    await engine._absences(back)
    for me, other in ((isolde, tobin), (tobin, isolde)):
        assert not me.is_away and me.mode == "acting" and me.plan_date is None
        assert (me.x, me.y) == engine.world.target_tile(me.persona.bed)
        assert any(other.first in m.text and reason in m.text and m.importance == 8 for m in me.memory.items)


async def test_a_birth_waits_for_an_away_mother(tmp_path):
    engine = await _engine(tmp_path)
    _send_away(engine.agents["isolde"])
    engine.pregnancies.append({"mother_id": "isolde", "father_id": "tobin", "start_day": "1347-11-01",
                               "due_day": "1347-11-05", "resolved": False})
    engine.now = NOW
    engine._background(NOW)
    assert engine.pregnancies[0]["resolved"] is False and not engine._bg.get("isolde:birth:isolde:1347-11-05")


async def test_a_departure_survives_save_and_load(tmp_path):
    engine = await _engine(tmp_path)
    engine.now = NOW
    engine.schedule_departure(engine.agents["isolde"], engine.agents["tobin"], NOW)
    await engine._absences(EVENING)
    engine.save()
    again = await _engine(tmp_path)
    assert again.agents["tobin"].is_away and again.agents["tobin"].mode == "away"
    assert again._queue == engine._queue
