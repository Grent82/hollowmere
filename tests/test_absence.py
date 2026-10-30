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


async def test_a_departure_takes_the_baby_along_and_hides_everyone(tmp_path):
    engine = await _engine(tmp_path)
    child = await _with_baby(engine)
    engine.now = NOW
    engine.schedule_departure([engine.agents["isolde"], engine.agents["tobin"]], NOW)
    await engine._absences(EVENING)
    for who in ("isolde", "tobin", child.id):
        a = engine.agents[who]
        assert a.is_away and a.mode == "away"
        assert datetime.fromisoformat(a.absence["until"]) - EVENING >= timedelta(hours=19)
    assert engine._queue and engine._queue[0]["do"] == "vanish"


async def test_turning_back_writes_the_reason_into_both_memories(tmp_path):
    engine = await _engine(tmp_path)
    engine.now = NOW
    engine.schedule_departure([engine.agents["isolde"], engine.agents["tobin"]], NOW)
    await engine._absences(EVENING)
    isolde, tobin = engine.agents["isolde"], engine.agents["tobin"]
    reason = isolde.absence["reason"]
    back = datetime.fromisoformat(isolde.absence["until"])
    await engine._absences(back)
    for me, other in ((isolde, tobin), (tobin, isolde)):
        assert not me.is_away and me.mode == "acting" and me.plan_date is None
        assert (me.x, me.y) == engine.world.target_tile(me.persona.bed)
        assert any(other.first in m.text and reason in m.text and m.importance == 8 for m in me.memory.items)


async def test_a_departure_survives_save_and_load(tmp_path):
    engine = await _engine(tmp_path)
    engine.now = NOW
    engine.schedule_departure([engine.agents["isolde"], engine.agents["tobin"]], NOW)
    await engine._absences(EVENING)
    engine.save()
    again = await _engine(tmp_path)
    assert again.agents["tobin"].is_away and again.agents["tobin"].mode == "away"
    assert again._queue == engine._queue


# ------------------------------------------------------------------ the village notices

async def _gone(engine):
    engine.now = NOW
    engine.schedule_departure([engine.agents["isolde"], engine.agents["tobin"]], NOW)
    await engine._absences(EVENING)


async def test_the_circle_hears_that_someone_is_missing_the_next_morning(tmp_path):
    engine = await _engine(tmp_path)
    await _gone(engine)
    agnes = engine.agents["agnes"]  # Isolde's mother: same roof
    await engine._absences(EVENING + timedelta(hours=9))
    assert any("Isolde and Tobin have not been seen since Sunday evening" in m.text and m.importance == 6
               for m in agnes.memory.items)


async def test_someone_outside_the_circle_hears_nothing(tmp_path):
    engine = await _engine(tmp_path)
    group = [engine.agents["isolde"], engine.agents["tobin"]]
    inside = {o.id for o in engine.circle(group)}
    outside = [a for a in engine.agents.values() if a.id not in inside and a not in group]
    assert outside and "agnes" in inside
    await _gone(engine)
    await engine._absences(EVENING + timedelta(hours=9))
    assert not any("Tobin have not been seen" in m.text for a in outside for m in a.memory.items)


async def test_the_circle_hears_why_they_turned_back(tmp_path):
    engine = await _engine(tmp_path)
    await _gone(engine)
    agnes = engine.agents["agnes"]
    reason = engine.agents["isolde"].absence["reason"]
    await engine._absences(datetime.fromisoformat(engine.agents["isolde"].absence["until"]))
    assert any("tried to leave Hollowmere" in m.text and reason in m.text for m in agnes.memory.items)


# ------------------------------------------------------------------ the cell

from hollowmere.agents.schemas import chat_digest_schema  # noqa: E402

CELL = "guard_cell/cell_pallet"


async def test_the_guard_cell_exists_and_is_walkable(tmp_path):
    world = (await _engine(tmp_path)).world
    assert world.areas["guard_cell"].indoor and not world.areas["guard_cell"].public
    x, y = world.target_tile(CELL)
    assert world.walkable(x, y) and world.area_at(x, y) == "guard_cell"
    assert world.find_path((x, y), world.target_tile("watchhouse/duty_desk"))


async def test_an_arrest_puts_the_prisoner_and_her_baby_in_the_cell(tmp_path):
    engine = await _engine(tmp_path)
    child = await _with_baby(engine)
    engine.now = NOW
    assert engine.arrest(engine.agents["roderic"], engine.agents["isolde"], NOW)
    for who in ("isolde", child.id):
        a = engine.agents[who]
        assert a.is_jailed and (a.x, a.y) == engine.world.target_tile(CELL)
    assert datetime.fromisoformat(engine.agents["isolde"].absence["until"]).hour == 8
    await engine._absences(NOW)
    assert any("locked" in m.text and m.importance == 9 for m in engine.agents["isolde"].memory.items)
    assert "guard_cell" in engine.agents["roderic"].known_areas


async def test_the_cell_holds_only_so_many(tmp_path):
    engine = await _engine(tmp_path)
    roderic = engine.agents["roderic"]
    assert engine.arrest(roderic, engine.agents["mara"], NOW)
    assert engine.arrest(roderic, engine.agents["wenna"], NOW)
    assert not engine.arrest(roderic, engine.agents["ros"], NOW)
    assert not engine.agents["ros"].is_jailed


async def test_nobody_locks_up_themselves_the_absent_or_the_locked(tmp_path):
    engine = await _engine(tmp_path)
    roderic, mara = engine.agents["roderic"], engine.agents["mara"]
    assert not engine.arrest(roderic, roderic, NOW)
    _send_away(mara)
    assert not engine.arrest(roderic, mara, NOW)
    mara.absence = None
    assert engine.arrest(roderic, mara, NOW)
    assert not engine.arrest(mara, roderic, NOW)  # she is in the cell herself


async def test_only_an_act_in_the_digest_arrests(tmp_path):
    engine = await _engine(tmp_path)
    assert chat_digest_schema(("tavern",)).model_fields["arrested"].default == "no"
    engine.now = NOW
    engine._settle(_talk(engine, "roderic", "mara"), {"roderic": Digest(), "mara": Digest()})
    assert not engine.agents["mara"].is_jailed
    engine._settle(_talk(engine, "roderic", "mara"), {"roderic": Digest(arrested="i_locked_them_up"), "mara": Digest()})
    assert engine.agents["mara"].is_jailed and not engine.agents["roderic"].is_jailed


async def test_a_jailed_agent_sits_in_the_cell_without_planning(tmp_path):
    engine = await _engine(tmp_path)
    mara = engine.agents["mara"]
    engine.arrest(engine.agents["roderic"], mara, NOW)
    engine.now = NOW
    await engine._schedule(NOW)
    assert mara.plan == [] and mara.task.place == CELL
    assert (mara.x, mara.y) == engine.world.target_tile(CELL)


async def test_release_returns_home_and_replans(tmp_path):
    engine = await _engine(tmp_path)
    mara = engine.agents["mara"]
    engine.arrest(engine.agents["roderic"], mara, NOW)
    await engine._absences(datetime.fromisoformat(mara.absence["until"]))
    assert not mara.is_jailed and mara.plan_date is None
    assert (mara.x, mara.y) == engine.world.target_tile(mara.persona.bed)
    assert any("let out" in m.text for m in mara.memory.items)


# ------------------------------------------------------------------ fixes from the final review

async def test_the_direction_of_an_arrest_is_what_the_digest_says(tmp_path):
    engine = await _engine(tmp_path)
    engine.now = NOW
    # Mara walked up to Roderic (she is `a`); only she wrote it down, as "they locked me up"
    engine._settle(_talk(engine, "mara", "roderic"), {"mara": Digest(arrested="they_locked_me_up"), "roderic": Digest()})
    assert engine.agents["mara"].is_jailed and not engine.agents["roderic"].is_jailed


async def test_the_jailer_saying_so_jails_the_other_even_when_he_is_the_one_who_walked_up(tmp_path):
    engine = await _engine(tmp_path)
    engine.now = NOW
    engine._settle(_talk(engine, "roderic", "mara"), {"roderic": Digest(arrested="i_locked_them_up"), "mara": Digest()})
    assert engine.agents["mara"].is_jailed and not engine.agents["roderic"].is_jailed


async def test_a_birth_waits_for_a_jailed_mother_too(tmp_path):
    engine = await _engine(tmp_path)
    engine.arrest(engine.agents["roderic"], engine.agents["isolde"], NOW)
    engine.pregnancies.append({"mother_id": "isolde", "father_id": "tobin", "start_day": "1347-11-01",
                               "due_day": "1347-11-05", "resolved": False})
    engine.now = NOW
    engine._background(NOW)
    assert engine.pregnancies[0]["resolved"] is False


async def test_nobody_in_the_cell_or_out_of_town_is_sought_or_reacted_to(tmp_path):
    engine = await _engine(tmp_path)
    mara, agnes = engine.agents["mara"], engine.agents["agnes"]
    engine.arrest(engine.agents["roderic"], mara, NOW)
    agnes.mode = "acting"
    assert not engine._may_react(agnes, mara, NOW) and not engine._may_react(mara, agnes, NOW)
    before = len(engine.conversations)
    engine._start_conversation(agnes, mara, "to visit", NOW)
    engine._start_conversation(mara, agnes, "to leave", NOW)
    assert len(engine.conversations) == before and agnes.convo_id is None


async def test_a_one_day_detention_lasts_a_day_whenever_it_starts(tmp_path):
    engine = await _engine(tmp_path)
    engine.cfg.detention_min_days = engine.cfg.detention_max_days = 1
    late = datetime(1347, 11, 6, 2, 0)  # before 04:00 still counts as the day before
    engine.arrest(engine.agents["roderic"], engine.agents["mara"], late)
    assert datetime.fromisoformat(engine.agents["mara"].absence["until"]) - late >= timedelta(days=1)


async def test_a_bad_queue_entry_does_not_lose_the_others(tmp_path):
    engine = await _engine(tmp_path)
    engine._queue = [{"do": "arrested", "ids": ["nobody"], "by": "roderic", "at": NOW.isoformat()},
                     {"do": "vanish", "ids": ["isolde"], "at": NOW.isoformat(), "since": "Sunday evening"}]
    await engine._absences(NOW)
    assert any("not been seen" in m.text for m in engine.agents["agnes"].memory.items)
    engine.agents["mara"].absence = {"kind": "away", "since": NOW.isoformat(), "until": NOW.isoformat(), "reason": "x"}
    await engine._absences(NOW)  # an old save without "with" must not freeze the town
    assert engine.agents["mara"].absence is None
