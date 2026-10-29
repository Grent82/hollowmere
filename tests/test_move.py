"""One shared call decides whether anyone settled on moving into the other's home; the engine then moves them."""

import asyncio
from datetime import datetime, timedelta

import pytest
from pydantic import ValidationError

from hollowmere.agents import prompts
from hollowmere.agents.agent import agreed_move
from hollowmere.agents.schemas import moving_schema
from hollowmere.config import Config
from hollowmere.llm.client import LLMError
from hollowmere.llm.fake import FakeLLM
from hollowmere.memory.embed import HashingEmbedder
from hollowmere.sim.engine import Conversation
from hollowmere.sim.setup import build_engine

TALK = [("Silas Mourne", "My shack is cold and you are alone. Come and live under my roof, Clemence."),
        ("Clemence Hale", "Then I will bring my things to your shack tonight, Silas, and stay.")]
LATER = datetime(1347, 11, 5, 21, 0)


def move(who="Clemence", settled=True, words="I will bring my things to your shack tonight"):
    return {"words": words, "who": who, "settled": settled}


class Brain(FakeLLM):
    """The fake brain, except that the shared moving call answers as scripted."""

    def __init__(self, moves=(), fail=False):
        super().__init__(latency=0)
        self.moves, self.fail = list(moves), fail
        self.prompts = []

    async def structured(self, system, user, schema, *, label, temperature=None):
        if label != "move":
            return await super().structured(system, user, schema, label=label, temperature=temperature)
        self.prompts.append(user)
        if self.fail:
            raise LLMError("the model is down")
        return schema.model_validate({"moves": self.moves})


async def town(tmp_path, brain):
    cfg = Config()
    cfg.sim.save_path = str(tmp_path / "move.db")
    engine = await build_engine(cfg, llm=brain, embedder=HashingEmbedder(128))
    for _ in range(3):
        await engine.tick()
    return engine


async def asked(engine, moves, transcript=TALK, a="silas", b="clemence"):
    engine.ctx.llm = Brain(moves)
    return await agreed_move(engine.ctx, engine.agents[a], engine.agents[b], engine.now, transcript)


# ------------------------------------------------------------------ schema and prompt


def test_given_two_names_when_the_schema_is_built_then_one_of_them_moves_and_at_most_one_decision():
    # Given a schema for Silas and Clemence
    schema = moving_schema("Silas", "Clemence")
    # When a reply names a stranger, "both", or lists two decisions
    # Then it is rejected, and a reply naming one of them is accepted
    for who in ("Silas", "Clemence"):
        assert schema.model_validate({"moves": [move(who)]}).moves[0].who == who
    with pytest.raises(ValidationError):
        schema.model_validate({"moves": [move("both")]})
    with pytest.raises(ValidationError):
        schema.model_validate({"moves": [move(), move()]})


def test_given_a_talk_when_the_prompt_is_built_then_it_carries_names_words_and_what_is_not_moving_in():
    text = prompts.moving("Silas", "Clemence", datetime(1347, 11, 5, 15, 0), TALK)
    assert "Silas" in text and "Clemence" in text and "Come and live under my roof" in text
    assert "visit" in text and "home" in text


# ------------------------------------------------------------------ the shared call


async def test_given_one_settled_on_moving_in_when_asked_then_the_mover_and_the_host_are_named(tmp_path):
    engine = await town(tmp_path, Brain())
    mover, host = await asked(engine, [move("Clemence")])
    assert (mover.id, host.id) == ("clemence", "silas")


async def test_given_the_other_way_round_when_asked_then_the_roles_swap(tmp_path):
    engine = await town(tmp_path, Brain())
    mover, host = await asked(engine, [move("Silas", words="Come and live under my roof")])
    assert (mover.id, host.id) == ("silas", "clemence")


async def test_given_nothing_settled_when_asked_then_nobody_moves(tmp_path):
    engine = await town(tmp_path, Brain())
    assert await asked(engine, [move(settled=False)]) is None
    assert await asked(engine, []) is None


async def test_given_words_nobody_said_when_asked_then_nobody_moves(tmp_path):
    engine = await town(tmp_path, Brain())
    assert await asked(engine, [move(words="I give up my cell for good and the cottage is mine")]) is None


async def test_given_the_model_fails_when_asked_then_nobody_moves_and_nothing_raises(tmp_path):
    engine = await town(tmp_path, Brain())
    engine.ctx.llm = Brain(fail=True)
    silas, clemence = engine.agents["silas"], engine.agents["clemence"]
    assert await agreed_move(engine.ctx, silas, clemence, engine.now, TALK) is None


async def test_given_the_fake_brain_when_asked_then_it_finds_no_move():
    out = await FakeLLM(latency=0).structured("s", "u", moving_schema("A", "B"), label="move")
    assert out.moves == []


# ------------------------------------------------------------------ in the town


async def finished(engine, a="silas", b="clemence", transcript=TALK):
    c = Conversation("c500", engine.agents[a], engine.agents[b], "to talk", engine.now, transcript=list(transcript))
    engine._end_conversation(c, "goodbye")
    await asyncio.gather(*engine._bg.values())


def queued(engine):
    return [(q["ids"], q["to"]) for q in engine._queue if q["do"] == "move_in"]


async def test_given_a_settled_move_when_the_talk_ends_then_the_move_is_queued_for_that_evening(tmp_path):
    engine = await town(tmp_path, Brain([move("Clemence")]))
    engine.now = LATER - timedelta(hours=2)
    await finished(engine)
    assert queued(engine) == [(["clemence"], "silas")]
    assert [q["at"] for q in engine._queue if q["do"] == "move_in"] == ["1347-11-05T20:00:00"]


async def test_given_nothing_settled_when_the_talk_ends_then_no_move_is_queued(tmp_path):
    engine = await town(tmp_path, Brain([]))
    await finished(engine)
    assert queued(engine) == []


async def test_given_the_same_move_settled_twice_when_the_talks_end_then_it_is_queued_once(tmp_path):
    engine = await town(tmp_path, Brain([move("Clemence")]))
    await finished(engine)
    await finished(engine)
    assert queued(engine) == [(["clemence"], "silas")]


async def test_given_two_who_already_share_a_home_when_a_move_is_settled_then_nothing_is_queued(tmp_path):
    engine = await town(tmp_path, Brain([move("Isolde")]))
    # Isolde lives in her mother's tavern room already; moving in with Agnes means nothing
    engine.agents["isolde"].persona.home = engine.agents["agnes"].persona.home
    await finished(engine, "agnes", "isolde")
    assert queued(engine) == []


async def test_given_a_human_moving_to_a_goblin_when_a_move_is_settled_then_it_is_queued(tmp_path):
    engine = await town(tmp_path, Brain([move("Isolde")]))
    await finished(engine, "nib", "isolde")  # the den is his home, and she may choose to live in it
    assert queued(engine) == [(["isolde"], "nib")]


async def test_given_a_goblin_moving_to_a_human_when_a_move_is_settled_then_nothing_is_queued(tmp_path):
    engine = await town(tmp_path, Brain([move("Nib")]))
    await finished(engine, "nib", "isolde")  # a goblin cannot go indoors, so he cannot live under a roof
    assert queued(engine) == []


async def test_given_the_mover_is_away_when_a_move_is_settled_then_nothing_is_queued(tmp_path):
    engine = await town(tmp_path, Brain([move("Clemence")]))
    engine.agents["clemence"].absence = {"kind": "away", "since": engine.now.isoformat(),
                                         "until": (engine.now + timedelta(days=2)).isoformat(),
                                         "reason": "wolves", "with": ["clemence"]}
    await finished(engine)
    assert queued(engine) == []


async def test_given_a_talk_that_never_began_when_it_ends_then_no_move_is_asked_for(tmp_path):
    brain = Brain([move("Clemence")])
    engine = await town(tmp_path, brain)
    c = Conversation("c501", engine.agents["silas"], engine.agents["clemence"], "to talk", engine.now)
    engine._end_conversation(c, "never_reached")
    await asyncio.gather(*engine._bg.values())
    assert brain.prompts == []


# ------------------------------------------------------------------ moving in


async def test_given_the_evening_has_come_when_the_move_runs_then_the_mover_lives_and_sleeps_at_the_hosts(tmp_path):
    engine = await town(tmp_path, Brain())
    silas, clemence = engine.agents["silas"], engine.agents["clemence"]
    old_home, new_home = clemence.persona.home, silas.persona.home
    # a plan for the night: sleep in the old bed
    engine.schedule_move_in(clemence, silas, LATER - timedelta(hours=3))
    await engine._absences(datetime(1347, 11, 6, 20, 0))
    assert (clemence.persona.home, clemence.persona.home_object) == (new_home, silas.persona.home_object)
    assert clemence.persona.bed == silas.persona.bed
    assert new_home in clemence.known_areas and old_home not in clemence.known_areas
    assert new_home in clemence.place_ids and old_home not in clemence.place_ids
    assert silas.persona.home == new_home  # the host stays put


async def test_given_a_plan_that_sleeps_in_the_old_bed_when_the_move_runs_then_it_sleeps_in_the_new_one(tmp_path):
    from hollowmere.agents.agent import Block, Task
    engine = await town(tmp_path, Brain())
    silas, clemence = engine.agents["silas"], engine.agents["clemence"]
    old_bed = clemence.persona.bed
    t0 = datetime(1347, 11, 5, 23, 0)
    clemence.plan = [Block(t0, t0 + timedelta(hours=6), "goes to sleep", old_bed)]
    clemence.tasks = [Task(t0, t0 + timedelta(hours=1), "sleeps", old_bed, "z")]
    engine.schedule_move_in(clemence, silas, LATER - timedelta(hours=3))
    await engine._absences(datetime(1347, 11, 6, 20, 0))
    assert clemence.plan[0].place == silas.persona.bed and clemence.tasks[0].place == silas.persona.bed


async def test_given_a_move_when_it_runs_then_both_remember_it_and_the_feed_shows_it(tmp_path):
    engine = await town(tmp_path, Brain())
    silas, clemence = engine.agents["silas"], engine.agents["clemence"]
    engine.schedule_move_in(clemence, silas, LATER - timedelta(hours=3))
    await engine._absences(datetime(1347, 11, 6, 20, 0))
    assert any("Silas" in m.text and "moved" in m.text for m in clemence.memory.items)
    assert any("Clemence" in m.text and "moved" in m.text for m in silas.memory.items)
    assert [e for e in engine.events if e["kind"] == "move"]


async def test_given_a_mother_with_a_baby_when_she_moves_then_the_baby_moves_with_her(tmp_path):
    engine = await town(tmp_path, Brain())
    preg = {"mother_id": "clemence", "father_id": "silas", "start_day": "1347-11-01", "due_day": "1347-11-05",
            "resolved": False}
    engine.pregnancies.append(preg)
    await engine._give_birth(preg, LATER)
    baby = next(a for a in engine.agents.values() if "clemence" in a.persona.parents)
    clemence, silas = engine.agents["clemence"], engine.agents["silas"]
    engine.schedule_move_in(clemence, silas, LATER - timedelta(hours=3))
    await engine._absences(datetime(1347, 11, 6, 20, 0))
    assert baby.persona.home == silas.persona.home


async def test_given_the_host_went_away_before_evening_when_the_move_runs_then_it_is_dropped(tmp_path):
    engine = await town(tmp_path, Brain())
    silas, clemence = engine.agents["silas"], engine.agents["clemence"]
    old = clemence.persona.home
    engine.schedule_move_in(clemence, silas, LATER - timedelta(hours=3))
    silas.absence = {"kind": "away", "since": LATER.isoformat(), "until": "1347-11-09T17:00:00",
                     "reason": "wolves", "with": ["silas"]}
    await engine._absences(datetime(1347, 11, 6, 20, 0))
    assert clemence.persona.home == old


async def test_given_a_move_when_the_game_is_saved_and_loaded_then_the_new_home_is_kept(tmp_path):
    engine = await town(tmp_path, Brain())
    silas, clemence = engine.agents["silas"], engine.agents["clemence"]
    engine.schedule_move_in(clemence, silas, LATER - timedelta(hours=3))
    await engine._absences(datetime(1347, 11, 6, 20, 0))
    engine.save()
    cfg = Config()
    cfg.sim.save_path = str(tmp_path / "move.db")
    again = await build_engine(cfg, llm=Brain(), embedder=HashingEmbedder(128))
    assert again.load()
    c2 = again.agents["clemence"]
    assert c2.persona.home == silas.persona.home and c2.persona.bed == silas.persona.bed
    assert silas.persona.home in c2.known_areas


async def test_given_a_human_moves_into_the_den_when_the_move_runs_then_she_sleeps_in_the_nest_and_knows_the_den(tmp_path):
    engine = await town(tmp_path, Brain())
    isolde, nib = engine.agents["isolde"], engine.agents["nib"]
    # Given Isolde settled on living with Nib
    engine.schedule_move_in(isolde, nib, LATER - timedelta(hours=3))
    # When the evening comes
    await engine._absences(datetime(1347, 11, 6, 20, 0))
    # Then her bed is the nest and the den is a place she can name; the tavern stays hers for work
    assert isolde.persona.bed == nib.persona.bed == "goblin_den/nib_nest"
    assert "goblin_den" in isolde.known_areas and isolde.persona.work in isolde.known_areas


# ------------------------------------------------------------------ a human moving into a goblin's den

DEN_TALK = [("Nib", "Stay in my burrow for good, Isolde. Make the nest of rags your home."),
            ("Isolde Vell", "Then I will leave my mother's garret and live in your burrow, Nib.")]


async def test_given_isolde_settled_on_living_in_the_den_when_the_talk_ends_then_the_move_is_queued(tmp_path):
    # Given a talk in which Isolde settles on living in Nib's burrow
    engine = await town(tmp_path, Brain([move("Isolde", words="I will leave my mother's garret and live in your burrow")]))
    # When it ends
    await finished(engine, "nib", "isolde", DEN_TALK)
    # Then the move of a human into a goblin's den is queued, not refused
    assert queued(engine) == [(["isolde"], "nib")]


async def test_given_the_evening_when_isolde_moves_in_then_the_den_is_her_home_and_bed_and_the_garret_is_not(tmp_path):
    # Given Isolde's move into Nib's den is due
    engine = await town(tmp_path, Brain())
    isolde, nib = engine.agents["isolde"], engine.agents["nib"]
    old_home = isolde.persona.home
    engine.schedule_move_in(isolde, nib, LATER - timedelta(hours=3))
    # When the evening comes
    await engine._absences(datetime(1347, 11, 6, 20, 0))
    # Then she lives and sleeps in the den, and the garret is no longer hers to name
    assert (isolde.persona.home, isolde.persona.home_object) == ("goblin_den", "nib_nest")
    assert isolde.persona.bed == nib.persona.bed
    assert "goblin_den" in isolde.known_areas and old_home not in isolde.known_areas
    assert nib.persona.bed in isolde.place_ids and old_home not in isolde.place_ids


async def test_given_isolde_has_moved_in_when_she_goes_to_bed_then_she_can_walk_to_the_nest(tmp_path):
    # Given Isolde has moved into the den
    engine = await town(tmp_path, Brain())
    isolde, nib = engine.agents["isolde"], engine.agents["nib"]
    engine.schedule_move_in(isolde, nib, LATER - timedelta(hours=3))
    await engine._absences(datetime(1347, 11, 6, 20, 0))
    # When we look for her way to bed
    goal = engine.world.target_tile(isolde.persona.bed)
    path = engine.world.find_path((isolde.x, isolde.y), goal) if goal else []
    # Then the nest is a place on the map, she may go there, and a path leads to it
    assert goal is not None
    assert isolde.can_go(isolde.persona.bed, engine.world)
    assert path and path[-1] == goal


async def test_given_isolde_slept_in_the_garret_when_she_moves_in_then_tonights_sleep_is_in_the_nest(tmp_path):
    from hollowmere.agents.agent import Block, Task
    # Given a plan and a task for the night in her old bed
    engine = await town(tmp_path, Brain())
    isolde, nib = engine.agents["isolde"], engine.agents["nib"]
    old_bed = isolde.persona.bed
    t0 = datetime(1347, 11, 6, 23, 0)
    isolde.plan = [Block(t0, t0 + timedelta(hours=6), "goes to sleep", old_bed)]
    isolde.tasks = [Task(t0, t0 + timedelta(hours=1), "sleeps", old_bed, "z")]
    # When she moves in
    engine.schedule_move_in(isolde, nib, LATER - timedelta(hours=3))
    await engine._absences(datetime(1347, 11, 6, 20, 0))
    # Then the night's sleep is in the nest
    assert isolde.plan[0].place == nib.persona.bed and isolde.tasks[0].place == nib.persona.bed


async def test_given_isolde_moves_in_when_it_is_done_then_nib_stays_and_the_baby_stays_with_him(tmp_path):
    # Given a baby of Isolde and Nib that already lives in the den
    engine = await town(tmp_path, Brain())
    isolde, nib = engine.agents["isolde"], engine.agents["nib"]
    preg = {"mother_id": "isolde", "father_id": "nib", "start_day": "1347-11-01", "due_day": "1347-11-05",
            "resolved": False}
    engine.pregnancies.append(preg)
    await engine._give_birth(preg, LATER)
    baby = next(a for a in engine.agents.values() if "nib" in a.persona.parents)
    baby.set_home(nib.persona.home, nib.persona.home_object, engine.world)
    # When Isolde moves in
    engine.schedule_move_in(isolde, nib, LATER - timedelta(hours=3))
    await engine._absences(datetime(1347, 11, 6, 20, 0))
    # Then Nib and the baby are still in the den, and Isolde is with them
    assert nib.persona.home == "goblin_den" and baby.persona.home == "goblin_den"
    assert isolde.persona.home == "goblin_den"


async def test_given_isolde_in_the_den_when_the_game_is_saved_and_loaded_then_she_still_lives_there(tmp_path):
    # Given Isolde moved into the den and the game is saved
    engine = await town(tmp_path, Brain())
    isolde, nib = engine.agents["isolde"], engine.agents["nib"]
    engine.schedule_move_in(isolde, nib, LATER - timedelta(hours=3))
    await engine._absences(datetime(1347, 11, 6, 20, 0))
    engine.save()
    cfg = Config()
    cfg.sim.save_path = str(tmp_path / "move.db")
    # When it is loaded again
    again = await build_engine(cfg, llm=Brain(), embedder=HashingEmbedder(128))
    assert again.load()
    # Then her home, her bed and the places she knows are the den's
    i2 = again.agents["isolde"]
    assert i2.persona.home == "goblin_den" and i2.persona.bed == nib.persona.bed
    assert "goblin_den" in i2.known_areas and nib.persona.bed in i2.place_ids
