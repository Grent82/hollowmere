"""A small event each morning (weather, a market, an omen of a god), seen only through the people who are there."""

import asyncio
from datetime import datetime

import pytest
from pydantic import ValidationError

from hollowmere.agents import prompts
from hollowmere.agents.schemas import daily_event_schema
from hollowmere.config import Config
from hollowmere.llm.client import LLMError
from hollowmere.llm.fake import FakeLLM
from hollowmere.memory.embed import HashingEmbedder
from hollowmere.sim.setup import build_engine

MORNING = datetime(1347, 11, 10, 4, 10)
NOON = datetime(1347, 11, 10, 12, 0)


class Eventful(FakeLLM):
    """Answers the daily-event call from a script (or fails), and remembers what it was asked."""

    def __init__(self, **event):
        super().__init__(latency=0)
        self.event = {"quiet": False, "kind": "weather", "text": "A cold fog settles over the whole town.",
                      "where": "town", "hour": 9, "god": "none", "clarity": "none", "importance": 4, **event}
        self.asked, self.fail = [], False

    async def structured(self, system, user, schema, *, label, temperature=None):
        if label == "daily_event":
            self.asked.append((user, schema))
            if self.fail:
                raise LLMError("down")
        return await super().structured(system, user, schema, label=label, temperature=temperature)

    def _dailyevent(self, js: dict, user: str) -> dict:
        return self.event


async def _engine(tmp_path, llm=None, **sim):
    cfg = Config()
    cfg.sim.save_path = str(tmp_path / "run.db")
    cfg.sim.daily_event_quiet_chance = 0.0
    for k, v in sim.items():
        setattr(cfg.sim, k, v)
    engine = await build_engine(cfg, llm=llm or Eventful(), embedder=HashingEmbedder(128))
    engine.now = MORNING
    return engine


async def _run_day(engine, morning=MORNING, noon=NOON):
    await engine._run_daily_event(morning)
    await engine._deliver_events(noon)


def _saw(agent, needle):
    return any(needle in m.text for m in agent.memory.items)


async def test_given_a_new_day_when_asked_then_the_event_is_due_once_and_not_again_that_day(tmp_path):
    # Given the engine at the start of a day
    engine = await _engine(tmp_path)
    # Then it is due, and no longer once it has started for that day
    assert engine._daily_due(MORNING)
    engine._daily_for = MORNING.date()
    assert not engine._daily_due(MORNING)
    assert engine._daily_due(datetime(1347, 11, 11, 4, 30))


async def test_given_weather_for_the_whole_town_when_it_comes_then_everyone_awake_has_seen_it(tmp_path):
    # Given a fog for the whole town, with Wat awake and Ros asleep
    engine = await _engine(tmp_path)
    wat, ros = engine.agents["wat"], engine.agents["ros"]
    wat.mode, ros.mode = "acting", "asleep"
    # When the day's event is made and its hour comes
    await _run_day(engine)
    # Then Wat remembers it and Ros, asleep, does not
    assert _saw(wat, "A cold fog settles") and not _saw(ros, "A cold fog settles")


async def test_given_an_event_at_one_place_when_it_comes_then_only_those_there_see_it(tmp_path):
    # Given a market at the bakery, Elsbeth in it and Wat elsewhere, both awake
    engine = await _engine(tmp_path, Eventful(kind="market", text="A hawker sets out turnips.", where="bakery"))
    elsbeth, wat = engine.agents["elsbeth"], engine.agents["wat"]
    elsbeth.mode = wat.mode = "acting"
    elsbeth.x, elsbeth.y = engine.world.target_tile("bakery")
    wat.x, wat.y = engine.world.target_tile("mill")
    # When the day's event is made and its hour comes
    await _run_day(engine)
    # Then only Elsbeth saw the turnips
    assert _saw(elsbeth, "hawker sets out turnips") and not _saw(wat, "hawker sets out turnips")


async def test_given_a_sign_of_a_god_when_seen_then_it_is_only_as_certain_as_the_god_made_it_and_the_feed_never_names_it(tmp_path):
    # Given Agrath recently prayed to, and a storm sign she lets be suspected
    llm = Eventful(kind="omen", text="Lightning splits the dead yew.", where="town", god="agrath", clarity="suspected")
    engine = await _engine(tmp_path, llm)
    engine._event("prayer", ["odo"], "odo prays to agrath: a roll of thunder.", form="sign", god="agrath", clarity="none")
    wat = engine.agents["wat"]
    wat.mode = "acting"
    # When the day's event is made and its hour comes
    await _run_day(engine)
    # Then Wat only suspects Agrath, and the event in the feed does not name her
    assert _saw(wat, "I suspect it was Agrath's doing")
    feed = [e for e in engine.events if e["kind"] == "daily"]
    assert feed and "Agrath" not in feed[0]["text"] and feed[0]["god"] == "agrath"


async def test_given_a_god_nobody_prayed_to_when_the_schema_is_built_then_that_god_cannot_be_chosen(tmp_path):
    # Given Agrath prayed to and nobody else
    llm = Eventful()
    engine = await _engine(tmp_path, llm)
    engine._event("prayer", ["odo"], "odo prays to agrath: thunder.", form="sign", god="agrath", clarity="none")
    # When the event is made
    await engine._run_daily_event(MORNING)
    # Then only Agrath, or no god, is a choice
    schema = llm.asked[0][1]
    ok = {"quiet": False, "kind": "omen", "text": "x", "where": "town", "hour": 9, "clarity": "none", "importance": 3}
    assert schema.model_validate({**ok, "god": "agrath"}) and schema.model_validate({**ok, "god": "none"})
    with pytest.raises(ValidationError):
        schema.model_validate({**ok, "god": "yevra"})


async def test_given_a_god_event_two_days_ago_when_the_next_is_made_then_no_god_may_be_chosen(tmp_path):
    # Given Agrath prayed to, and her sign two days ago
    llm = Eventful()
    engine = await _engine(tmp_path, llm)
    engine._event("prayer", ["odo"], "odo prays to agrath: thunder.", form="sign", god="agrath", clarity="none")
    engine.now = datetime(1347, 11, 8, 9)
    engine._event("daily", [], "A great storm breaks.", god="agrath", sort="omen")
    # When the next event is made
    await engine._run_daily_event(MORNING)
    # Then no god is a choice
    schema = llm.asked[0][1]
    ok = {"quiet": False, "kind": "omen", "text": "x", "where": "town", "hour": 9, "clarity": "none", "importance": 3}
    with pytest.raises(ValidationError):
        schema.model_validate({**ok, "god": "agrath"})


async def test_given_the_prompt_when_written_then_it_shows_earlier_events_and_gods_and_has_no_strangers(tmp_path):
    # Given an earlier daily event and Yevra's wish on file
    llm = Eventful()
    engine = await _engine(tmp_path, llm)
    engine.now = datetime(1347, 11, 9, 9)
    engine._event("daily", [], "A hawker sets out turnips.", sort="market")
    engine._event("prayer", ["odo"], "odo prays to yevra: warmth.", form="sign", god="yevra", clarity="none")
    # When the event is made
    await engine._run_daily_event(MORNING)
    # Then the prompt lists the earlier event and Yevra's nature, and no stranger is offered
    user = llm.asked[0][0]
    assert "A hawker sets out turnips." in user and "Yevra" in user and "love" in user
    assert "stranger" not in user.lower()


async def test_given_a_quiet_day_when_the_model_says_so_then_nothing_happens(tmp_path):
    # Given a model that finds the day quiet
    engine = await _engine(tmp_path, Eventful(quiet=True))
    wat = engine.agents["wat"]
    wat.mode = "acting"
    # When the day is run
    await _run_day(engine)
    # Then nobody saw anything and no event was written
    assert not [e for e in engine.events if e["kind"] == "daily"] and engine._pending_events == []


async def test_given_a_failing_model_when_the_day_starts_then_nothing_is_hurt_and_it_is_not_tried_again(tmp_path):
    # Given a model that is down
    llm = Eventful()
    llm.fail = True
    engine = await _engine(tmp_path, llm)
    # When the event task runs, for the day
    await engine._run_daily_event_task(MORNING)
    await engine._run_daily_event_task(MORNING)
    # Then nothing was raised, nothing pending, and it asked once
    assert engine._pending_events == [] and len(llm.asked) == 1


async def test_given_an_event_still_to_come_when_saved_and_loaded_then_it_is_kept(tmp_path):
    # Given a pending event for later in the day
    engine = await _engine(tmp_path)
    await engine._run_daily_event(MORNING)
    assert engine._pending_events
    engine.save()
    # When the save is loaded again
    again = await build_engine(engine_cfg(tmp_path), llm=Eventful(), embedder=HashingEmbedder(128))
    # Then the event is still waiting
    assert [e["text"] for e in again._pending_events] == ["A cold fog settles over the whole town."]


def engine_cfg(tmp_path):
    cfg = Config()
    cfg.sim.save_path = str(tmp_path / "run.db")
    return cfg


async def test_given_the_schema_when_built_then_the_hour_and_place_are_bounded(tmp_path):
    # Given the schema for two places and no god
    schema = daily_event_schema(("bakery", "mill"), ())
    ok = {"quiet": False, "kind": "weather", "text": "x", "where": "mill", "hour": 9, "god": "none",
          "clarity": "none", "importance": 3}
    # Then a place of the town or "town" passes, and an invented place or an hour in the night does not
    assert schema.model_validate(ok) and schema.model_validate({**ok, "where": "town"})
    with pytest.raises(ValidationError):
        schema.model_validate({**ok, "where": "moon"})
    with pytest.raises(ValidationError):
        schema.model_validate({**ok, "hour": 2})


async def test_given_no_god_may_stir_when_the_schema_is_built_then_an_omen_is_not_a_choice():
    # Given the schema with no god, and then with one
    plain = daily_event_schema(("bakery",), ())
    holy = daily_event_schema(("bakery",), ("agrath",))
    ok = {"quiet": False, "kind": "omen", "text": "x", "where": "town", "hour": 9, "god": "none",
          "clarity": "none", "importance": 3}
    # Then an omen is only possible where a god may be behind it
    with pytest.raises(ValidationError):
        plain.model_validate(ok)
    assert holy.model_validate(ok)


async def test_given_the_prompt_when_written_then_ordinary_and_small_is_asked_and_omens_are_the_gods(tmp_path):
    # Given any day
    llm = Eventful()
    engine = await _engine(tmp_path, llm)
    # When the event is made
    await engine._run_daily_event(MORNING)
    # Then the prompt asks for plain, everyday things and keeps eerie signs for the gods
    user = llm.asked[0][0]
    assert "everyday" in user and "eerie" in user
