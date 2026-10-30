"""What an agent sees of the others is put in their own words, an hour at a time, and nothing is dropped."""

import asyncio
from datetime import datetime, timedelta

from hollowmere.agents.agent import Ctx
from hollowmere.llm.client import LLMError
from hollowmere.config import Config
from hollowmere.llm.fake import FakeLLM
from hollowmere.memory.embed import HashingEmbedder
from hollowmere.sim.setup import build_engine

NOW = datetime(1347, 11, 9, 12)


async def _engine(tmp_path):
    cfg = Config()
    cfg.sim.save_path = str(tmp_path / "run.db")
    return await build_engine(cfg, llm=FakeLLM(latency=0), embedder=HashingEmbedder(128))


class Rewriting:
    """A model that puts every note into the watcher's words, or does whatever `reply` says."""

    def __init__(self, reply=None):
        self.reply, self.seen, self.fake = reply, [], FakeLLM(latency=0)

    async def structured(self, system, user, schema, **kw):
        if schema.__name__ != "Glimpses":
            return await self.fake.structured(system, user, schema, **kw)
        self.seen.append(user)
        notes = [ln.split(". ", 1)[1] for ln in user.splitlines() if ln[:1].isdigit() and ". " in ln]
        lines = self.reply(notes) if self.reply else [f"I saw how {n[0].lower()}{n[1:]}" for n in notes]
        return schema.model_validate({"lines": lines})


def _ctx(engine, llm):
    return Ctx(llm, engine.world, engine.ctx.names, {})


async def test_given_a_sighting_when_it_is_noted_then_it_waits_and_is_not_yet_a_memory(tmp_path):
    # Given Wat who sees Isolde in the lane
    engine = await _engine(tmp_path)
    wat = engine.agents["wat"]
    before = len(wat.memory.items)
    # When he notes it
    queued = wat.glimpse("Isolde Vell is in the lane.", NOW, 3, ["isolde"])
    # Then it is waiting, and memory has not grown
    assert queued and len(wat.glimpses) == 1 and len(wat.memory.items) == before


async def test_given_waiting_sightings_when_the_hour_is_written_then_each_comes_back_in_his_words_with_its_own_time(tmp_path):
    # Given two sightings noted at different minutes
    engine = await _engine(tmp_path)
    wat = engine.agents["wat"]
    wat.glimpse("Isolde Vell is in the lane.", NOW, 3, ["isolde"])
    wat.glimpse("Ros Mourne is standing about.", NOW + timedelta(minutes=10), 2, ["ros"])
    # When the hour is written
    await wat.write_glimpses(_ctx(engine, Rewriting()), NOW + timedelta(hours=1))
    # Then two memories stand, in his words, seen, about the right people, at the minutes they were seen
    new = wat.memory.items[-2:]
    assert [m.text for m in new] == ["I saw how isolde Vell is in the lane.", "I saw how ros Mourne is standing about."]
    assert [m.source for m in new] == ["saw", "saw"] and [m.about for m in new] == [["isolde"], ["ros"]]
    assert [m.created for m in new] == [NOW, NOW + timedelta(minutes=10)] and wat.glimpses == []


async def test_given_a_failing_model_when_the_hour_is_written_then_the_plain_notes_are_kept(tmp_path):
    # Given a model that fails
    engine = await _engine(tmp_path)
    wat = engine.agents["wat"]

    class Failing:
        async def structured(self, *a, **kw):
            raise LLMError("down")

    wat.glimpse("Isolde Vell is in the lane.", NOW, 3, ["isolde"])
    # When the hour is written
    await wat.write_glimpses(_ctx(engine, Failing()), NOW + timedelta(hours=1))
    # Then nothing is lost: the note stands as it was
    assert wat.memory.items[-1].text == "Isolde Vell is in the lane." and wat.glimpses == []


async def test_given_a_line_that_loses_the_person_when_the_hour_is_written_then_that_note_stays_plain(tmp_path):
    # Given a model whose first line forgets to name Isolde
    engine = await _engine(tmp_path)
    wat = engine.agents["wat"]
    wat.glimpse("Isolde Vell is in the lane.", NOW, 3, ["isolde"])
    wat.glimpse("Ros Mourne is standing about.", NOW, 2, ["ros"])
    reply = lambda notes: ["Someone hurries past.", "Ros stands about, as ever."]  # noqa: E731
    # When the hour is written
    await wat.write_glimpses(_ctx(engine, Rewriting(reply)), NOW + timedelta(hours=1))
    # Then the first is kept plain and the second is in his words
    assert [m.text for m in wat.memory.items[-2:]] == ["Isolde Vell is in the lane.", "Ros stands about, as ever."]


async def test_given_the_same_sighting_again_within_ninety_minutes_when_noted_then_it_is_not_queued_twice(tmp_path):
    # Given a sighting already noted and written
    engine = await _engine(tmp_path)
    wat = engine.agents["wat"]
    wat.glimpse("Ros Mourne is standing about.", NOW, 2, ["ros"])
    await wat.write_glimpses(_ctx(engine, Rewriting()), NOW + timedelta(minutes=30))
    # When the same sight comes again soon after
    again = wat.glimpse("Ros Mourne is standing about.", NOW + timedelta(minutes=40), 2, ["ros"])
    later = wat.glimpse("Ros Mourne is standing about.", NOW + timedelta(minutes=100), 2, ["ros"])
    # Then it is not noted twice, but after ninety minutes it is
    assert not again and later


async def test_given_the_prompt_when_written_then_it_asks_for_the_same_facts_in_the_watchers_voice(tmp_path):
    # Given waiting sightings
    engine = await _engine(tmp_path)
    wat = engine.agents["wat"]
    wat.glimpse("Isolde Vell is in the lane.", NOW, 3, ["isolde"])
    llm = Rewriting()
    # When the hour is written
    await wat.write_glimpses(_ctx(engine, llm), NOW + timedelta(hours=1))
    # Then the note is numbered in the prompt, and the call asks to add and drop nothing
    assert "1. Isolde Vell is in the lane." in llm.seen[0]
    assert "nothing added" in llm.seen[0] and "nothing left out" in llm.seen[0]


async def test_given_unwritten_sightings_when_saved_and_restored_then_they_are_still_waiting(tmp_path):
    # Given a waiting sighting
    engine = await _engine(tmp_path)
    wat = engine.agents["wat"]
    wat.glimpse("Isolde Vell is in the lane.", NOW, 3, ["isolde"])
    # When the agent is snapshotted and restored
    snap = wat.snapshot()
    wat.glimpses = []
    wat.restore(snap)
    # Then it waits again
    assert [g["text"] for g in wat.glimpses] == ["Isolde Vell is in the lane."]


async def test_given_agents_who_see_each_other_when_the_town_perceives_then_sightings_wait_until_the_hour_turns(tmp_path):
    # Given a town and a model that rewrites
    engine = await _engine(tmp_path)
    llm = Rewriting()
    engine.ctx.llm = llm
    wat, ros = engine.agents["wat"], engine.agents["ros"]
    for a in (wat, ros):
        a.mode = "acting"
    ros.x, ros.y = wat.x + 1, wat.y
    waiting = lambda: sum(len(a.glimpses) for a in engine.agents.values())  # noqa: E731
    # When it perceives, and then the hour turns
    await engine._perceive(NOW)
    queued, stored = waiting(), sum(len(a.memory.items) for a in engine.agents.values())
    engine._write_glimpses(NOW + timedelta(hours=1))
    await asyncio.gather(*[t for t in engine._bg.values() if not t.done()])
    # Then there was something waiting, nothing was in memory yet, and afterwards nothing waits and the model was asked
    assert queued > 0 and waiting() == 0 and llm.seen
    assert sum(len(a.memory.items) for a in engine.agents.values()) > stored


async def test_given_a_line_far_longer_than_its_note_when_the_hour_is_written_then_the_note_stays_plain(tmp_path):
    # Given a model that spins a long tale out of a short note
    engine = await _engine(tmp_path)
    wat = engine.agents["wat"]
    wat.glimpse("Ros Mourne is standing about.", NOW, 2, ["ros"])
    reply = lambda notes: ["Ros Mourne lingers, a shadow among shadows, watching the world with a hunger that has "  # noqa: E731
                           "nothing to do with food and a heart as heavy as the stones."]
    # When the hour is written
    await wat.write_glimpses(_ctx(engine, Rewriting(reply)), NOW + timedelta(hours=1))
    # Then the plain note is kept
    assert wat.memory.items[-1].text == "Ros Mourne is standing about."


async def test_given_sightings_when_the_prompt_is_written_then_it_says_who_is_he_and_who_is_she(tmp_path):
    # Given a sighting of Wat (a man) and of Elsbeth (a woman)
    engine = await _engine(tmp_path)
    odo = engine.agents["odo"]
    odo.glimpse("Wat Milner sits on the steps.", NOW, 2, ["wat"], "he")
    odo.glimpse("Elsbeth Graves sweeps the step.", NOW, 2, ["elsbeth"], "she")
    llm = Rewriting()
    # When the hour is written
    await odo.write_glimpses(_ctx(engine, llm), NOW + timedelta(hours=1))
    # Then the prompt says which pronoun belongs to whom
    assert "Wat Milner (he)" in llm.seen[0] and "Elsbeth Graves (she)" in llm.seen[0]


async def test_given_a_child_when_the_same_sight_comes_again_within_hours_then_it_is_not_noted(tmp_path):
    # Given a child who noted his mother asleep
    engine = await _engine(tmp_path)
    dunstan = engine.agents["wat"]
    dunstan.persona.age = 4
    dunstan.glimpse("Agnes Vell is asleep.", NOW, 3, ["agnes"])
    # When the same sight comes four hours later, and again after the day is long past
    soon = dunstan.glimpse("Agnes Vell is asleep.", NOW + timedelta(hours=4), 3, ["agnes"])
    later = dunstan.glimpse("Agnes Vell is asleep.", NOW + timedelta(hours=7), 3, ["agnes"])
    # Then only the late one is noted
    assert not soon and later


async def test_given_an_adult_when_the_same_sight_comes_after_two_hours_then_it_is_noted_again(tmp_path):
    # Given Wat who noted Isolde in the lane
    engine = await _engine(tmp_path)
    wat = engine.agents["wat"]
    wat.glimpse("Isolde Vell is in the lane.", NOW, 3, ["isolde"])
    # When the same sight comes two hours later
    again = wat.glimpse("Isolde Vell is in the lane.", NOW + timedelta(hours=2), 3, ["isolde"])
    # Then it is noted
    assert again


async def test_given_a_sight_already_in_memory_when_it_comes_again_after_a_restart_then_it_is_not_noted(tmp_path):
    # Given Wat whose memory holds the sight from an hour ago, and no note of it kept in his head
    engine = await _engine(tmp_path)
    wat = engine.agents["wat"]
    await wat.observe("Isolde Vell is in the lane.", NOW - timedelta(hours=1), 3, source="saw", about=["isolde"])
    # When the same sight comes
    noted = wat.glimpse("Isolde Vell is in the lane.", NOW, 3, ["isolde"])
    # Then it is not noted again
    assert not noted
