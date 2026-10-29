"""One shared call decides what two people agreed to, instead of each side's digest guessing on its own."""

import asyncio
from datetime import datetime

import pytest
from pydantic import ValidationError

from hollowmere.agents import prompts
from hollowmere.agents.agent import PROMISE_LEAD, Ctx, agreed_meetings, due_at
from hollowmere.agents.schemas import TIME_OF_DAY, agreement_schema
from hollowmere.config import Config
from hollowmere.llm.client import LLMError
from hollowmere.llm.fake import FakeLLM
from hollowmere.memory.embed import HashingEmbedder
from hollowmere.sim.engine import Conversation
from hollowmere.sim.setup import build_engine

TALK = [("Roderic Vane", "Join me in the back room at dusk?"), ("Agnes Vell", "At dusk, then. I will bring the wine.")]


def meeting(when_said="at dusk", time_of_day="dusk", place="tavern_back_room", day="today", both_agreed=True,
            activity="drink wine together"):
    return {"when_said": when_said, "time_of_day": time_of_day, "place": place, "day": day,
            "both_agreed": both_agreed, "activity": activity}


class Brain(FakeLLM):
    """The fake brain, except that the shared agreement call answers as scripted."""

    def __init__(self, meetings=(), fail=False):
        super().__init__(latency=0)
        self.meetings, self.fail = list(meetings), fail
        self.schemas, self.prompts = [], []

    async def structured(self, system, user, schema, *, label, temperature=None):
        if label != "agree":
            return await super().structured(system, user, schema, label=label, temperature=temperature)
        self.schemas.append(schema)
        self.prompts.append(user)
        if self.fail:
            raise LLMError("the model is down")
        return schema.model_validate({"meetings": self.meetings})


async def town(tmp_path, brain):
    cfg = Config()
    cfg.sim.save_path = str(tmp_path / "agree.db")
    engine = await build_engine(cfg, llm=brain, embedder=HashingEmbedder(128))
    for _ in range(3):
        await engine.tick()  # everyone has a plan for the day
    return engine


async def agreed(engine, brain_meetings, a="roderic", b="agnes", now=None, transcript=TALK):
    engine.ctx.llm = Brain(brain_meetings)
    x, y = engine.agents[a], engine.agents[b]
    return await agreed_meetings(engine.ctx, x, y, now or engine.now, transcript)


# ------------------------------------------------------------------ schema and prompt


def test_given_the_times_of_day_when_looked_up_then_each_names_a_clock_time_or_going_now():
    # Given the names the model may use for a time
    # When they are looked up
    # Then "straightaway" means no fixed hour and every other one is an HH:MM
    assert TIME_OF_DAY["straightaway"] is None
    assert TIME_OF_DAY["dusk"] == "18:00"
    assert all(len(t) == 5 and t[2] == ":" for k, t in TIME_OF_DAY.items() if k != "straightaway")


def test_given_a_place_list_when_the_schema_is_built_then_only_those_places_and_times_validate():
    # Given a schema for two places
    schema = agreement_schema(("tavern", "riverbank"))
    ok = meeting(place="tavern")
    # When a reply names a place or a time of day outside the lists
    # Then it is rejected, and a reply inside them is accepted
    assert schema.model_validate({"meetings": [ok]}).meetings[0].place == "tavern"
    with pytest.raises(ValidationError):
        schema.model_validate({"meetings": [{**ok, "place": "the moon"}]})
    with pytest.raises(ValidationError):
        schema.model_validate({"meetings": [{**ok, "time_of_day": "16:20"}]})
    with pytest.raises(ValidationError):
        schema.model_validate({"meetings": [ok, ok, ok]})  # at most two


def test_given_a_talk_when_the_prompt_is_built_then_it_carries_names_words_and_places():
    # Given a finished talk
    now = datetime(1347, 11, 5, 14, 0)
    # When the agreement prompt is built
    text = prompts.agreement("Roderic", "Agnes", now, TALK, {"tavern_back_room": "the back room"})
    # Then both speakers, the spoken words and the place ids are in it
    assert "Roderic" in text and "Agnes" in text
    assert "Join me in the back room at dusk?" in text
    assert "tavern_back_room" in text and "the back room" in text
    assert "14:00" in text


def test_given_one_phrase_for_two_readers_when_the_prompt_is_built_then_it_forbids_home_and_possessives():
    # Given a finished talk
    now = datetime(1347, 11, 5, 14, 0)
    # When the agreement prompt is built
    text = prompts.agreement("Roderic", "Agnes", now, TALK, {"tavern_back_room": "the back room"})
    # Then it says both read the activity, so "home", "my" and "your" must not appear in it
    assert "both of them read" in text and '"home"' in text


# ------------------------------------------------------------------ the shared call


async def test_given_both_said_yes_to_dusk_when_asked_then_one_entry_is_set_at_the_mapped_hour(tmp_path):
    engine = await town(tmp_path, Brain())
    # When the shared call reports a meeting at dusk
    found = await agreed(engine, [meeting()])
    # Then it is one entry for today at 18:00, in the place named
    assert len(found) == 1
    assert (found[0]["time"], found[0]["place"], found[0]["date"]) == (
        "18:00", "tavern_back_room", engine.now.date().isoformat())
    assert found[0]["activity"] == "drink wine together"


async def test_given_only_one_side_said_yes_when_asked_then_nothing_is_kept(tmp_path):
    engine = await town(tmp_path, Brain())
    assert await agreed(engine, [meeting(both_agreed=False)]) == []


async def test_given_a_time_nobody_spoke_when_asked_then_it_is_refused(tmp_path):
    engine = await town(tmp_path, Brain())
    # the model filled in "at midnight", which appears nowhere in the talk
    assert await agreed(engine, [meeting(when_said="at midnight", time_of_day="night")]) == []


async def test_given_tomorrow_when_asked_then_the_entry_is_dated_the_next_day(tmp_path):
    engine = await town(tmp_path, Brain())
    found = await agreed(engine, [meeting(day="tomorrow")])
    assert found[0]["date"] == "1347-11-03" and found[0]["time"] == "18:00"


async def test_given_going_straightaway_when_asked_then_it_is_set_just_ahead_of_now(tmp_path):
    engine = await town(tmp_path, Brain())
    talk = [("Roderic Vane", "Come, right now."), ("Agnes Vell", "Lead on.")]
    found = await agreed(engine, [meeting(when_said="right now", time_of_day="straightaway")], transcript=talk)
    assert due_at(found[0]) == engine.now + PROMISE_LEAD


async def test_given_the_hour_has_passed_today_when_asked_then_today_rolls_to_tomorrow(tmp_path):
    engine = await town(tmp_path, Brain())
    evening = datetime(1347, 11, 2, 20, 0)  # dusk (18:00) is behind us
    found = await agreed(engine, [meeting(day="today")], now=evening)
    assert found[0]["date"] == "1347-11-03"


async def test_given_two_people_when_asked_then_the_model_is_offered_only_places_both_can_go_to(tmp_path):
    engine = await town(tmp_path, Brain())
    isolde, nib = engine.agents["isolde"], engine.agents["nib"]
    brain = Brain()
    engine.ctx.llm = brain
    # When the shared call is made for a human and the goblin
    await agreed_meetings(engine.ctx, isolde, nib, engine.now, TALK)
    # Then the tavern (he never goes in) and her garret are not offered, but his den and the riverbank are
    offered = brain.schemas[0].model_json_schema()["$defs"]["Meeting"]["properties"]["place"]["enum"]
    assert "tavern" not in offered and isolde.persona.home not in offered
    assert nib.persona.home in offered and "riverbank" in offered


async def test_given_the_model_fails_when_asked_then_no_meeting_is_made_and_nothing_raises(tmp_path):
    engine = await town(tmp_path, Brain())
    engine.ctx.llm = Brain(fail=True)
    roderic, agnes = engine.agents["roderic"], engine.agents["agnes"]
    assert await agreed_meetings(engine.ctx, roderic, agnes, engine.now, TALK) == []


async def test_given_the_fake_brain_when_asked_for_an_agreement_then_it_finds_none():
    brain = FakeLLM(latency=0)
    out = await brain.structured("s", "u", agreement_schema(("tavern",)), label="agree")
    assert out.meetings == []


# ------------------------------------------------------------------ in the town


async def test_given_a_talk_with_a_meeting_when_it_ends_then_both_calendars_hold_the_same_one(tmp_path):
    engine = await town(tmp_path, Brain([meeting()]))
    roderic, agnes, now = engine.agents["roderic"], engine.agents["agnes"], engine.now
    c = Conversation("c300", roderic, agnes, "drink with Agnes", now, transcript=list(TALK))
    # When the talk ends and the background work finishes
    engine._end_conversation(c, "goodbye")
    await asyncio.gather(*engine._bg.values())
    # Then both hold the identical meeting, each naming the other, and the feed shows one agreement
    mine = [x for x in roderic.commitments if x["with"] == "agnes"]
    hers = [x for x in agnes.commitments if x["with"] == "roderic"]
    assert len(mine) == len(hers) == 1
    assert (mine[0]["time"], mine[0]["place"], mine[0]["date"]) == (hers[0]["time"], hers[0]["place"], hers[0]["date"])
    agreements = [e for e in engine.events if e["kind"] == "agreement"]
    assert len(agreements) == 1 and set(agreements[0]["agents"]) == {"roderic", "agnes"}


async def test_given_a_talk_without_a_meeting_when_it_ends_then_nobody_is_committed(tmp_path):
    engine = await town(tmp_path, Brain([]))
    roderic, agnes, now = engine.agents["roderic"], engine.agents["agnes"], engine.now
    before = (list(roderic.commitments), list(agnes.commitments))
    c = Conversation("c301", roderic, agnes, "chat", now, transcript=list(TALK))
    engine._end_conversation(c, "goodbye")
    await asyncio.gather(*engine._bg.values())
    assert (roderic.commitments, agnes.commitments) == before
    assert not [e for e in engine.events if e["kind"] == "agreement"]


async def test_given_one_side_is_busy_that_hour_when_the_talk_ends_then_neither_is_committed(tmp_path):
    engine = await town(tmp_path, Brain([meeting()]))
    roderic, agnes, now = engine.agents["roderic"], engine.agents["agnes"], engine.now
    day = now.date().isoformat()
    agnes.commitments = [{"date": day, "time": "18:10", "activity": "count the takings", "place": "tavern", "with": None}]
    c = Conversation("c302", roderic, agnes, "drink with Agnes", now, transcript=list(TALK))
    engine._end_conversation(c, "goodbye")
    await asyncio.gather(*engine._bg.values())
    assert not [x for x in roderic.commitments if x["with"] == "agnes"]  # a meeting only one attends is no meeting
    assert not [e for e in engine.events if e["kind"] == "agreement"]


async def test_given_a_talk_that_never_began_when_it_ends_then_no_agreement_is_asked_for(tmp_path):
    brain = Brain([meeting()])
    engine = await town(tmp_path, brain)
    roderic, agnes = engine.agents["roderic"], engine.agents["agnes"]
    c = Conversation("c303", roderic, agnes, "drink", engine.now)  # nobody spoke
    engine._end_conversation(c, "never_reached")
    await asyncio.gather(*engine._bg.values())
    assert brain.prompts == []


# ------------------------------------------------------------------ the digest keeps its errands only


async def test_given_the_digest_names_a_shared_meeting_when_digested_then_only_the_shared_call_may_keep_it(tmp_path):
    engine = await town(tmp_path, Brain())
    agnes, roderic = engine.agents["agnes"], engine.agents["roderic"]
    agnes.commitments = []
    base = {"day": "today", "time": "18:00", "place": "tavern_back_room", "accepted": True,
            "when_said": "at dusk"}

    class Reply:
        async def structured(self, system, user, schema, **kw):
            return schema.model_validate({
                "summary": "We talked.", "importance": 4, "feeling": "fond", "view_of_them": "Kind.",
                "commitments": [{**base, "who": "both", "activity": "drink wine together"},
                                {**base, "who": "me", "activity": "bring the wine", "place": "tavern"}]})

    ctx = Ctx(Reply(), engine.world, engine.ctx.names, {})
    # When Agnes digests a talk in which the digest lists both a shared meeting and her own errand
    await agnes.digest_chat(ctx, engine.now, roderic, TALK)
    # Then only her own errand is kept
    assert [c["activity"] for c in agnes.commitments] == ["bring the wine"]
