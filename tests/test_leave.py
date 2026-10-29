"""One shared call decides whether anyone settled on leaving Hollowmere, instead of both digests having to say so."""

import asyncio
from datetime import datetime

import pytest
from pydantic import ValidationError

from hollowmere.agents import prompts
from hollowmere.agents.agent import agreed_departure
from hollowmere.agents.schemas import leaving_schema
from hollowmere.config import Config
from hollowmere.llm.client import LLMError
from hollowmere.llm.fake import FakeLLM
from hollowmere.memory.embed import HashingEmbedder
from hollowmere.sim.engine import Conversation
from hollowmere.sim.setup import build_engine

TALK = [("Silas Mourne", "We leave by the north gate as the light fails, and do not look back."),
        ("Isolde Wren", "Then lead the way, Silas. I am ready to go.")]
DEFIANCE = [("Agnes Vell", "Pull back from that window, girl, before you give the watch a reason."),
            ("Isolde Wren", "I will not hide in this garret, Mother, for I am already gone before the sun sets.")]


def decision(who="both", settled=True, words="We leave as the light fails", way_out="the north gate"):
    return {"words": words, "who": who, "settled": settled, "way_out": way_out}


class Brain(FakeLLM):
    """The fake brain, except that the shared leaving call answers as scripted."""

    def __init__(self, decisions=(), fail=False):
        super().__init__(latency=0)
        self.decisions, self.fail = list(decisions), fail
        self.prompts = []

    async def structured(self, system, user, schema, *, label, temperature=None):
        if label != "leave":
            return await super().structured(system, user, schema, label=label, temperature=temperature)
        self.prompts.append(user)
        if self.fail:
            raise LLMError("the model is down")
        return schema.model_validate({"departures": self.decisions})


async def town(tmp_path, brain):
    cfg = Config()
    cfg.sim.save_path = str(tmp_path / "leave.db")
    engine = await build_engine(cfg, llm=brain, embedder=HashingEmbedder(128))
    for _ in range(3):
        await engine.tick()
    return engine


async def asked(engine, decisions, transcript=TALK, a="silas", b="isolde"):
    engine.ctx.llm = Brain(decisions)
    return await agreed_departure(engine.ctx, engine.agents[a], engine.agents[b], engine.now, transcript)


# ------------------------------------------------------------------ schema and prompt


def test_given_two_names_when_the_schema_is_built_then_only_they_or_both_can_go_and_at_most_one_decision():
    # Given a schema for Silas and Isolde
    schema = leaving_schema("Silas", "Isolde")
    # When a reply names someone else, or lists two decisions
    # Then it is rejected, and a reply naming one of them or both is accepted
    for who in ("Silas", "Isolde", "both"):
        assert schema.model_validate({"departures": [decision(who)]}).departures[0].who == who
    with pytest.raises(ValidationError):
        schema.model_validate({"departures": [decision("Tobin")]})
    with pytest.raises(ValidationError):
        schema.model_validate({"departures": [decision(), decision()]})


def test_given_a_talk_when_the_prompt_is_built_then_it_carries_the_names_the_words_and_what_is_not_leaving():
    # Given a finished talk
    # When the leaving prompt is built
    text = prompts.leaving("Silas", "Isolde", datetime(1347, 11, 12, 15, 0), TALK)
    # Then both names and the spoken words are in it, and going home or elsewhere in town is ruled out
    assert "Silas" in text and "Isolde" in text and "We leave by the north gate as the light fails" in text
    assert "Hollowmere" in text and "home" in text


def test_given_a_walk_to_a_place_beside_the_town_when_the_prompt_is_built_then_it_is_ruled_out_as_leaving():
    # Given a finished talk
    # When the leaving prompt is built
    text = prompts.leaving("Silas", "Isolde", datetime(1347, 11, 12, 15, 0), TALK)
    # Then sneaking past the Watch, a walk by the river and the places beside the town are named as not leaving
    assert "slipping past the Watch" in text and "riverbank" in text and "Hollow Keep" in text
    assert "to be gone for days" in text


# ------------------------------------------------------------------ the shared call


async def test_given_both_settled_on_leaving_when_asked_then_both_go(tmp_path):
    engine = await town(tmp_path, Brain())
    group = await asked(engine, [decision("both")])
    assert [x.id for x in group] == ["silas", "isolde"]


async def test_given_one_settled_on_leaving_alone_when_asked_then_only_that_one_goes(tmp_path):
    engine = await town(tmp_path, Brain())
    group = await asked(engine, [decision("Isolde", words="I am ready to go")])
    assert [x.id for x in group] == ["isolde"]


async def test_given_nothing_settled_when_asked_then_nobody_goes(tmp_path):
    engine = await town(tmp_path, Brain())
    assert await asked(engine, [decision(settled=False)]) == []
    assert await asked(engine, []) == []


async def test_given_words_nobody_said_when_asked_then_nobody_goes(tmp_path):
    engine = await town(tmp_path, Brain())
    assert await asked(engine, [decision(words="We take the north road at midnight")]) == []


async def test_given_defiance_without_a_way_out_when_asked_then_nobody_goes(tmp_path):
    # Given Isolde, answering her mother, says she is "already gone" and names no gate, road or place beyond the town
    engine = await town(tmp_path, Brain())
    said = decision("Isolde", words="I am already gone before the sun sets", way_out="")
    # When the reading still marks it settled
    group = await asked(engine, [said], transcript=DEFIANCE, a="agnes", b="isolde")
    # Then nobody goes: a way out has to be named
    assert group == []


async def test_given_a_way_out_nobody_named_when_asked_then_nobody_goes(tmp_path):
    # Given a reading that makes up a road the talk never mentioned
    engine = await town(tmp_path, Brain())
    said = decision("Isolde", words="I am already gone before the sun sets", way_out="the king's road")
    # When asked
    group = await asked(engine, [said], transcript=DEFIANCE, a="agnes", b="isolde")
    # Then nobody goes
    assert group == []


async def test_given_words_the_other_one_said_when_asked_then_nobody_goes(tmp_path):
    # Given the reading says Isolde decided it, but the words are Silas's
    engine = await town(tmp_path, Brain())
    said = decision("Isolde", words="We leave by the north gate")
    # When asked
    group = await asked(engine, [said])
    # Then nobody goes: whoever leaves has to have said it
    assert group == []


def test_given_a_talk_when_the_prompt_is_built_then_defiance_without_a_way_out_is_ruled_out():
    # Given a finished talk
    # When the leaving prompt is built
    text = prompts.leaving("Silas", "Isolde", datetime(1347, 11, 12, 15, 0), TALK)
    # Then a bare "I am gone" and flight to someone are not leaving, and a named way out is asked for
    assert "already gone" in text and "way out" in text and "deep wood" in text


async def test_given_the_model_fails_when_asked_then_nobody_goes_and_nothing_raises(tmp_path):
    engine = await town(tmp_path, Brain())
    engine.ctx.llm = Brain(fail=True)
    assert await agreed_departure(engine.ctx, engine.agents["silas"], engine.agents["isolde"], engine.now, TALK) == []


async def test_given_the_fake_brain_when_asked_then_it_finds_no_departure():
    out = await FakeLLM(latency=0).structured("s", "u", leaving_schema("A", "B"), label="leave")
    assert out.departures == []


# ------------------------------------------------------------------ in the town


async def finished(engine, a="silas", b="isolde", transcript=TALK):
    c = Conversation("c400", engine.agents[a], engine.agents[b], "to plan", engine.now, transcript=list(transcript))
    engine._end_conversation(c, "goodbye")
    await asyncio.gather(*engine._bg.values())


async def test_given_both_settled_on_leaving_when_the_talk_ends_then_a_departure_is_queued_for_both(tmp_path):
    engine = await town(tmp_path, Brain([decision("both")]))
    await finished(engine)
    assert [(q["do"], q["ids"]) for q in engine._queue if q["do"] == "depart"] == [("depart", ["silas", "isolde"])]


async def test_given_one_settled_on_leaving_alone_when_the_talk_ends_then_only_that_one_is_queued(tmp_path):
    engine = await town(tmp_path, Brain([decision("Isolde", words="I am ready to go")]))
    await finished(engine)
    assert [q["ids"] for q in engine._queue if q["do"] == "depart"] == [["isolde"]]


async def test_given_nothing_settled_when_the_talk_ends_then_nobody_is_queued(tmp_path):
    engine = await town(tmp_path, Brain([]))
    await finished(engine)
    assert not [q for q in engine._queue if q["do"] == "depart"]


async def test_given_someone_already_leaving_when_they_settle_it_again_then_it_is_queued_once(tmp_path):
    engine = await town(tmp_path, Brain([decision("both")]))
    await finished(engine)
    await finished(engine)  # they plan it again the next time they meet
    assert len([q for q in engine._queue if q["do"] == "depart"]) == 1


async def test_given_a_talk_that_never_began_when_it_ends_then_no_leaving_is_asked_for(tmp_path):
    brain = Brain([decision("both")])
    engine = await town(tmp_path, brain)
    c = Conversation("c401", engine.agents["silas"], engine.agents["isolde"], "to plan", engine.now)
    engine._end_conversation(c, "never_reached")
    await asyncio.gather(*engine._bg.values())
    assert brain.prompts == []
