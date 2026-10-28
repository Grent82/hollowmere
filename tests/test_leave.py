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

TALK = [("Silas Mourne", "We leave as the light fails, and do not look back."),
        ("Isolde Wren", "Then lead the way, Silas. I am ready to go.")]


def decision(who="both", settled=True, words="We leave as the light fails"):
    return {"words": words, "who": who, "settled": settled}


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
    assert "Silas" in text and "Isolde" in text and "We leave as the light fails" in text
    assert "Hollowmere" in text and "home" in text


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
