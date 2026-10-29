"""Passing things on: where a memory came from is shown to the speaker, hearsay is kept with its doubt, and what
enough people whisper becomes what the town knows."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from hollowmere.agents import prompts
from hollowmere.agents.agent import Ctx
from hollowmere.agents.schemas import chat_digest_schema
from hollowmere.config import Config
from hollowmere.llm.fake import FakeLLM
from hollowmere.memory.embed import HashingEmbedder
from hollowmere.memory.stream import Memory
from hollowmere.sim import knowledge
from hollowmere.sim.setup import build_engine

NOW = datetime(1347, 11, 9, 12)
TALK = [("Ros Mourne", "Isolde is with child, they say, and it is Nib's, mark my words."), ("Wat Milner", "Never!")]


class Digesting(FakeLLM):
    """A listener's digest that reports the claims it is told to."""

    def __init__(self, heard):
        super().__init__(latency=0)
        self.heard = heard

    def _chatdigest(self, js: dict, user: str) -> dict:
        out = super()._chatdigest(js, user)
        out["heard"] = self.heard
        return out


async def _engine(tmp_path, llm=None):
    cfg = Config()
    cfg.sim.save_path = str(tmp_path / "run.db")
    return await build_engine(cfg, llm=llm or FakeLLM(latency=0), embedder=HashingEmbedder(128))


def _mem(text, source="", when=NOW):
    return Memory(1, "observation" if source in ("saw", "lived", "word", "") else "chat", text, when, when, 4, [],
                  source, [])


def test_given_memories_from_several_sources_when_listed_then_each_says_where_it_came_from():
    # Given a memory per kind of source
    names = {"ros": "Ros Mourne", "wat": "Wat Milner"}
    mems = [_mem("Agnes is round.", "saw"), _mem("Ros says it is Brannoc's.", "told:ros"),
            _mem("Word is she gave birth.", "word"), _mem("Wat must be mad.", "own"), _mem("No source.", "")]
    # When they are listed with names
    text = prompts.memories(mems, NOW, names=names)
    # Then each line says how it was known, and an old memory with no source says nothing
    assert "you saw it" in text and "Ros told you" in text and "word going round" in text and "your own thought" in text
    assert "No source." in text and text.splitlines()[-1].count(";") == 0


async def test_given_a_talk_when_the_speaker_prompt_is_written_then_it_has_the_sources_and_the_rules_of_passing_on(tmp_path):
    # Given Wat with a memory told by Ros
    engine = await _engine(tmp_path)
    wat = engine.agents["wat"]
    told = Memory(5, "chat", "Ros says Agnes is with Brannoc's child.", NOW, NOW, 6, [], "told:ros", ["agnes"])
    # When his next line is prompted
    text = prompts.converse(wat, engine.ctx.names, NOW, "the bar", "Ros Mourne", "wipes the bar", "", [], [told])
    # Then the source is shown, and the rules say a claim about someone is not the truth and may be kept, bent or passed on
    assert "Ros told you" in text and "Passing things on" in text
    assert "keep it back" in text and "not the truth" in text


def test_given_the_digest_schema_with_people_when_a_claim_is_validated_then_only_known_names_and_three_claims_pass():
    # Given the schema that knows the names in town
    schema = chat_digest_schema(("tavern",), people=("Isolde", "Nib"))
    base = {"summary": "s", "importance": 4, "feeling": "neutral", "view_of_them": "v", "commitments": []}
    ok = {"about": "Isolde", "claim": "is with child", "trust": "believed"}
    # Then a claim about a known person passes, an unknown name, a bad trust and a fourth claim do not
    assert schema.model_validate({**base, "heard": [ok]})
    with pytest.raises(ValidationError):
        schema.model_validate({**base, "heard": [{**ok, "about": "Nobody"}]})
    with pytest.raises(ValidationError):
        schema.model_validate({**base, "heard": [{**ok, "trust": "sure"}]})
    with pytest.raises(ValidationError):
        schema.model_validate({**base, "heard": [ok] * 4})


async def _digest(engine, heard, transcript=TALK):
    wat, ros = engine.agents["wat"], engine.agents["ros"]
    await wat.digest_chat(Ctx(engine.ctx.llm, engine.world, engine.ctx.names, {}), NOW, ros, transcript)
    return wat


async def test_given_a_claim_the_other_made_when_digested_then_it_is_kept_as_told_by_them_with_its_trust(tmp_path):
    # Given Wat hears Ros say Isolde is with child, and believes it
    heard = [{"about": "Isolde", "claim": "Isolde is with child", "trust": "believed"}]
    engine = await _engine(tmp_path, Digesting(heard))
    # When he digests the talk
    wat = await _digest(engine, heard)
    # Then he remembers that Ros told him, about Isolde, as something he believes
    m = next(m for m in wat.memory.items if m.text.startswith("Ros told me that"))
    assert "Isolde is with child" in m.text and m.source == "told:ros" and m.about == ["isolde"] and m.importance == 6


async def test_given_a_doubted_claim_when_digested_then_the_memory_says_so_and_weighs_less(tmp_path):
    # Given the same claim, doubted
    heard = [{"about": "Isolde", "claim": "Isolde is with child", "trust": "doubted"}]
    engine = await _engine(tmp_path, Digesting(heard))
    # When he digests the talk
    wat = await _digest(engine, heard)
    # Then the doubt is in the memory and the weight is lower
    m = next(m for m in wat.memory.items if m.text.startswith("Ros told me that"))
    assert "I do not believe it" in m.text and m.importance == 3


async def test_given_a_claim_nobody_said_when_digested_then_it_is_dropped(tmp_path):
    # Given a claim the talk does not hold
    heard = [{"about": "Isolde", "claim": "Isolde has stolen the silver spoons", "trust": "believed"}]
    engine = await _engine(tmp_path, Digesting(heard))
    # When he digests the talk
    wat = await _digest(engine, heard)
    # Then it is not kept
    assert not any(m.text.startswith("Ros told me that") for m in wat.memory.items)


async def test_given_enough_hearsay_when_the_whispers_are_made_then_the_town_knows_it_of_the_person(tmp_path):
    # Given Isolde's pregnancy, known by sight to Wat and told or heard by Ros, Tobin and Mara
    engine = await _engine(tmp_path)
    engine.pregnancies.append({"mother_id": "isolde", "father_id": "nib", "start_day": "1347-11-03",
                               "due_day": "1347-11-08", "resolved": False, "known": True, "showing": True})
    text = "Isolde Vell is with child, so it is said."
    for who, source in (("ros", "told:wat"), ("tobin", "told:ros"), ("mara", "word")):
        await engine.agents[who].memory.add("chat", text, NOW, 5, source=source, about=["isolde"])
    base = engine.ctx.known_as.get("isolde", "")
    # When the whispers are made, twice
    engine._refresh_whispers()
    engine._refresh_whispers()
    # Then Isolde's line says it is whispered, once, after what was known before
    line = engine.ctx.known_as["isolde"]
    assert line.startswith(base) and line.count("It is whispered that") == 1 and "Isolde Vell is with child" in line


async def test_given_too_little_hearsay_when_the_whispers_are_made_then_nothing_is_added(tmp_path):
    # Given the same pregnancy, but only two who were told
    engine = await _engine(tmp_path)
    engine.pregnancies.append({"mother_id": "isolde", "father_id": "nib", "start_day": "1347-11-03",
                               "due_day": "1347-11-08", "resolved": False, "known": True, "showing": True})
    for who in ("ros", "tobin"):
        await engine.agents[who].memory.add("chat", "Isolde Vell is with child.", NOW, 5, source="told:wat")
    base = engine.ctx.known_as.get("isolde", "")
    # When the whispers are made
    engine._refresh_whispers()
    # Then nothing was added
    assert engine.ctx.known_as.get("isolde", "") == base


async def test_given_the_knowledge_when_built_then_each_fact_names_whom_it_is_about(tmp_path):
    # Given the pregnancy and a memory for each agent
    engine = await _engine(tmp_path)
    engine.pregnancies.append({"mother_id": "isolde", "father_id": "nib", "start_day": "1347-11-03",
                               "due_day": "1347-11-08", "resolved": False})
    for a in engine.agents.values():
        await a.memory.add("observation", "Isolde Vell is with child.", NOW, 5, source="saw", about=["isolde"])
    # When the knowledge is built and the facts say whom they are about
    facts = knowledge.build(engine)
    # Then each fact carries the people it is about
    assert all("about" in f for f in facts) and next(f for f in facts if f["kind"] == "pregnancy")["about"] == ["isolde"]


async def test_given_a_claim_that_does_not_name_its_subject_when_digested_then_it_is_dropped(tmp_path):
    # Given a "claim about Isolde" that is only about the talk itself and never names her
    heard = [{"about": "Isolde", "claim": "Isolde is with child", "trust": "believed"},
             {"about": "Nib", "claim": "they say mark my words", "trust": "believed"}]
    engine = await _engine(tmp_path, Digesting(heard))
    # When Wat digests Ros's talk
    wat = await _digest(engine, heard)
    # Then the one naming Isolde stays, and the one that names no one is dropped
    told = [m for m in wat.memory.items if m.text.startswith("Ros told me that")]
    assert [m.about for m in told] == [["isolde"]]


async def test_given_a_claim_made_mostly_of_the_listeners_own_words_when_digested_then_it_is_dropped(tmp_path):
    # Given a talk where Ros only said a little, and a claim that lifts most of its words from Wat's own lines
    talk = [("Ros Mourne", "Isolde seems tired."), ("Wat Milner", "Isolde is building my whole life with me, a man who wants to stay.")]
    heard = [{"about": "Isolde", "claim": "Isolde is building my whole life with me, a man who wants to stay",
              "trust": "believed"}]
    engine = await _engine(tmp_path, Digesting(heard))
    # When Wat digests it
    wat = await _digest(engine, heard, talk)
    # Then nothing of it is kept as Ros's
    assert not any(m.text.startswith("Ros told me that") for m in wat.memory.items)


async def test_given_the_digest_when_the_schema_is_built_then_neither_the_listener_nor_the_speaker_is_a_third_person(tmp_path):
    # Given a digest by Wat of a talk with Ros, with a model that records what schema it was given
    seen = []

    class Spy(Digesting):
        def _chatdigest(self, js: dict, user: str) -> dict:
            seen.append(js["$defs"]["Heard"]["properties"]["about"]["enum"])
            return super()._chatdigest(js, user)

    engine = await _engine(tmp_path, Spy([]))
    # When he digests it
    await _digest(engine, [])
    # Then Ros and Wat are not among those a claim may be about, and Isolde is
    assert "Ros" not in seen[0] and "Wat" not in seen[0] and "Isolde" in seen[0]
