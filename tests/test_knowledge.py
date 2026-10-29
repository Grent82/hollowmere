"""Who knows what, and how they came to know it: read off the memories, without asking any model."""

import json
from datetime import datetime

from hollowmere.config import Config
from hollowmere.llm.fake import FakeLLM
from hollowmere.memory.embed import HashingEmbedder
from hollowmere.sim import knowledge
from hollowmere.sim.setup import build_engine

NOW = datetime(1347, 11, 9, 12)


async def _engine(tmp_path):
    cfg = Config()
    cfg.sim.save_path = str(tmp_path / "run.db")
    engine = await build_engine(cfg, llm=FakeLLM(latency=0), embedder=HashingEmbedder(128))
    engine.pregnancies.append({"mother_id": "isolde", "father_id": "nib", "start_day": "1347-11-03",
                               "due_day": "1347-11-08", "resolved": False, "known": True, "showing": True})
    return engine


def _fact(engine, kind):
    return next(f for f in knowledge.build(engine) if f["kind"] == kind)


def _knower(fact, agent_id):
    return next((k for k in fact["knowers"] if k["agent"] == agent_id), None)


async def test_given_sight_and_hearsay_when_the_pregnancy_is_looked_up_then_each_knows_it_by_the_way_they_came_to_it(tmp_path):
    # Given Wat saw Isolde's belly, Ros was told by Wat, and Mara knows nothing
    engine = await _engine(tmp_path)
    await engine.agents["wat"].memory.add("observation", "Isolde Vell is visibly with child; her belly is round.",
                                          NOW, 5, source="saw", about=["isolde"])
    await engine.agents["ros"].memory.add("chat", "Wat told me Isolde is carrying a child.", NOW, 6,
                                          source="told:wat", about=["isolde"])
    # When the knowledge is built
    fact = _fact(engine, "pregnancy")
    # Then Wat saw it, Ros was told by Wat, the two involved are marked, and Mara is among those who do not know
    assert _knower(fact, "wat")["how"] == "saw" and _knower(fact, "ros")["how"] == "told by Wat"
    assert _knower(fact, "isolde")["involved"] and _knower(fact, "nib")["involved"]
    assert _knower(fact, "mara") is None and any("Mara" in n for n in fact["unaware"])


async def test_given_a_memory_naming_only_the_mother_when_the_father_is_looked_up_then_it_does_not_count(tmp_path):
    # Given Tobin only knows of the pregnancy, and Agnes knows both names in one memory
    engine = await _engine(tmp_path)
    await engine.agents["tobin"].memory.add("observation", "Isolde Vell is visibly with child.", NOW, 5, source="saw")
    await engine.agents["agnes"].memory.add("chat", "Isolde confessed she is with child, Nib's child.", NOW, 7,
                                            source="told:isolde")
    # When the knowledge is built
    fact = _fact(engine, "father")
    # Then Agnes knows who the father is and Tobin does not
    assert _knower(fact, "agnes")["how"] == "told by Isolde" and _knower(fact, "tobin") is None


async def test_given_an_old_memory_without_a_source_when_it_is_read_then_its_kind_stands_in_for_it(tmp_path):
    # Given a memory from before sources were kept
    engine = await _engine(tmp_path)
    await engine.agents["wat"].memory.add("observation", "Isolde Vell is visibly with child.", NOW, 5)
    await engine.agents["ros"].memory.add("thought", "Isolde must be pregnant, I think.", NOW, 5)
    # When the knowledge is built
    fact = _fact(engine, "pregnancy")
    # Then a sight reads as seen, a thought as a thought
    assert _knower(fact, "wat")["how"] == "saw" and _knower(fact, "ros")["how"] == "thought"


async def test_given_a_pair_who_made_love_when_the_knowledge_is_built_then_they_are_lovers_and_who_knows_is_listed(tmp_path):
    # Given Isolde and Nib intimate, and Wat who heard of it
    engine = await _engine(tmp_path)
    engine.agents["isolde"].bonds["nib"]["intimate"] = True
    engine.agents["nib"].bonds["isolde"]["intimate"] = True
    await engine.agents["wat"].memory.add("chat", "Ros says Isolde and Nib are lovers, can you believe it.", NOW, 6,
                                          source="told:ros")
    # When the knowledge is built
    fact = _fact(engine, "lovers")
    # Then it names them, they are involved, and Wat was told by Ros
    assert "Isolde" in fact["text"] and "Nib" in fact["text"]
    assert _knower(fact, "isolde")["involved"] and _knower(fact, "wat")["how"] == "told by Ros"


async def test_given_a_birth_when_the_knowledge_is_built_then_the_parents_and_those_with_word_of_it_know_it(tmp_path):
    # Given a child of Agnes and Brannoc, and Wat who has word of it
    engine = await _engine(tmp_path)
    preg = {"mother_id": "agnes", "father_id": "brannoc", "start_day": "1347-11-01", "due_day": "1347-11-05",
            "resolved": False}
    engine.pregnancies.append(preg)
    await engine._give_birth(preg, NOW)
    child = next(a for a in engine.agents.values() if a.persona.parents == ["agnes", "brannoc"])
    await engine.agents["wat"].memory.add("observation", f"Word is that Agnes Vell has given birth to {child.name}.",
                                          NOW, 6, source="word")
    # When the knowledge is built
    fact = next(f for f in knowledge.build(engine) if f["kind"] == "birth" and child.name in f["text"])
    # Then the parents lived it and Wat heard of it
    assert _knower(fact, "agnes")["involved"] and _knower(fact, "wat")["how"] == "heard"


async def test_given_the_knowledge_when_it_is_sent_then_it_is_plain_data_with_counts_and_times(tmp_path):
    # Given a fact known by one
    engine = await _engine(tmp_path)
    await engine.agents["wat"].memory.add("observation", "Isolde Vell is visibly with child.", NOW, 5, source="saw")
    # When the knowledge is built and turned to JSON
    fact = _fact(engine, "pregnancy")
    text = json.dumps(fact)
    # Then it has a count of those who know, a time and the unaware
    assert fact["count"] >= 3 and fact["since"] and isinstance(fact["unaware"], list) and text


async def test_given_a_word_inside_another_when_the_birth_is_looked_up_then_it_does_not_count(tmp_path):
    # Given a child of Agnes and Brannoc, and Ros whose memory names the child but says "reason", not "son"
    engine = await _engine(tmp_path)
    preg = {"mother_id": "agnes", "father_id": "brannoc", "start_day": "1347-11-01", "due_day": "1347-11-05",
            "resolved": False}
    engine.pregnancies.append(preg)
    await engine._give_birth(preg, NOW)
    child = next(a for a in engine.agents.values() if a.persona.parents == ["agnes", "brannoc"])
    await engine.agents["ros"].memory.add("observation", f"No reason to wonder at {child.name}'s person.", NOW, 3)
    # When the knowledge is built
    fact = next(f for f in knowledge.build(engine) if f["kind"] == "birth" and child.name in f["text"])
    # Then Ros does not count as knowing of the birth
    assert _knower(fact, "ros") is None
