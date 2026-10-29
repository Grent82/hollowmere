"""Where a memory comes from: seen, lived, told by someone, word going round, or one's own conclusion."""

import sqlite3
from datetime import datetime

import numpy as np

from hollowmere.agents.agent import Ctx, mentions
from hollowmere.config import Config
from hollowmere.llm.fake import FakeLLM
from hollowmere.memory.embed import HashingEmbedder
from hollowmere.memory.store import Store
from hollowmere.sim.setup import build_engine

NOW = datetime(1347, 11, 9, 12)


async def _engine(tmp_path):
    cfg = Config()
    cfg.sim.save_path = str(tmp_path / "run.db")
    return await build_engine(cfg, llm=FakeLLM(latency=0), embedder=HashingEmbedder(128))


async def test_given_a_memory_with_a_source_when_stored_and_loaded_then_source_and_subjects_come_back(tmp_path):
    # Given a memory seen by Wat, about Isolde
    engine = await _engine(tmp_path)
    wat = engine.agents["wat"]
    await wat._remember("observation", "Isolde is in the lane.", NOW, 3, source="saw", about=["isolde"])
    # When the agent's memories are loaded again from the save
    await wat.memory.load()
    # Then source and subject are there
    last = wat.memory.items[-1]
    assert last.text == "Isolde is in the lane." and last.source == "saw" and last.about == ["isolde"]
    assert last.as_dict()["source"] == "saw" and last.as_dict()["about"] == ["isolde"]


def test_given_a_save_from_before_sources_when_opened_then_the_columns_are_added_and_old_rows_stay_readable(tmp_path):
    # Given an old save with the memories table as it was
    path = tmp_path / "old.db"
    db = sqlite3.connect(path)
    db.executescript("""CREATE TABLE memories (id INTEGER PRIMARY KEY, agent_id TEXT NOT NULL, kind TEXT NOT NULL,
        text TEXT NOT NULL, created TEXT NOT NULL, last_accessed TEXT NOT NULL, importance INTEGER NOT NULL,
        evidence TEXT NOT NULL DEFAULT '[]', embedding BLOB);
        INSERT INTO memories(agent_id, kind, text, created, last_accessed, importance) VALUES
        ('wat', 'observation', 'old', '1347-11-02T05:00:00', '1347-11-02T05:00:00', 3);""")
    db.commit()
    db.close()
    # When it is opened and a new memory is added
    store = Store(path)
    store.insert_memory("wat", "chat", "new", NOW, 4, [], np.zeros(4), source="told:ros", about=["agnes"])
    rows = store.load_memories("wat")
    # Then the old row has no source and the new one has
    assert rows[0]["source"] == "" and rows[0]["about"] == "[]"
    assert rows[1]["source"] == "told:ros"


async def test_given_thoughts_and_reflections_when_remembered_then_their_source_is_own(tmp_path):
    # Given Wat
    engine = await _engine(tmp_path)
    wat = engine.agents["wat"]
    # When he thinks and reflects
    await wat._remember("thought", "I wonder.", NOW, 3)
    await wat._remember("reflection", "I see now.", NOW, 5)
    # Then both are his own conclusions
    assert [m.source for m in wat.memory.items[-2:]] == ["own", "own"]


def test_given_a_text_when_names_are_looked_for_then_full_and_first_names_count_but_not_the_speaker_or_part_words():
    # Given a few names
    names = {"isolde": "Isolde Vell", "nib": "Nib", "ros": "Ros Mourne", "wat": "Wat Milner"}
    # When a text mentions Isolde by first name, Nib in full, and "across" (which holds "ros")
    found = mentions("Isolde told me Nib was across the river; I, Wat, said nothing.", names, exclude="wat")
    # Then Isolde and Nib are found, and neither Ros nor the speaker
    assert found == ["isolde", "nib"]


async def test_given_two_people_in_sight_when_one_looks_then_the_sentence_is_stored_as_seen_and_about_the_other(tmp_path):
    # Given Wat and Ros side by side, both awake
    engine = await _engine(tmp_path)
    wat, ros = engine.agents["wat"], engine.agents["ros"]
    for a in (wat, ros):
        a.mode = "acting"
    ros.x, ros.y = wat.x + 1, wat.y
    # When the town is perceived
    await engine._perceive(NOW)
    # Then what Wat stored of Ros is a sight, about Ros
    seen = [m for m in wat.memory.items if m.text.startswith(ros.name)]
    assert seen and seen[-1].source == "saw" and seen[-1].about == [ros.id]


async def test_given_a_talk_when_it_is_digested_then_the_summary_is_told_by_the_other_and_names_whom_it_mentions(tmp_path):
    # Given Wat and Ros and a talk, with the fake model's summary
    engine = await _engine(tmp_path)
    wat, ros = engine.agents["wat"], engine.agents["ros"]
    ctx = Ctx(engine.ctx.llm, engine.world, engine.ctx.names, {})
    # When Wat digests it
    await wat.digest_chat(ctx, NOW, ros, [("Ros Mourne", "Isolde is with child, they say."), ("Wat Milner", "Never!")])
    # Then the summary is a thing told by Ros
    summary = next(m for m in reversed(wat.memory.items) if m.kind == "chat")
    assert summary.source == "told:ros"


async def test_given_a_birth_when_it_is_told_then_parents_lived_it_and_those_who_heard_had_word_of_it(tmp_path):
    # Given Wat saw Agnes's belly
    engine = await _engine(tmp_path)
    wat = engine.agents["wat"]
    await wat.observe(engine._belly_text(engine.agents["agnes"]), NOW, 5, source="saw", about=["agnes"])
    preg = {"mother_id": "agnes", "father_id": "brannoc", "start_day": "1347-11-01", "due_day": "1347-11-05",
            "resolved": False}
    engine.pregnancies.append(preg)
    # When the child is born
    await engine._give_birth(preg, NOW)
    # Then Agnes lived it, and Wat has only word of it
    child = next(a for a in engine.agents.values() if a.persona.parents == ["agnes", "brannoc"])
    lived = next(m for m in engine.agents["agnes"].memory.items if child.name in m.text and "gave birth" in m.text)
    heard = next(m for m in wat.memory.items if child.name in m.text and "Word is" in m.text)
    assert lived.source == "lived" and heard.source == "word" and set(heard.about) == {"agnes", child.id}
