"""The player whispers a dream or a rumor to a person; it reaches only their memory, never their bonds."""

from datetime import datetime

import pytest

from hollowmere.agents.agent import Ctx
from hollowmere.config import Config
from hollowmere.llm.fake import FakeLLM
from hollowmere.memory.embed import HashingEmbedder
from hollowmere.sim.setup import build_engine


async def _engine(tmp_path):
    cfg = Config()
    cfg.sim.save_path = str(tmp_path / "run.db")
    return await build_engine(cfg, llm=FakeLLM(latency=0), embedder=HashingEmbedder(128))


async def test_given_a_rumor_when_whispered_then_the_person_remembers_having_heard_it(tmp_path):
    # Given a town and a rumor for Wat
    engine = await _engine(tmp_path)
    wat = engine.agents["wat"]
    # When it is whispered
    await engine.whisper("wat", "The Captain keeps a second ledger.", "rumor")
    # Then Wat holds it as word going round, and the player's feed records the whisper
    m = wat.memory.items[-1]
    assert m.text == "The Captain keeps a second ledger." and m.source == "word" and m.importance == 6
    assert any(e["kind"] == "whisper" for e in engine.events)


async def test_given_a_dream_when_whispered_then_it_comes_with_the_morning_as_their_own(tmp_path):
    # Given a dream for Wat
    engine = await _engine(tmp_path)
    wat = engine.agents["wat"]
    await engine.whisper("wat", "I dreamt the mill wheel ran backwards.", "dream")
    before = len(wat.memory.items)
    # When Wat plans the next day
    await wat.plan_day(Ctx(engine.ctx.llm, engine.world, engine.ctx.names, {}), engine.now)
    # Then the dream is a memory of his own, and nothing is left waiting
    dream = next(m for m in wat.memory.items[before - 1:] if m.text.startswith("I dreamt"))
    assert dream.source == "own" and wat.pending_dream is None


async def test_given_two_dreams_before_morning_when_whispered_then_both_are_kept(tmp_path):
    # Given a first dream waiting
    engine = await _engine(tmp_path)
    await engine.whisper("wat", "I dreamt of rain.", "dream")
    # When a second is whispered
    await engine.whisper("wat", "I dreamt of fire.", "dream")
    # Then one dream holds both
    text = engine.agents["wat"].pending_dream["text"]
    assert "rain" in text and "fire" in text


@pytest.mark.parametrize("who, text, kind", [("nobody", "x", "rumor"), ("wat", "  ", "rumor"), ("wat", "x", "prank")])
async def test_given_a_bad_whisper_when_made_then_it_is_refused(tmp_path, who, text, kind):
    # Given a town
    engine = await _engine(tmp_path)
    # When the whisper names no one known, says nothing, or is of no kind we know
    # Then it is refused
    with pytest.raises(ValueError):
        await engine.whisper(who, text, kind)
