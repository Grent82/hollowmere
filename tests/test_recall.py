"""The player calls someone who is away back home; it wins over whatever the model had in mind for them."""

from datetime import datetime, timedelta

import pytest

from hollowmere.config import Config
from hollowmere.llm.fake import FakeLLM
from hollowmere.memory.embed import HashingEmbedder
from hollowmere.sim.setup import build_engine

NOW = datetime(1347, 11, 5, 12)


class Silent(FakeLLM):
    """A model that must not be asked about the journey."""

    async def structured(self, system, user, schema, **kw):
        assert schema.__name__ != "Journey", "a recall is the player's, the model is not asked"
        return await super().structured(system, user, schema, **kw)


async def _engine(tmp_path):
    cfg = Config()
    cfg.sim.save_path = str(tmp_path / "run.db")
    engine = await build_engine(cfg, llm=Silent(latency=0), embedder=HashingEmbedder(128))
    engine.now = NOW
    return engine


def _away(*agents, stays=True, days=30):
    ids = [a.id for a in agents]
    for a in agents:
        a.absence = {"kind": "away", "since": NOW.isoformat(), "until": (NOW + timedelta(days=days)).isoformat(),
                     "reason": "the north road", "with": ids, "stays": stays}
        a.mode, a.path, a.task = "away", [], None


async def _kin(engine):
    preg = {"mother_id": "isolde", "father_id": "tobin", "start_day": "1347-11-01", "due_day": "1347-11-05",
            "resolved": False}
    engine.pregnancies.append(preg)
    await engine._give_birth(preg, NOW)
    return next(a for a in engine.agents.values() if "isolde" in a.persona.parents)


async def test_given_two_away_with_a_child_when_the_mother_is_recalled_then_she_and_the_child_are_home_by_evening(tmp_path):
    # Given Isolde and Tobin away with their child
    engine = await _engine(tmp_path)
    child = await _kin(engine)
    _away(engine.agents["isolde"], engine.agents["tobin"], child)
    # When Isolde is called home
    await engine.recall("isolde")
    # Then she and the child are due home at five that evening, and Tobin is left on his own road
    isolde, tobin = engine.agents["isolde"], engine.agents["tobin"]
    assert datetime.fromisoformat(isolde.absence["until"]) == datetime(1347, 11, 5, 17)
    assert datetime.fromisoformat(child.absence["until"]) == datetime(1347, 11, 5, 17)
    assert tobin.absence["until"] != isolde.absence["until"] and tobin.absence["with"] == ["tobin"]
    assert isolde.absence["with"] == ["isolde", child.id]


async def test_given_a_recall_when_the_hour_comes_then_they_are_home_with_a_memory_of_being_drawn_home(tmp_path):
    # Given Isolde called home
    engine = await _engine(tmp_path)
    _away(engine.agents["isolde"])
    await engine.recall("isolde")
    # When the evening comes
    await engine._absences(datetime(1347, 11, 5, 17))
    # Then she is home, and the model was not asked
    isolde = engine.agents["isolde"]
    assert not isolde.is_away and isolde.mode == "acting"
    assert any("home" in m.text and m.importance == 8 and m.source == "lived" for m in isolde.memory.items)


async def test_given_a_recall_when_she_is_home_then_the_town_hears_she_is_back(tmp_path):
    # Given Isolde called home
    engine = await _engine(tmp_path)
    _away(engine.agents["isolde"])
    await engine.recall("isolde")
    # When she is home
    await engine._absences(datetime(1347, 11, 5, 17))
    # Then her mother hears she is back, and the feed has the recall
    assert any("Isolde" in m.text and "back" in m.text for m in engine.agents["agnes"].memory.items)
    assert any(e["kind"] == "recall" for e in engine.events)


@pytest.mark.parametrize("who", ["nobody", "agnes"])
async def test_given_someone_unknown_or_at_home_when_recalled_then_it_is_refused(tmp_path, who):
    # Given a town where Agnes is at home
    engine = await _engine(tmp_path)
    # When someone unknown or at home is recalled
    # Then it is refused
    with pytest.raises(ValueError):
        await engine.recall(who)


async def test_given_a_jailed_person_when_recalled_then_it_is_refused(tmp_path):
    # Given Roderic... no, Mara, in the cell
    engine = await _engine(tmp_path)
    engine.arrest(engine.agents["roderic"], engine.agents["mara"], NOW)
    # When she is recalled
    # Then it is refused: the cell is not a road
    with pytest.raises(ValueError):
        await engine.recall("mara")


async def test_given_a_recall_late_in_the_evening_when_made_then_they_are_home_the_next_evening(tmp_path):
    # Given it is after five already
    engine = await _engine(tmp_path)
    engine.now = datetime(1347, 11, 5, 19)
    _away(engine.agents["isolde"])
    # When Isolde is called home
    await engine.recall("isolde")
    # Then she is due the next evening
    assert datetime.fromisoformat(engine.agents["isolde"].absence["until"]) == datetime(1347, 11, 6, 17)
