"""A child born while its mother is away comes into the world out of sight, and the town learns of it later."""

from datetime import datetime, timedelta

from hollowmere.config import Config
from hollowmere.llm.fake import FakeLLM
from hollowmere.memory.embed import HashingEmbedder
from hollowmere.sim.setup import build_engine

NOW = datetime(1347, 11, 5, 12)
DUE = "1347-11-05"


async def _engine(tmp_path):
    cfg = Config()
    cfg.sim.save_path = str(tmp_path / "run.db")
    engine = await build_engine(cfg, llm=FakeLLM(latency=0), embedder=HashingEmbedder(128))
    engine.now = NOW
    return engine


def _away(*agents, days=5, stays=False):
    ids = [a.id for a in agents]
    for a in agents:
        a.absence = {"kind": "away", "since": NOW.isoformat(), "until": (NOW + timedelta(days=days)).isoformat(),
                     "reason": "the road", "with": ids, "stays": stays}
        a.mode, a.path, a.task = "away", [], None


def _due(engine, mother="isolde", father="tobin"):
    preg = {"mother_id": mother, "father_id": father, "start_day": "1347-11-01", "due_day": DUE, "resolved": False}
    engine.pregnancies.append(preg)
    return preg


def _child(engine):
    return next(a for a in engine.agents.values() if "isolde" in a.persona.parents)


def _heard(agent, text):
    return [m for m in agent.memory.items if text in m.text]


async def test_given_an_away_mother_when_her_day_comes_then_the_child_is_born_and_stays_out_of_sight(tmp_path):
    # Given Isolde and Tobin away, her child due today
    engine = await _engine(tmp_path)
    _away(engine.agents["isolde"], engine.agents["tobin"])
    preg = _due(engine)
    # When her day comes
    await engine._give_birth(preg, NOW)
    # Then the child is in the world, away with them, and in nobody's sight
    child = _child(engine)
    assert child.is_away and child.mode == "away" and child.absence["with"] == ["isolde", "tobin", child.id]
    assert engine.agents["isolde"].absence["with"] == ["isolde", "tobin", child.id]
    assert not [e for e in engine.events if e["kind"] == "birth"]


async def test_given_an_away_mother_when_the_child_is_born_then_the_town_has_not_heard_of_it(tmp_path):
    # Given the same
    engine = await _engine(tmp_path)
    _away(engine.agents["isolde"], engine.agents["tobin"])
    await engine._give_birth(_due(engine), NOW)
    child = _child(engine)
    # When the child is born
    # Then her mother at home has no memory of the birth, but the two who were there have
    assert not any(child.name in m.text for m in engine.agents["agnes"].memory.items)
    assert any(child.name in m.text for m in engine.agents["isolde"].memory.items)
    assert any(child.name in m.text for m in engine.agents["tobin"].memory.items)


async def test_given_the_father_stayed_behind_when_the_child_is_born_away_then_he_does_not_know_yet(tmp_path):
    # Given Isolde alone away
    engine = await _engine(tmp_path)
    _away(engine.agents["isolde"])
    await engine._give_birth(_due(engine), NOW)
    # When the child is born
    child = _child(engine)
    # Then Tobin at home has no memory of it
    assert not any(child.name in m.text for m in engine.agents["tobin"].memory.items)


async def test_given_a_child_born_away_when_the_mother_comes_home_then_family_father_and_town_hear(tmp_path):
    # Given a child born while Isolde was alone away
    engine = await _engine(tmp_path)
    _away(engine.agents["isolde"], days=2)
    await engine.agents["agnes"].observe(engine._belly_text(engine.agents["isolde"]), NOW, 5, source="saw")
    await engine._give_birth(_due(engine), NOW)
    child = _child(engine)
    # When she comes home with it
    await engine._absences(NOW + timedelta(days=2))
    # Then the child is home, and whoever saw her belly, the father and the town learn of it
    assert not child.is_away and not engine.agents["isolde"].is_away
    assert any(child.name in m.text for m in engine.agents["agnes"].memory.items)
    assert any(child.name in m.text for m in engine.agents["tobin"].memory.items)
    assert [e for e in engine.events if e["kind"] == "birth"]


class Watching(FakeLLM):
    """A fake that keeps what the growth call was told."""

    def __init__(self):
        super().__init__(latency=0)
        self.told = []

    async def structured(self, system, user, schema, **kw):
        if schema.__name__ == "ChildGrowth":
            self.told.append(user)
        return await super().structured(system, user, schema, **kw)


async def test_given_a_baby_away_when_four_days_pass_then_it_grows_up_out_there_and_not_in_the_feed(tmp_path):
    # Given a baby born away, whose road the model has told
    cfg = Config()
    cfg.sim.save_path = str(tmp_path / "run.db")
    llm = Watching()
    engine = await build_engine(cfg, llm=llm, embedder=HashingEmbedder(128))
    engine.now = NOW
    _away(engine.agents["isolde"], days=9, stays=True)
    await engine._give_birth(_due(engine), NOW)
    child = _child(engine)
    child.absence["kids"] = "The child cut a tooth in a shepherd's hut and slept well in the straw."
    # When four days have passed
    engine._grow_children(NOW + timedelta(days=4))
    await engine._bg[f"{child.id}:grow:{child.id}"]
    # Then it is a small child, still away, the growth knew the road, and the feed says nothing of it
    assert child.persona.age == 3 and child.is_away
    assert "shepherd's hut" in llm.told[0] and "away from Hollowmere" in llm.told[0]
    assert not [e for e in engine.events if e["kind"] == "grew"]
    assert any("grown" in m.text for m in engine.agents["isolde"].memory.items)


async def test_given_a_jailed_mother_when_her_day_comes_then_the_birth_waits(tmp_path):
    # Given Isolde in the cell
    engine = await _engine(tmp_path)
    isolde = engine.agents["isolde"]
    isolde.absence = {"kind": "jailed", "since": NOW.isoformat(), "until": (NOW + timedelta(days=2)).isoformat(),
                      "reason": "x", "with": ["isolde"]}
    _due(engine)
    # When her day comes
    engine._background(NOW)
    # Then the birth waits for her
    assert engine.pregnancies[0]["resolved"] is False


async def test_given_an_away_mother_when_her_day_comes_in_the_background_then_the_birth_is_started(tmp_path):
    # Given Isolde away, her child due
    engine = await _engine(tmp_path)
    _away(engine.agents["isolde"])
    _due(engine)
    # When the background runs
    engine._background(NOW)
    # Then the birth is started
    assert engine.pregnancies[0]["resolved"] is True
