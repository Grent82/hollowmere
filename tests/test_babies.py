"""Where a baby is: with whoever carries it, and the choice a mother makes for a goblin child."""

from datetime import datetime

from hollowmere.config import Config
from hollowmere.llm.fake import FakeLLM
from hollowmere.memory.embed import HashingEmbedder
from hollowmere.sim.setup import build_engine

NOW = datetime(1347, 11, 5, 12)


class Deciding(FakeLLM):
    """A mother who always makes the same choice for her goblin child."""

    def __init__(self, where: str):
        super().__init__(latency=0)
        self.where = where

    def _babyhome(self, js: dict, user: str) -> dict:
        return {"reason": "It is the only way I can see.", "where": self.where}


async def _engine(tmp_path, llm=None):
    cfg = Config()
    cfg.sim.save_path = str(tmp_path / "run.db")
    return await build_engine(cfg, llm=llm or FakeLLM(latency=0), embedder=HashingEmbedder(128))


async def _birth(engine, mother_id, father_id):
    preg = {"mother_id": mother_id, "father_id": father_id, "start_day": "1347-11-01", "due_day": "1347-11-05",
            "resolved": False}
    engine.pregnancies.append(preg)
    await engine._give_birth(preg, NOW)
    return next(a for a in engine.agents.values() if a.persona.parents == [mother_id, father_id])


async def test_given_a_human_baby_when_its_mother_walks_off_then_it_is_carried_with_her(tmp_path):
    # Given a baby of Agnes, who stands somewhere else than the cradle
    engine = await _engine(tmp_path)
    baby = await _birth(engine, "agnes", "brannoc")
    mother = engine.agents["agnes"]
    mother.x, mother.y = mother.x + 6, mother.y + 3
    # When the babies are carried
    engine._carry_babies()
    # Then the baby is where its mother is
    assert (baby.x, baby.y) == (mother.x, mother.y)


async def test_given_a_goblin_baby_in_a_human_home_when_its_mother_walks_off_then_it_is_left_hidden(tmp_path):
    # Given a goblin child of Isolde, kept at her home
    engine = await _engine(tmp_path, Deciding("keep"))
    baby = await _birth(engine, "isolde", "nib")
    cradle = (baby.x, baby.y)
    mother = engine.agents["isolde"]
    mother.x, mother.y = mother.x + 6, mother.y + 3
    # When the babies are carried
    engine._carry_babies()
    # Then the baby stays where it lies
    assert (baby.x, baby.y) == cradle


async def test_given_a_goblin_baby_in_the_den_when_the_father_moves_then_it_goes_with_him(tmp_path):
    # Given Isolde brought her child to Nib's den
    engine = await _engine(tmp_path, Deciding("father"))
    baby = await _birth(engine, "isolde", "nib")
    father = engine.agents["nib"]
    father.x, father.y = father.x + 4, father.y + 2
    # When the babies are carried
    engine._carry_babies()
    # Then the baby is with the father
    assert (baby.x, baby.y) == (father.x, father.y)


async def test_given_a_goblin_child_when_the_mother_chooses_the_den_then_it_lives_there_and_both_remember(tmp_path):
    # Given Isolde decides to give the child to its father's den
    engine = await _engine(tmp_path, Deciding("father"))
    # When it is born
    baby = await _birth(engine, "isolde", "nib")
    # Then its home is Nib's, and both parents remember the choice
    assert baby.persona.home == engine.agents["nib"].persona.home
    for parent in ("isolde", "nib"):
        assert any(baby.first in m.text and "den" in m.text for m in engine.agents[parent].memory.items
                   if m.kind == "observation")


async def test_given_a_goblin_child_when_the_mother_keeps_it_then_it_stays_in_her_home(tmp_path):
    # Given Isolde decides to keep the child at home
    engine = await _engine(tmp_path, Deciding("keep"))
    # When it is born
    baby = await _birth(engine, "isolde", "nib")
    # Then it lives where she does
    assert baby.persona.home == engine.agents["isolde"].persona.home


async def test_given_a_human_baby_when_it_is_born_then_the_mother_is_not_asked_where_to_put_it(tmp_path):
    # Given a child of two humans
    llm = FakeLLM(latency=0)
    engine = await _engine(tmp_path, llm)
    # When it is born
    await _birth(engine, "agnes", "brannoc")
    # Then no choice is asked for
    assert "baby_home" not in llm.stats.by_label
