"""A newborn is written from its parents, and a child of a goblin is always a goblin."""

import random
from datetime import date, datetime

from hollowmere.agents import prompts, romance
from hollowmere.agents.persona import Persona
from hollowmere.config import Config
from hollowmere.llm.client import LLMError
from hollowmere.llm.fake import FakeLLM
from hollowmere.memory.embed import HashingEmbedder
from hollowmere.sim.setup import build_engine

NOW = datetime(1347, 11, 9, 12)
DRAFT = {"name": "Maude Hale", "appearance": "A round-faced baby with her mother's green eyes and a thatch of dark hair.",
         "background": "Maude Hale is the first child of Agnes Vell and Brannoc Hale, born in a cold garret."}


class Writing(FakeLLM):
    """A model that writes the newborn as scripted, or fails."""

    def __init__(self, draft=None, fail=False, then=None):
        super().__init__(latency=0)
        self.draft, self.fail, self.asked, self.then = draft or DRAFT, fail, [], list(then or [])

    async def structured(self, system, user, schema, *, label, temperature=None):
        if label == "child_persona":
            self.asked.append(user)
            if self.fail:
                raise LLMError("down")
            return schema.model_validate(self.then.pop(0) if self.asked[1:] and self.then else self.draft)
        return await super().structured(system, user, schema, label=label, temperature=temperature)


async def _engine(tmp_path, llm):
    cfg = Config()
    cfg.sim.save_path = str(tmp_path / "run.db")
    return await build_engine(cfg, llm=llm, embedder=HashingEmbedder(128))


async def _birth(engine, mother_id="agnes", father_id="brannoc"):
    preg = {"mother_id": mother_id, "father_id": father_id, "start_day": "1347-11-01", "due_day": "1347-11-05",
            "resolved": False}
    engine.pregnancies.append(preg)
    await engine._give_birth(preg, NOW)
    return next(a for a in engine.agents.values() if a.persona.parents == [mother_id, father_id])


def _parent(species: str, name="Parent X") -> Persona:
    return Persona(id=name.split()[0].lower(), name=name, age=30, sex="male", color="#fff", traits="plain",
                   background="b", currently="c", lifestyle="l", home="h", home_object="o", work="w",
                   start="h/o", species=species)


def test_given_a_goblin_parent_on_either_side_when_a_child_is_made_then_it_is_a_goblin():
    # Given each pairing
    human, goblin = _parent("human"), _parent("goblin", "Goblin G")
    for mother, father, species in ((human, goblin, "goblin"), (goblin, human, "goblin"), (human, human, "human"),
                                    (goblin, goblin, "goblin")):
        # When a child is made
        child = romance.generate_child_persona(mother, father, date(1347, 11, 9), random.Random(1))
        # Then it is a goblin if either parent is
        assert child.species == species


def test_given_the_lore_when_read_then_goblins_have_their_three_traits_and_the_world_says_their_children_are_goblins():
    # Given the goblin traits as a constant and Nib as the persona file writes him
    from hollowmere.agents.persona import load_personas
    nib = next(p for p in load_personas() if p.id == "nib")
    # Then Nib has all three, and the world text states the rule for the model
    assert romance.GOBLIN_TRAITS == ("horny", "lustful", "driven to breed")
    assert all(t in nib.traits for t in romance.GOBLIN_TRAITS)
    assert "child of a goblin" in prompts.WORLD_FACTS and "always a goblin" in prompts.WORLD_FACTS


async def test_given_a_model_that_writes_the_newborn_when_it_is_born_then_name_looks_and_background_are_hers(tmp_path):
    # Given a model that writes Maude
    llm = Writing()
    engine = await _engine(tmp_path, llm)
    # When Agnes and Brannoc have a child
    baby = await _birth(engine)
    # Then the written name, looks and background are the child's, with the birth date added
    p = baby.persona
    assert p.name == "Maude Hale" and "green eyes" in p.appearance and "cold garret" in p.background
    assert "born on 1347-11-09" in p.background
    # And what the code fixes stays: age, parents, species, a temperament too young to show
    assert p.age == 0 and p.parents == ["agnes", "brannoc"] and p.species == "human"
    assert p.traits.startswith("too young")


async def test_given_the_draft_prompt_when_written_then_it_shows_both_parents_and_what_is_fixed(tmp_path):
    # Given a birth
    llm = Writing()
    engine = await _engine(tmp_path, llm)
    # When the model is asked
    await _birth(engine)
    # Then both parents' looks and background and the child's fixed sex and species are in the prompt
    ask = llm.asked[0]
    agnes, brannoc = engine.agents["agnes"].persona, engine.agents["brannoc"].persona
    assert agnes.background[:40] in ask and brannoc.background[:40] in ask and agnes.appearance[:30] in ask
    assert "human" in ask and ("daughter" in ask or "son" in ask)


async def test_given_a_failing_model_when_the_child_is_born_then_it_still_comes_with_a_name_from_the_list(tmp_path):
    # Given a model that is down
    engine = await _engine(tmp_path, Writing(fail=True))
    # When a child is born
    baby = await _birth(engine)
    # Then the birth went through with the fallback persona
    assert baby.persona.name and baby.persona.name != "Maude Hale" and baby.persona.parents == ["agnes", "brannoc"]


async def test_given_a_name_already_taken_when_the_model_offers_it_then_the_fallback_name_is_kept(tmp_path):
    # Given a model that names the child after Agnes herself
    engine = await _engine(tmp_path, Writing({**DRAFT, "name": "Agnes Vell"}))
    # When a child is born
    baby = await _birth(engine)
    # Then the name is not a duplicate
    assert baby.persona.name != "Agnes Vell"
    assert sum(a.name == "Agnes Vell" for a in engine.agents.values()) == 1


async def test_given_a_birth_when_the_child_is_made_then_its_first_lasting_fact_names_its_parents(tmp_path):
    # Given a birth
    engine = await _engine(tmp_path, Writing())
    baby = await _birth(engine)
    # Then the child begins with one fact, from the data
    assert baby.history[0]["text"] == "Born on 1347-11-09 to Agnes Vell and Brannoc Hale."
    assert baby.history[0]["since"] == "1347-11-09"


async def test_given_a_first_name_already_in_town_when_the_model_offers_it_then_it_is_asked_once_more(tmp_path):
    # Given a model that first names the child Mara Hale (Mara lives in town), then Maude Hale
    llm = Writing({**DRAFT, "name": "Mara Hale"}, then=[DRAFT])
    engine = await _engine(tmp_path, llm)
    # When a child is born
    baby = await _birth(engine)
    # Then the second try is used, and the second prompt says which name was refused
    assert baby.persona.name == "Maude Hale" and len(llm.asked) == 2
    assert "Mara Hale" in llm.asked[1] and "taken" in llm.asked[1]


async def test_given_the_draft_prompt_when_written_then_it_lists_the_first_names_already_in_town(tmp_path):
    # Given a birth
    llm = Writing()
    engine = await _engine(tmp_path, llm)
    # When the model is asked
    await _birth(engine)
    # Then first names are named as taken
    assert "FIRST NAMES ALREADY IN TOWN" in llm.asked[0] and "Mara" in llm.asked[0]


async def test_given_an_existing_child_when_rewritten_with_its_name_kept_then_only_looks_and_background_change(tmp_path):
    # Given a child already named Dunstan, and a model that would have named it Maude
    llm = Writing()
    engine = await _engine(tmp_path, llm)
    dunstan = romance.generate_child_persona(
        engine.agents["agnes"].persona, engine.agents["brannoc"].persona, date(1347, 11, 7), random.Random(3)
    ).model_copy(update={"name": "Dunstan Hale"})
    # When it is written again with the name kept
    out = await romance.write_child_persona(engine.ctx, dunstan, engine.agents["agnes"].persona,
                                            engine.agents["brannoc"].persona, date(1347, 11, 7),
                                            keep_name="Dunstan Hale")
    # Then the name stays, the looks and background are the model's, and the prompt says the name is fixed
    assert out.name == "Dunstan Hale" and "green eyes" in out.appearance
    assert "Dunstan Hale was born on 1347-11-07" in out.background
    assert "already named Dunstan Hale" in llm.asked[0]
