"""A newborn grows into a child: written on from its parents, still kept close to whoever looks after it."""

from datetime import date, datetime, timedelta

import pytest
from pydantic import ValidationError

from hollowmere.agents import prompts, romance
from hollowmere.agents.persona import Persona
from hollowmere.agents.schemas import child_growth_schema
from hollowmere.config import Config
from hollowmere.llm.client import LLMError
from hollowmere.llm.fake import FakeLLM
from hollowmere.memory.embed import HashingEmbedder
from hollowmere.sim.setup import build_engine

BORN = datetime(1347, 11, 9, 12)


class Growing(FakeLLM):
    """A model that grows the child as scripted (traits chosen from what the schema allows), or fails."""

    def __init__(self, traits=None, fail=False):
        super().__init__(latency=0)
        self.traits, self.fail, self.asked = traits, fail, []

    async def structured(self, system, user, schema, *, label, temperature=None):
        if label == "child_growth":
            self.asked.append(user)
            if self.fail:
                raise LLMError("down")
        return await super().structured(system, user, schema, label=label, temperature=temperature)

    def _childgrowth(self, js: dict, user: str) -> dict:
        items = js["properties"]["traits"]["items"]
        allowed = items.get("enum") or [items["const"]]
        low = js["properties"]["traits"]["minItems"]
        return {"traits": self.traits or allowed[:low],
                "look_now": "Walks on unsteady legs, the same dark curls now longer.",
                "currently": "a small child who toddles after a parent and babbles.",
                "lifestyle": "plays at a parent's feet, naps at noon, and is fed by hand."}


async def _engine(tmp_path, llm=None):
    cfg = Config()
    cfg.sim.save_path = str(tmp_path / "run.db")
    return await build_engine(cfg, llm=llm or Growing(), embedder=HashingEmbedder(128))


async def _birth(engine, mother_id="agnes", father_id="brannoc"):
    preg = {"mother_id": mother_id, "father_id": father_id, "start_day": "1347-11-01", "due_day": "1347-11-05",
            "resolved": False}
    engine.pregnancies.append(preg)
    await engine._give_birth(preg, BORN)
    return next(a for a in engine.agents.values() if a.persona.parents == [mother_id, father_id])


async def _grow(engine, days):
    engine._grow_children(BORN + timedelta(days=days))
    import asyncio
    await asyncio.gather(*[t for t in engine._bg.values() if not t.done()])


def _persona(age: int, species: str = "human") -> Persona:
    return Persona(id="x", name="X Y", age=age, sex="male", color="#fff", traits="t", background="b", currently="c",
                   lifestyle="l", home="h", home_object="o", work="", start="h/o", species=species)


def test_given_ages_when_asked_then_infant_child_and_dependent_are_told_apart():
    # Given a newborn, a small child, an adult
    newborn, child, adult = _persona(0), _persona(3), _persona(17)
    # Then only the newborn is an infant, a minor is dependent, and an adult is neither
    assert newborn.infant and not child.infant and not adult.infant
    assert newborn.dependent and child.dependent and not adult.dependent
    assert child.child and not newborn.child and not adult.child
    # And a goblin child is no stealthy goblin yet, a grown one is
    assert not _persona(3, "goblin").is_goblin and _persona(17, "goblin").is_goblin


def test_given_a_pool_of_parents_traits_when_the_growth_is_validated_then_only_those_words_are_allowed():
    # Given a pool from two parents
    schema = child_growth_schema(("restless", "cheeky", "weak"), 2, 3)
    base = {"look_now": "x", "currently": "y", "lifestyle": "z"}
    # Then traits from the pool pass, and an invented one or too few do not
    assert schema.model_validate({**base, "traits": ["restless", "cheeky"]})
    with pytest.raises(ValidationError):
        schema.model_validate({**base, "traits": ["restless", "wicked"]})
    with pytest.raises(ValidationError):
        schema.model_validate({**base, "traits": ["restless"]})


def test_given_two_parents_when_the_pool_is_made_then_it_holds_their_words_and_never_the_goblin_ones():
    # Given a human mother and a goblin father whose traits include the fixed three
    mother = _persona(20).model_copy(update={"traits": "restless, sharp-tongued, braver than she lets on"})
    father = _persona(30, "goblin").model_copy(update={"traits": "cheeky, horny, lustful, driven to breed, weak"})
    # When the pool for a goblin child is made
    pool = romance.growth_pool(mother, father)
    # Then it holds the words of both, without the three every goblin has anyway
    assert set(pool) == {"restless", "sharp-tongued", "braver than she lets on", "cheeky", "weak"}


async def test_given_a_human_newborn_when_four_days_have_passed_then_it_is_a_child_with_traits_from_its_parents(tmp_path):
    # Given a newborn of Agnes and Brannoc
    engine = await _engine(tmp_path)
    baby = await _birth(engine)
    # When three days pass nothing happens, and when four have passed it grows
    await _grow(engine, 3)
    assert baby.persona.infant
    await _grow(engine, 4)
    # Then it is a small child, three years old, whose traits come from the two parents' own words
    p = baby.persona
    pool = set(romance.growth_pool(engine.agents["agnes"].persona, engine.agents["brannoc"].persona))
    assert p.age == 3 and not p.infant and p.child and p.dependent
    assert p.traits and set(t.strip() for t in p.traits.split(",")) <= pool and 3 <= len(p.traits.split(",")) <= 4


async def test_given_a_goblin_newborn_when_three_days_have_passed_then_it_is_a_child_with_the_three_goblin_traits(tmp_path):
    # Given a goblin newborn of Isolde and Nib
    engine = await _engine(tmp_path)
    baby = await _birth(engine, "isolde", "nib")
    # When two days pass nothing happens, and when three have passed it grows
    await _grow(engine, 2)
    assert baby.persona.infant
    await _grow(engine, 3)
    # Then it has the goblin traits and some of its parents' own
    traits = [t.strip() for t in baby.persona.traits.split(",")]
    assert all(t in traits for t in romance.GOBLIN_TRAITS) and len(traits) >= 5 and baby.persona.species == "goblin"


async def test_given_a_growing_child_when_written_on_then_what_the_code_fixed_and_the_old_looks_stay(tmp_path):
    # Given a newborn with its looks, name and home
    engine = await _engine(tmp_path)
    baby = await _birth(engine)
    before = baby.persona
    # When it grows
    await _grow(engine, 4)
    # Then name, sex, species, parents and home are unchanged, and the new look is added after the old one
    p = baby.persona
    assert (p.name, p.sex, p.species, p.parents, p.home, p.id) == (
        before.name, before.sex, before.species, before.parents, before.home, before.id)
    assert p.appearance.startswith(before.appearance) and "unsteady legs" in p.appearance
    assert p.currently.startswith("a small child") and "naps at noon" in p.lifestyle


async def test_given_a_grown_child_when_the_parents_are_told_then_they_remember_it_and_the_bond_reads_young(tmp_path):
    # Given a newborn and its parents' bond
    engine = await _engine(tmp_path)
    baby = await _birth(engine)
    assert engine.agents["agnes"].bonds[baby.id]["kin"].startswith("your newborn")
    # When it grows
    await _grow(engine, 4)
    # Then both parents have a memory of it, and the bond now says young
    for parent in ("agnes", "brannoc"):
        assert any(baby.first in m.text and "grown" in m.text for m in engine.agents[parent].memory.items)
        assert engine.agents[parent].bonds[baby.id]["kin"].startswith("your young")
    assert any("Grew from a newborn" in f["text"] for f in baby.history)


async def test_given_a_failing_model_when_the_child_should_grow_then_it_stays_a_newborn_and_tries_the_next_day(tmp_path):
    # Given a model that is down
    llm = Growing(fail=True)
    engine = await _engine(tmp_path, llm)
    baby = await _birth(engine)
    # When the day comes and the next day comes, still down
    await _grow(engine, 4)
    await _grow(engine, 4)
    # Then it is a newborn still, and asked only once that day
    assert baby.persona.infant and len(llm.asked) == 1
    # And once the model is back, the next day it grows
    llm.fail = False
    await _grow(engine, 5)
    assert baby.persona.age == 3


async def test_given_a_child_when_its_mother_walks_off_then_it_goes_with_her_and_can_be_spoken_to(tmp_path):
    # Given a grown child of Agnes
    engine = await _engine(tmp_path)
    baby = await _birth(engine)
    await _grow(engine, 4)
    mother = engine.agents["agnes"]
    mother.x, mother.y = mother.x + 5, mother.y + 2
    # When the babies and children are carried
    engine._carry_babies()
    # Then the child is with her, and the converse prompt speaks of a small child, not of a baby
    assert (baby.x, baby.y) == (mother.x, mother.y)
    text = prompts.converse(baby, engine.ctx.names, BORN, "the bar", "Agnes", "wipes the bar", "", [], [])
    assert "short, simple sentences" in text and "only days old" not in text


async def test_given_a_child_when_love_or_leaving_is_asked_then_it_is_ruled_out_and_it_goes_with_its_mother(tmp_path):
    # Given a grown child of Agnes
    engine = await _engine(tmp_path)
    baby = await _birth(engine)
    await _grow(engine, 4)
    wat = engine.agents["wat"]
    # Then nobody is lovers with it, it cannot set out alone, and it travels with its mother
    engine.schedule_departure([baby], BORN)
    assert engine._queue == []
    assert baby in engine._dependents([engine.agents["agnes"].id])
    await engine._confirm_love(baby, wat, BORN)
    assert not wat.bonds.get(baby.id, {}).get("intimate")


class Restating(Growing):
    """A model that copies the old description into the new look, with a stray foreign word."""

    def _childgrowth(self, js: dict, user: str) -> dict:
        out = super()._childgrowth(js, user)
        out["look_now"] = "A tiny, wrinkled baby. Now she moves with jerky twitches clutching at衣物s."
        return out


async def test_given_a_look_that_repeats_the_old_description_when_the_child_grows_then_it_is_not_said_twice(tmp_path):
    # Given a newborn whose looks begin "A tiny, wrinkled baby", and a model that repeats them in the new look
    engine = await _engine(tmp_path, Restating())
    baby = await _birth(engine)
    baby.persona = baby.persona.model_copy(update={"appearance": "A tiny, wrinkled baby."})
    # When it grows
    await _grow(engine, 4)
    # Then the old words appear once, and the stray non-latin characters are gone
    assert baby.persona.appearance.count("A tiny, wrinkled baby") == 1
    assert "衣物" not in baby.persona.appearance


async def test_given_the_growth_prompt_when_written_then_it_says_the_child_is_three_and_no_longer_an_infant(tmp_path):
    # Given a newborn
    engine = await _engine(tmp_path)
    baby = await _birth(engine)
    # When the growth prompt is written
    text = prompts.child_growth(baby, engine.agents["agnes"].persona, engine.agents["brannoc"].persona, "human")
    # Then the model is told its age and not to repeat the old description
    assert "about 3 years old" in text and "no longer an infant" in text and "do not repeat" in text.lower()


async def test_given_the_childs_home_when_the_growth_prompt_is_written_then_its_day_is_set_there(tmp_path):
    # Given a newborn at its mother's home
    engine = await _engine(tmp_path)
    baby = await _birth(engine)
    home = engine.world.place_name(baby.persona.home)
    # When the growth prompt is written with that home
    text = prompts.child_growth(baby, engine.agents["agnes"].persona, engine.agents["brannoc"].persona, "human", home)
    # Then it names the home and keeps the day there, with only the parents mentioned
    assert f"lives at {home}" in text and "no other place" in text


def test_given_long_trait_phrases_when_the_pool_is_made_then_only_short_words_a_child_could_show_remain():
    # Given a mother whose traits include a trade phrase
    mother = _persona(20).model_copy(update={"traits": "shrewd, taciturn, warm to paying customers"})
    father = _persona(30).model_copy(update={"traits": "restless, stubborn, proud"})
    # When the pool is made
    pool = romance.growth_pool(mother, father)
    # Then the phrase is not among the choices
    assert "warm to paying customers" not in pool and "shrewd" in pool


async def test_given_an_old_look_that_calls_the_baby_an_infant_when_it_grows_then_the_kept_words_say_small_child(tmp_path):
    # Given a newborn described as an infant, and a new look that does not repeat it
    engine = await _engine(tmp_path)
    baby = await _birth(engine)
    baby.persona = baby.persona.model_copy(update={"appearance": "The infant has a heavy brow and honey-brown hair."})
    # When it grows
    await _grow(engine, 4)
    # Then the inherited features stay, but no one is called an infant any more
    assert "heavy brow" in baby.persona.appearance and "infant" not in baby.persona.appearance.lower()
