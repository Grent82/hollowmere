"""Who is family to whom, and who is expecting: what the detail panel shows."""

import random
from datetime import date, timedelta

from hollowmere.agents import romance
from hollowmere.agents.kinship import Kinship
from hollowmere.agents.persona import Persona, load_personas
from hollowmere.config import Config
from hollowmere.llm.fake import FakeLLM
from hollowmere.memory.embed import HashingEmbedder
from hollowmere.sim.setup import build_engine


def _p(pid, sex, parents=(), **kin):
    return Persona(id=pid, name=pid.title(), age=30, sex=sex, color="#fff", traits="t", background="b",
                   currently="c", lifestyle="l", home="h", home_object="ho", work="w", start="h/ho",
                   parents=list(parents), **kin)


def _kin(*personas):
    return Kinship({p.id: p for p in personas})


def test_the_towns_own_families_read_correctly_from_both_sides():
    k = Kinship({p.id: p for p in load_personas()})  # the ties are written into personas.json, one side each
    assert k.what_is("agnes", "isolde") == "mother"      # Agnes is Isolde's mother
    assert k.what_is("isolde", "agnes") == "daughter"
    assert k.what_is("brannoc", "mara") == "brother"
    assert k.what_is("mara", "brannoc") == "sister"
    assert k.what_is("silas", "ros") == "brother"
    assert k.what_is("ros", "silas") == "sister"
    assert k.what_is("odo", "clemence") == "uncle"
    assert k.what_is("clemence", "odo") == "niece"
    assert k.what_is("roderic", "elsbeth") == "husband"
    assert k.what_is("elsbeth", "roderic") == "wife"


def test_people_who_are_not_family_have_no_label():
    k = _kin(_p("a", "male"), _p("b", "female"))
    assert k.what_is("a", "b") is None
    assert k.what_is("a", "a") is None
    town = Kinship({p.id: p for p in load_personas()})
    assert town.what_is("brannoc", "tobin") is None  # his apprentice, not his kin


def test_children_know_their_parents_and_each_other():
    mum, dad = _p("mum", "female"), _p("dad", "male")
    son, daughter = _p("son", "male", ["mum", "dad"]), _p("daughter", "female", ["mum", "dad"])
    k = _kin(mum, dad, son, daughter)
    assert k.what_is("mum", "son") == "mother" and k.what_is("dad", "daughter") == "father"
    assert k.what_is("son", "mum") == "son" and k.what_is("daughter", "dad") == "daughter"
    assert k.what_is("son", "daughter") == "brother" and k.what_is("daughter", "son") == "sister"


def test_half_siblings_through_one_parent_are_still_siblings():
    mum, dad, other = _p("mum", "female"), _p("dad", "male"), _p("other", "male")
    a, b = _p("a", "female", ["mum", "dad"]), _p("b", "male", ["mum", "other"])
    assert _kin(mum, dad, other, a, b).what_is("a", "b") == "sister"


def test_grandparents_and_grandchildren():
    gran, mum, dad, kid = _p("gran", "female"), _p("mum", "female", ["gran"]), _p("dad", "male"), _p("kid", "male", ["mum", "dad"])
    k = _kin(gran, mum, dad, kid)
    assert k.what_is("gran", "kid") == "grandmother"
    assert k.what_is("kid", "gran") == "grandson"


def test_an_aunt_or_uncle_is_a_parents_sibling():
    mum, dad = _p("mum", "female"), _p("dad", "male", siblings=["uncle_bob"])
    kid = _p("kid", "female", ["mum", "dad"])
    k = _kin(mum, dad, kid, _p("uncle_bob", "male"))
    assert k.what_is("uncle_bob", "kid") == "uncle"
    assert k.what_is("kid", "uncle_bob") == "niece"


def test_a_newborn_is_recorded_with_both_parents():
    mother = _p("mum", "female")
    father = _p("dad", "male")
    child = romance.generate_child_persona(mother, father, date(1347, 11, 8), random.Random(1))
    assert child.parents == ["mum", "dad"]


async def _engine(tmp_path):
    cfg = Config()
    cfg.sim.save_path = str(tmp_path / "run.db")
    return await build_engine(cfg, llm=FakeLLM(latency=0), embedder=HashingEmbedder(128))


async def test_the_detail_panel_labels_relatives_in_the_bond_list(tmp_path):
    engine = await _engine(tmp_path)
    bonds = {b["id"]: b for b in engine.agent_detail("isolde")["bonds"]}
    assert bonds["agnes"]["kin"] == "mother"
    assert bonds["tobin"]["kin"] is None
    bonds = {b["id"]: b for b in engine.agent_detail("agnes")["bonds"]}
    assert bonds["isolde"]["kin"] == "daughter"
    assert {b["id"]: b["kin"] for b in engine.agent_detail("clemence")["bonds"]}["odo"] == "uncle"


async def test_a_mother_to_be_shows_the_pregnancy_and_nobody_else_does(tmp_path):
    engine = await _engine(tmp_path)
    today = engine.now.date()
    engine.pregnancies.append({"mother_id": "isolde", "father_id": "nib", "start_day": today.isoformat(),
                               "due_day": (today + timedelta(days=5)).isoformat(), "resolved": False})
    p = engine.agent_detail("isolde")["pregnancy"]
    assert p["father"] == "Nib" and p["due"] == (today + timedelta(days=5)).isoformat() and p["days_left"] == 5
    assert engine.agent_detail("nib")["pregnancy"] is None
    assert engine.agent_detail("tobin")["pregnancy"] is None


async def test_a_pregnancy_that_has_ended_is_no_longer_shown(tmp_path):
    engine = await _engine(tmp_path)
    today = engine.now.date()
    engine.pregnancies.append({"mother_id": "isolde", "father_id": "nib", "start_day": today.isoformat(),
                               "due_day": today.isoformat(), "resolved": True})
    assert engine.agent_detail("isolde")["pregnancy"] is None


def test_a_tie_written_on_one_persona_reads_from_both_sides():
    k = _kin(_p("a", "male", spouse="b"), _p("b", "female"), _p("c", "female", siblings=["a"]),
             _p("d", "female", aunt_or_uncle=["a"]))
    assert k.what_is("a", "b") == "husband" and k.what_is("b", "a") == "wife"
    assert k.what_is("a", "c") == "brother" and k.what_is("c", "a") == "sister"
    assert k.what_is("a", "d") == "uncle" and k.what_is("d", "a") == "niece"
