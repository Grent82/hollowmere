"""Family, babies and pregnancy as the town can see it."""

from datetime import datetime, timedelta

from hollowmere.agents import romance
from hollowmere.agents.kinship import Kinship
from hollowmere.config import Config
from hollowmere.llm.fake import FakeLLM
from hollowmere.memory.embed import HashingEmbedder
from hollowmere.sim.setup import build_engine

NOW = datetime(1347, 11, 5, 12)


async def _engine(tmp_path):
    cfg = Config()
    cfg.sim.save_path = str(tmp_path / "run.db")
    return await build_engine(cfg, llm=FakeLLM(latency=0), embedder=HashingEmbedder(128))


def test_kin_may_not_be_lovers_but_spouses_may():
    bond = {"stance": "close", "affinity": 6.0}
    assert romance.eligible_for_love("female", "male", bond, kin=True) is False
    assert romance.eligible_for_love("female", "male", bond, kin=False) is True


async def test_a_mother_and_her_son_are_related_in_the_engine(tmp_path):
    engine = await _engine(tmp_path)
    mother, father = engine.agents["isolde"], engine.agents["nib"]
    preg = {"mother_id": "isolde", "father_id": "nib", "start_day": "1347-11-01", "due_day": "1347-11-05",
            "resolved": False}
    engine.pregnancies.append(preg)
    await engine._give_birth(preg, NOW)
    child = next(a for a in engine.agents.values() if "isolde" in a.persona.parents)
    assert engine.ctx.kin.related("isolde", child.id) and engine.ctx.kin.related("nib", child.id)
    assert not engine.ctx.kin.related("isolde", "tobin")
    await engine._confirm_love(mother, child, NOW)  # even if both digests said so
    assert not mother.bonds[child.id].get("intimate")


async def test_a_birth_is_remembered_by_both_parents_and_named_in_their_prompt(tmp_path):
    engine = await _engine(tmp_path)
    preg = {"mother_id": "isolde", "father_id": "nib", "start_day": "1347-11-01", "due_day": "1347-11-05",
            "resolved": False}
    engine.pregnancies.append(preg)
    await engine._give_birth(preg, NOW)
    child = next(a for a in engine.agents.values() if "isolde" in a.persona.parents)
    for parent in (engine.agents["isolde"], engine.agents["nib"]):
        assert any(child.name in m.text and m.importance >= 9 for m in parent.memory.items)
    from hollowmere.agents import prompts
    line = prompts.bond_line(engine.agents["isolde"], child.id)
    assert "newborn" in line and ("son" in line or "daughter" in line)


async def test_a_baby_is_no_goblin_and_never_leaves_its_mothers_home(tmp_path):
    engine = await _engine(tmp_path)
    preg = {"mother_id": "isolde", "father_id": "nib", "start_day": "1347-11-01", "due_day": "1347-11-05",
            "resolved": False}
    engine.pregnancies.append(preg)
    await engine._give_birth(preg, NOW)
    child = next(a for a in engine.agents.values() if "isolde" in a.persona.parents)
    assert child.persona.species == "goblin" and not child.persona.is_goblin
    assert set(child.known_areas) == {"isolde_room"}
    child.path = [(child.x + 1, child.y)]
    x, y = child.x, child.y
    engine._move(NOW)
    assert (child.x, child.y) == (x, y)


async def test_pregnancy_is_known_to_the_mother_then_shows_to_everyone(tmp_path):
    engine = await _engine(tmp_path)
    mother = engine.agents["isolde"]
    engine.pregnancies.append({"mother_id": "isolde", "father_id": "nib", "start_day": "1347-11-05",
                               "due_day": "1347-11-10", "resolved": False})
    engine.now = NOW
    engine._pregnancy_phases(NOW)
    assert mother.condition == ""  # day 0: nothing yet
    later = NOW + timedelta(days=1)
    engine._pregnancy_phases(later)
    assert "with child" in mother.condition and "Nib" in mother.condition and not engine.is_showing("isolde")
    shows = NOW + timedelta(days=2)
    engine.now = shows
    engine._pregnancy_phases(shows)
    assert engine.is_showing("isolde")
    assert engine.store.events_since(NOW - timedelta(hours=1), {"showing"})


async def test_anyone_who_sees_a_showing_mother_notes_it(tmp_path):
    engine = await _engine(tmp_path)
    mother, other = engine.agents["isolde"], engine.agents["tobin"]
    engine.pregnancies.append({"mother_id": "isolde", "father_id": "nib", "start_day": "1347-11-01",
                               "due_day": "1347-11-06", "resolved": False, "showing": True})
    tile = engine.world.target_tile("tavern/bar")
    for x in (mother, other):
        x.x, x.y = tile
        x.mode = "acting"
    engine.now = NOW
    await engine._perceive(NOW)
    assert any("with child" in m.text for m in other.memory.items)
    assert not any("with child" in m.text for m in mother.memory.items if m.kind == "observation")


async def test_children_born_before_the_family_tie_get_it_and_a_birth_memory_once(tmp_path):
    engine = await _engine(tmp_path)
    preg = {"mother_id": "isolde", "father_id": "nib", "start_day": "1347-11-01", "due_day": "1347-11-05",
            "resolved": False}
    engine.pregnancies.append(preg)
    await engine._give_birth(preg, NOW)
    child = next(a for a in engine.agents.values() if "isolde" in a.persona.parents)
    for parent in (engine.agents["isolde"], engine.agents["nib"]):  # as a save from before this existed
        parent.bonds[child.id].pop("kin")
        parent.memory.items[:] = [m for m in parent.memory.items if child.name not in m.text]
    await engine.backfill_family()
    await engine.backfill_family()
    for parent in (engine.agents["isolde"], engine.agents["nib"]):
        assert parent.bonds[child.id]["kin"].startswith("your newborn")
        assert sum(child.name in m.text for m in parent.memory.items) == 1


async def test_the_mothers_family_learns_of_the_birth_but_not_the_fathers(tmp_path):
    engine = await _engine(tmp_path)
    preg = {"mother_id": "agnes", "father_id": "brannoc", "start_day": "1347-11-01", "due_day": "1347-11-05",
            "resolved": False}
    engine.pregnancies.append(preg)
    await engine._give_birth(preg, NOW)
    child = next(a for a in engine.agents.values() if len(a.persona.parents) == 2)
    isolde, mara = engine.agents["isolde"], engine.agents["mara"]  # Agnes's daughter; Brannoc's sister
    kin = isolde.bonds[child.id]["kin"]
    assert kin.startswith("your newborn") and ("brother" in kin or "sister" in kin)
    assert any(child.name in m.text for m in isolde.memory.items)
    assert child.id not in mara.bonds
    assert not any(child.name in m.text for m in mara.memory.items)


async def test_a_mothers_own_parents_are_not_told_of_a_birth_she_kept_secret(tmp_path):
    engine = await _engine(tmp_path)
    preg = {"mother_id": "isolde", "father_id": "nib", "start_day": "1347-11-01", "due_day": "1347-11-05",
            "resolved": False}
    engine.pregnancies.append(preg)
    await engine._give_birth(preg, NOW)
    child = next(a for a in engine.agents.values() if len(a.persona.parents) == 2)
    assert child.id not in engine.agents["agnes"].bonds
