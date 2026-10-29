"""A goblin's own child is kin, not a stranger who could give him away."""

from datetime import datetime, timedelta

from hollowmere.config import Config
from hollowmere.llm.fake import FakeLLM
from hollowmere.memory.embed import HashingEmbedder
from hollowmere.sim.setup import build_engine

NOW = datetime(1347, 11, 9, 12, 0)


async def _scene(tmp_path):
    """Isolde and Nib have a child, and the baby is in its father's keeping, wherever he goes."""
    cfg = Config()
    cfg.sim.save_path = str(tmp_path / "gob.db")
    engine = await build_engine(cfg, llm=FakeLLM(latency=0), embedder=HashingEmbedder(128))
    preg = {"mother_id": "isolde", "father_id": "nib", "start_day": "1347-11-04", "due_day": "1347-11-09",
            "resolved": False}
    engine.pregnancies.append(preg)
    await engine._give_birth(preg, NOW)
    engine.now = NOW
    baby = next(a for a in engine.agents.values() if "nib" in a.persona.parents)
    nib = engine.agents["nib"]
    wx, wy = engine.world.target_tile("fields") or (10, 10)
    for a in engine.agents.values():
        if a is not baby and a is not nib:
            a.absence = {"kind": "away", "since": NOW.isoformat(), "until": (NOW + timedelta(days=3)).isoformat(),
                         "reason": "out of the scene", "with": [a.id]}  # nobody else about
    nib.mode, nib.hiding_until, nib.path, nib.convo_id, nib.thinking = "acting", None, [], None, None
    nib.x, nib.y = wx, wy
    baby.x, baby.y = wx, wy
    baby.mode, baby.convo_id, baby.thinking = "acting", None, None
    return engine, nib, baby


def test_given_a_goblin_baby_when_asked_what_it_is_then_it_is_no_human_though_no_grown_goblin(tmp_path=None):
    from hollowmere.agents.persona import Persona
    persona = Persona(id="x", name="X", age=0, sex="male", species="goblin", color="#000", traits="t",
                      background="b", currently="c", lifestyle="l", home="goblin_den", home_object="nib_nest",
                      work="", start="goblin_den/nib_nest")
    assert persona.infant and not persona.is_goblin and not persona.is_human


async def test_given_a_goblin_baby_with_its_father_when_he_looks_around_then_it_is_not_in_sight(tmp_path):
    # Given Nib with his baby in his keeping
    engine, nib, baby = await _scene(tmp_path)
    # When he looks around
    seen = engine._in_sight(nib)
    # Then the baby is not someone who could give him away
    assert baby not in seen


async def test_given_only_his_baby_near_when_the_stealth_runs_then_the_goblin_decides_nothing(tmp_path):
    # Given Nib with only his baby near
    engine, nib, baby = await _scene(tmp_path)
    # When the stealth check runs
    await engine._stealth(NOW)
    # Then he is not asked what to do: no situation was noted, and he stays where he is
    assert nib.id not in engine._situations
    assert nib.hiding_until is None


async def test_given_his_baby_and_a_stranger_when_he_picks_cover_then_the_baby_is_not_a_threat(tmp_path):
    # Given a stranger in the open and the baby at his side
    engine, nib, baby = await _scene(tmp_path)
    ros = engine.agents["ros"]
    ros.absence = None
    ros.x, ros.y = nib.x + 3, nib.y
    # When he works out whom to keep clear of
    hostile = engine._hostile(nib, [ros])
    # Then only the stranger is in it
    assert ros in hostile and baby not in hostile


async def test_given_a_goblin_baby_beside_its_father_when_the_town_looks_then_it_glimpses_no_goblin(tmp_path):
    # Given the baby at its father's side, and a roll that would always spot a goblin
    engine, nib, baby = await _scene(tmp_path)
    nib.mode = "hiding"  # one in hiding is glimpsed, not seen plainly
    engine.rng.random = lambda: 0.0
    # When everyone perceives
    await engine._perceive(NOW)
    # Then the baby has seen no stranger of a goblin
    assert not [e for e in engine.events if e["kind"] == "sighting" and baby.id in e["agents"]]


async def test_given_a_grown_human_beside_the_goblin_when_the_town_looks_then_she_does_glimpse_him(tmp_path):
    # Given the same scene with Ros beside Nib (control: the roll really would spot him)
    engine, nib, baby = await _scene(tmp_path)
    ros = engine.agents["ros"]
    ros.absence, ros.mode, ros.convo_id, ros.thinking = None, "acting", None, None
    ros.x, ros.y = nib.x + 1, nib.y
    nib.mode = "hiding"
    engine.rng.random = lambda: 0.0
    # When everyone perceives
    await engine._perceive(NOW)
    # Then Ros has glimpsed him
    assert [e for e in engine.events if e["kind"] == "sighting" and ros.id in e["agents"]]


async def test_given_a_goblin_baby_when_the_father_is_judged_then_it_does_not_judge_him(tmp_path):
    # Given the baby and its father
    engine, nib, baby = await _scene(tmp_path)
    # When the engine asks whether the baby should weigh him up as a stranger would
    # Then it does not
    assert engine._judged(baby, nib) is None
