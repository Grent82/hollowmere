import asyncio
import random
from datetime import date, timedelta

from hollowmere.agents import romance
from hollowmere.agents.agent import Ctx, Digest, day_key
from hollowmere.agents.persona import Persona, starting_affinity
from hollowmere.config import Config
from hollowmere.llm.fake import FakeLLM
from hollowmere.memory.embed import HashingEmbedder
from hollowmere.sim.engine import Conversation
from hollowmere.sim.setup import build_engine


def _persona(**over):
    base = dict(id="x", name="X Y", age=30, sex="female", color="#fff", traits="t",
               background="b", currently="c", lifestyle="l", home="h", home_object="ho",
               work="w", start="h/ho")
    base.update(over)
    return Persona(**base)


def test_eligible_for_love_requires_opposite_sex():
    assert romance.eligible_for_love("male", "male", {"stance": "close", "affinity": 6.0}) is False


def test_eligible_for_love_requires_fond_or_above():
    assert romance.eligible_for_love("male", "female", {"stance": "neutral", "affinity": 3.0}) is False
    assert romance.eligible_for_love("male", "female", {"stance": "fond", "affinity": 4.0}) is True


def test_eligible_for_love_ignores_the_affinity_gate_once_already_intimate():
    assert romance.eligible_for_love(
        "male", "female", {"stance": "wary", "affinity": 2.0, "intimate": True}) is True


def test_eligible_for_love_with_no_bond_yet_is_false():
    assert romance.eligible_for_love("male", "female", None) is False


def test_generate_child_persona_inherits_fathers_species_and_mothers_home():
    mother = _persona(id="mum", name="Mum Hale", sex="female", home="mum_home", home_object="mum_bed")
    father = _persona(id="dad", name="Grik", sex="male", species="goblin")
    child = romance.generate_child_persona(mother, father, date(1347, 6, 1), random.Random(1))
    assert child.species == "goblin"
    assert child.sex in ("male", "female")
    assert child.age == 0
    assert child.home == "mum_home" and child.home_object == "mum_bed"
    assert child.start == "mum_home/mum_bed"
    assert child.relationships == {"mum": "my mother", "dad": "my father"}
    assert child.stances == {"mum": "fond", "dad": "fond"}


def test_generate_child_persona_is_fully_deterministic_given_the_same_rng_state():
    # Real callers pass the engine's one shared, ever-advancing rng, so two births never actually
    # draw from the same state - but given literally identical state, the result must match exactly.
    mother = _persona(id="mum", sex="female")
    father = _persona(id="dad", sex="male")
    a = romance.generate_child_persona(mother, father, date(1347, 6, 1), random.Random(42))
    b = romance.generate_child_persona(mother, father, date(1347, 6, 1), random.Random(42))
    assert a.name == b.name and a.sex == b.sex and a.id == b.id

    c = romance.generate_child_persona(mother, father, date(1347, 6, 1), random.Random(43))
    assert c.id != a.id  # different rng state: different id


class _StubAgent:
    """Just enough of Agent's shape (id + bonds) for recognize_child, with no engine/world/memory."""
    def __init__(self, id):
        self.id = id
        self.bonds: dict[str, dict] = {}


def test_recognize_child_sets_a_fond_bond_on_both_parents():
    mother, father, child = _StubAgent("mum"), _StubAgent("dad"), _StubAgent("kid")
    romance.recognize_child(mother, father, child)
    assert mother.bonds["kid"]["stance"] == "fond" and mother.bonds["kid"]["affinity"] == starting_affinity("fond")
    assert father.bonds["kid"]["stance"] == "fond" and father.bonds["kid"]["affinity"] == starting_affinity("fond")


class LoveScripted(FakeLLM):
    """Reports made_love for every digest call it's asked, when the field is offered at all."""

    def __init__(self, made_love: bool):
        super().__init__(latency=0)
        self.made_love = made_love
        self.offered: list[bool] = []

    async def structured(self, system, user, schema, *, label, temperature=None):
        if label != "digest":
            return await super().structured(system, user, schema, label=label, temperature=temperature)
        js = schema.model_json_schema()
        offered = "made_love" in js.get("properties", {})
        self.offered.append(offered)
        data = {"summary": "we talked", "importance": 5, "feeling": "fond",
                "view_of_them": "warmly", "commitments": []}
        if offered:
            data["made_love"] = self.made_love
        return schema.model_validate(data)


async def _pair(tmp_path, name="romance.db"):
    cfg = Config()
    cfg.sim.save_path = str(tmp_path / name)
    engine = await build_engine(cfg, llm=FakeLLM(latency=0), embedder=HashingEmbedder(128))
    # roderic (male) and isolde (female) are both in the default roster
    return engine, engine.agents["roderic"], engine.agents["isolde"]


def _conv(a, b, now):
    return Conversation(id="c1", a=a, b=b, purpose="", started=now)


async def test_digest_chat_does_not_offer_made_love_below_fond(tmp_path):
    engine, a, b = await _pair(tmp_path)
    a.bonds[b.id] = {"stance": "neutral", "affinity": 3.0, "note": "", "talks": 0, "last": None}
    llm = LoveScripted(made_love=True)
    result = await a.digest_chat(Ctx(llm, engine.world, engine.ctx.names, {}), engine.now, b, [("A", "hi")])
    assert llm.offered == [False]
    assert result.made_love is False


async def test_digest_chat_offers_and_reports_made_love_at_fond_and_above(tmp_path):
    engine, a, b = await _pair(tmp_path)
    a.bonds[b.id] = {"stance": "fond", "affinity": 4.0, "note": "", "talks": 0, "last": None}
    llm = LoveScripted(made_love=True)
    result = await a.digest_chat(Ctx(llm, engine.world, engine.ctx.names, {}), engine.now, b, [("A", "hi")])
    assert llm.offered == [True]
    assert result.made_love is True


async def test_settle_confirms_love_only_when_both_sides_report_it(tmp_path):
    engine, a, b = await _pair(tmp_path)
    a.bonds[b.id] = {"stance": "fond", "affinity": 4.0, "note": "", "talks": 0, "last": None}
    b.bonds[a.id] = {"stance": "fond", "affinity": 4.0, "note": "", "talks": 0, "last": None}
    c = _conv(a, b, engine.now)
    engine._settle(c, {a.id: Digest(made_love=True), b.id: Digest(made_love=False)})
    await asyncio.sleep(0.05)  # let any spawned background task run
    assert a.bonds[b.id].get("intimate") is None
    assert engine.pregnancies == []


async def test_settle_confirms_love_and_bumps_the_lower_side_to_close(tmp_path):
    engine, a, b = await _pair(tmp_path)
    a.bonds[b.id] = {"stance": "close", "affinity": 6.0, "note": "", "talks": 0, "last": None}
    b.bonds[a.id] = {"stance": "fond", "affinity": 4.0, "note": "", "talks": 0, "last": None}
    c = _conv(a, b, engine.now)
    engine._settle(c, {a.id: Digest(made_love=True), b.id: Digest(made_love=True)})
    await asyncio.sleep(0.05)
    assert a.bonds[b.id]["stance"] == "close"  # already there: unaffected
    assert b.bonds[a.id]["stance"] == "close"  # was fond: bumped up
    assert a.bonds[b.id]["intimate"] is True and b.bonds[a.id]["intimate"] is True
    assert any("made love" in m.text for m in a.memory.items)
    assert len(engine.pregnancies) == 1
    preg = engine.pregnancies[0]
    assert preg["mother_id"] == b.id and preg["father_id"] == a.id  # roderic is male, isolde the mother
    assert preg["resolved"] is False
    assert preg["due_day"] == (day_key(engine.now) + timedelta(days=engine.cfg.gestation_days)).isoformat()


async def test_settle_does_not_start_a_second_pregnancy_for_an_already_pregnant_mother(tmp_path):
    engine, a, b = await _pair(tmp_path)
    for x, y in ((a, b), (b, a)):
        x.bonds[y.id] = {"stance": "close", "affinity": 6.0, "note": "", "talks": 0, "last": None}
    engine.pregnancies.append({"mother_id": b.id, "father_id": a.id, "start_day": "1347-11-01",
                               "due_day": "1347-11-06", "resolved": False})
    c = _conv(a, b, engine.now)
    engine._settle(c, {a.id: Digest(made_love=True), b.id: Digest(made_love=True)})
    await asyncio.sleep(0.05)
    assert len(engine.pregnancies) == 1  # not two


async def test_a_due_pregnancy_resolves_into_a_real_agent_once(tmp_path):
    engine, a, b = await _pair(tmp_path)
    mother, father = b, a
    due = engine.now.date() - timedelta(days=1)
    engine.pregnancies.append({"mother_id": mother.id, "father_id": father.id,
                               "start_day": (due - timedelta(days=engine.cfg.gestation_days)).isoformat(),
                               "due_day": due.isoformat(), "resolved": False})
    before = set(engine.agents)
    engine._background(engine.now)
    await asyncio.sleep(0.1)  # let the spawned birth task run
    new_ids = set(engine.agents) - before
    assert len(new_ids) == 1, new_ids
    child = engine.agents[next(iter(new_ids))]
    assert child.persona.species == father.persona.species
    assert mother.bonds[child.id]["stance"] == "fond" and father.bonds[child.id]["stance"] == "fond"
    assert child.bonds[mother.id]["stance"] == "fond" and child.bonds[father.id]["stance"] == "fond"
    assert engine.pregnancies[0]["resolved"] is True
    assert any(e["kind"] == "birth" for e in engine.events)

    # firing _background again must not create a second child from the same, now-resolved pregnancy
    engine._background(engine.now)
    await asyncio.sleep(0.1)
    assert set(engine.agents) - before == new_ids


async def test_a_newborn_survives_a_restart(tmp_path):
    engine, a, b = await _pair(tmp_path, name="restart.db")
    mother, father = b, a
    due = engine.now.date() - timedelta(days=1)
    engine.pregnancies.append({"mother_id": mother.id, "father_id": father.id,
                               "start_day": due.isoformat(), "due_day": due.isoformat(), "resolved": False})
    before = set(engine.agents)
    engine._background(engine.now)
    await asyncio.sleep(0.1)
    child_id = next(iter(set(engine.agents) - before))
    engine.save()

    cfg2 = Config()
    cfg2.sim.save_path = engine.cfg.save_path
    resumed = await build_engine(cfg2, llm=FakeLLM(latency=0), embedder=HashingEmbedder(128))
    assert child_id in resumed.agents
    assert resumed.agents[child_id].persona.name == engine.agents[child_id].persona.name
