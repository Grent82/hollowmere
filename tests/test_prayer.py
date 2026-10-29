from datetime import date, datetime, timedelta

from hollowmere.agents.agent import Block, Ctx, normalise_plan, parse_hhmm
from hollowmere.config import Config
from hollowmere.llm.fake import FakeLLM
from hollowmere.memory.embed import HashingEmbedder
from hollowmere.sim.setup import build_engine


def test_block_round_trips_its_god_field_through_to_dict_from_dict():
    now = datetime(1347, 11, 2, 12)
    b = Block(now, now + timedelta(hours=1), "prays", "chapel", "caedros")
    assert Block.from_dict(b.to_dict()).god == "caedros"
    assert Block.from_dict({**b.to_dict(), "god": None}).god is None


def test_normalise_plan_carries_god_through_from_raw_tuples_to_blocks():
    day = date(1347, 11, 2)
    wake = parse_hhmm(day, "06:00")
    blocks = normalise_plan(day, wake, [
        ("07:00", "08:00", "prays for mercy", "chapel", "caedros"),
        ("08:00", "12:00", "works", "smithy", None),
    ], "brannoc_room/brannoc_bed")
    assert blocks[0].god == "caedros"
    assert blocks[1].god is None


async def _engine(tmp_path, name="prayer.db"):
    cfg = Config()
    cfg.sim.save_path = str(tmp_path / name)
    return await build_engine(cfg, llm=FakeLLM(latency=0), embedder=HashingEmbedder(128))


class DivineReply:
    """Scripted answers for Agent.pray(), keyed by form; everything else fails loudly."""

    def __init__(self, form: str, text: str = "A warmth settles over you.", effect: str = "",
                duration_hours: int = 0, importance: int = 5, clarity: str | None = None):
        self.form, self.text, self.effect, self.duration_hours, self.importance = (
            form, text, effect, duration_hours, importance)
        self.clarity = clarity
        self.asked: list[str] = []

    async def structured(self, system, user, schema, *, label, temperature=None):
        assert label == "divine_answer"
        self.asked.append(user)
        reply = {"form": self.form, "text": self.text, "effect": self.effect,
                 "duration_hours": self.duration_hours, "importance": self.importance}
        if self.clarity is not None:
            reply["clarity"] = self.clarity
        return schema.model_validate(reply)


async def test_pray_observes_the_text_for_every_form_except_dream_which_is_deferred(tmp_path):
    engine = await _engine(tmp_path)
    odo, now = engine.agents["odo"], engine.now
    for form in ("sign", "omen", "coincidence", "buff"):
        before = len(odo.memory.items)
        ctx = Ctx(DivineReply(form, text=f"the {form} text"), engine.world, engine.ctx.names, {})
        await odo.pray(ctx, now, "caedros")
        assert any(m.text == f"the {form} text" for m in odo.memory.items[before:]), form
        assert odo.pending_dream is None

    ctx = Ctx(DivineReply("dream", text="a dream of the dying"), engine.world, engine.ctx.names, {})
    before = len(odo.memory.items)
    await odo.pray(ctx, now, "caedros")
    assert len(odo.memory.items) == before  # not observed yet
    assert odo.pending_dream == {"text": "a dream of the dying", "importance": 5}


async def test_pray_sets_favor_only_on_buff_and_otherwise_leaves_it_untouched(tmp_path):
    engine = await _engine(tmp_path)
    odo, now = engine.agents["odo"], engine.now
    for form in ("sign", "omen", "coincidence", "dream"):
        odo.favor = None
        ctx = Ctx(DivineReply(form), engine.world, engine.ctx.names, {})
        await odo.pray(ctx, now, "caedros")
        assert odo.favor is None, form

    ctx = Ctx(DivineReply("buff", effect="an unlooked-for calm", duration_hours=6), engine.world,
              engine.ctx.names, {})
    await odo.pray(ctx, now, "caedros")
    assert odo.favor == {"god": "caedros", "text": "an unlooked-for calm",
                         "expires": (now + timedelta(hours=6)).isoformat(), "clarity": "none"}


async def test_pray_never_writes_to_bonds_for_any_form(tmp_path):
    """The mightiness boundary: a god may only ever touch the praying agent's own state - never
    anyone's bonds, including the praying agent's own. This is a regression guard, not a smoke test."""
    import copy

    engine = await _engine(tmp_path)
    odo, now = engine.agents["odo"], engine.now
    before = copy.deepcopy({a.id: dict(a.bonds) for a in engine.agents.values()})
    for form in ("dream", "sign", "omen", "coincidence", "buff"):
        ctx = Ctx(DivineReply(form, duration_hours=1), engine.world, engine.ctx.names, {})
        await odo.pray(ctx, now, "caedros")
    after = {a.id: dict(a.bonds) for a in engine.agents.values()}
    assert before == after


async def test_pray_does_nothing_for_an_unknown_god(tmp_path):
    engine = await _engine(tmp_path)
    odo, now = engine.agents["odo"], engine.now
    before = len(odo.memory.items)

    class Boom:
        async def structured(self, *a, **kw):
            raise AssertionError("should never be called for an unknown god")

    ctx = Ctx(Boom(), engine.world, engine.ctx.names, {})
    await odo.pray(ctx, now, "thor")
    assert len(odo.memory.items) == before


async def test_plan_day_delivers_the_pending_dream_before_retrieving_memories_for_the_new_plan(tmp_path):
    engine = await _engine(tmp_path)
    odo, now = engine.agents["odo"], engine.now
    odo.pending_dream = {"text": "a dream of cold hands", "importance": 6}
    await odo.plan_day(engine.ctx, now)
    assert odo.pending_dream is None
    dream = next(m for m in odo.memory.items if m.text == "a dream of cold hands")
    assert dream.importance == 6 and dream.created == now


def test_divine_answer_clarity_defaults_to_none_and_rejects_unknown_levels():
    import pytest
    from pydantic import ValidationError

    from hollowmere.agents.schemas import DivineAnswer
    base = {"form": "sign", "text": "t", "effect": "", "duration_hours": 0, "importance": 5}
    assert DivineAnswer.model_validate(base).clarity == "none"
    assert DivineAnswer.model_validate({**base, "clarity": "certain"}).clarity == "certain"
    with pytest.raises(ValidationError):
        DivineAnswer.model_validate({**base, "clarity": "obvious"})


async def test_pray_frames_the_memory_by_how_clearly_the_mortal_recognises_the_god(tmp_path):
    engine = await _engine(tmp_path)
    odo, now = engine.agents["odo"], engine.now
    expected = {
        "none": "the candle steadied",
        "doubt": "the candle steadied Was it only chance? Or did Caedros hear me?",
        "suspected": "the candle steadied I think Caedros answered me, though I cannot be sure.",
        "certain": "Caedros answered my prayer: the candle steadied",
    }
    for clarity, memory in expected.items():
        before = len(odo.memory.items)
        ctx = Ctx(DivineReply("sign", text="the candle steadied", clarity=clarity), engine.world,
                  engine.ctx.names, {})
        await odo.pray(ctx, now, "caedros")
        assert [m.text for m in odo.memory.items[before:]] == [memory], clarity


async def test_pray_frames_a_dream_the_same_way_when_it_is_delivered(tmp_path):
    engine = await _engine(tmp_path)
    odo, now = engine.agents["odo"], engine.now
    ctx = Ctx(DivineReply("dream", text="ash and grey sky", clarity="certain"), engine.world, engine.ctx.names, {})
    await odo.pray(ctx, now, "caedros")
    assert odo.pending_dream["text"] == "Caedros answered my prayer: ash and grey sky"


async def test_pray_records_clarity_on_the_favor_and_the_prayer_event_payload(tmp_path):
    engine = await _engine(tmp_path)
    odo, now = engine.agents["odo"], engine.now
    ctx = Ctx(DivineReply("buff", effect="calm", duration_hours=2, clarity="suspected"), engine.world,
              engine.ctx.names, {})
    await odo.pray(ctx, now, "caedros")
    assert odo.favor["clarity"] == "suspected"
    assert odo.new_prayers[-1]["clarity"] == "suspected"


class StatefulReply(DivineReply):
    """A god who also eases one of the mortal's passing states, or leaves a new one."""

    def __init__(self, eases: str = "", lingers: str = "", **kw):
        super().__init__(kw.pop("form", "sign"), **kw)
        self.eases, self.lingers = eases, lingers

    async def structured(self, system, user, schema, *, label, temperament=None, temperature=None):
        self.asked.append(user)
        return schema.model_validate({
            "form": self.form, "text": self.text, "effect": self.effect, "duration_hours": self.duration_hours,
            "importance": self.importance, "eases": self.eases, "lingers": self.lingers})


async def _pray_with(engine, who, reply):
    await who.pray(Ctx(reply, engine.world, engine.ctx.names, {}), engine.now, "yevra")


async def test_given_a_grieving_woman_when_the_god_eases_her_grief_then_the_state_ends_and_others_stay(tmp_path):
    # Given Odo, exhausted and guilt-ridden
    engine = await _engine(tmp_path)
    odo = engine.agents["odo"]
    # When the god eases his guilt
    await _pray_with(engine, odo, StatefulReply(eases="guilt-ridden"))
    # Then only the exhaustion is left
    assert [s["text"] for s in odo.states] == ["exhausted"]


async def test_given_a_name_the_mortal_does_not_carry_when_the_god_eases_it_then_nothing_changes(tmp_path):
    # Given Odo with two states
    engine = await _engine(tmp_path)
    odo = engine.agents["odo"]
    # When the god eases a state he does not have
    await _pray_with(engine, odo, StatefulReply(eases="lovesick"))
    # Then both remain
    assert [s["text"] for s in odo.states] == ["exhausted", "guilt-ridden"]


async def test_given_an_answer_that_leaves_a_state_when_the_mortal_prays_then_it_begins_today(tmp_path):
    # Given Clemence, exhausted
    engine = await _engine(tmp_path)
    clemence = engine.agents["clemence"]
    # When the god leaves her a quiet certainty
    await _pray_with(engine, clemence, StatefulReply(lingers="a quiet certainty that she was heard"))
    # Then she carries it, dated today, beside the old one
    assert [s["text"] for s in clemence.states] == ["exhausted", "a quiet certainty that she was heard"]
    assert clemence.states[1]["since"] == engine.now.date().isoformat()


async def test_given_an_answer_that_eases_and_leaves_when_the_mortal_prays_then_one_makes_room_for_the_other(tmp_path):
    # Given Odo with two states and room for one more
    engine = await _engine(tmp_path)
    odo = engine.agents["odo"]
    # When the god lifts the guilt and leaves him peace
    await _pray_with(engine, odo, StatefulReply(eases="guilt-ridden", lingers="at peace with the night"))
    # Then the new state replaces the old one
    assert [s["text"] for s in odo.states] == ["exhausted", "at peace with the night"]


async def test_given_a_prayer_when_the_god_changes_states_then_no_one_elses_are_touched(tmp_path):
    # Given everyone's states
    engine = await _engine(tmp_path)
    before = {a.id: [dict(s) for s in a.states] for a in engine.agents.values() if a.id != "odo"}
    # When the god answers Odo and eases and leaves a state
    await _pray_with(engine, engine.agents["odo"], StatefulReply(eases="exhausted", lingers="rested"))
    # Then no one else's states changed
    assert before == {a.id: a.states for a in engine.agents.values() if a.id != "odo"}


async def test_given_a_praying_mortal_when_the_prayer_is_written_then_the_god_is_told_how_to_ease_a_state(tmp_path):
    # Given Odo with states
    engine = await _engine(tmp_path)
    reply = StatefulReply()
    # When he prays
    await _pray_with(engine, engine.agents["odo"], reply)
    # Then the god's prompt names his states and the two things it may do with them
    asked = reply.asked[0]
    assert "Right now: exhausted, guilt-ridden" in asked and "eases" in asked and "lingers" in asked
