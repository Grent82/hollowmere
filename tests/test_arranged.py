"""A meeting two people arranged is the one reason they may talk again inside the 3-hour cooldown."""

from datetime import datetime, timedelta

from hollowmere.agents import prompts
from hollowmere.config import Config
from hollowmere.llm.fake import FakeLLM
from hollowmere.memory.embed import HashingEmbedder
from hollowmere.sim.setup import build_engine

NOW = datetime(1347, 11, 5, 21, 30)


async def _engine(tmp_path):
    cfg = Config()
    cfg.sim.save_path = str(tmp_path / "run.db")
    return await build_engine(cfg, llm=FakeLLM(latency=0), embedder=HashingEmbedder(128))


def _arrange(a, b, at: datetime, activity="meet at the weeping stones", place="standing_stones"):
    a.mode = b.mode = "acting"
    entry = {"date": at.date().isoformat(), "time": at.strftime("%H:%M"), "activity": activity, "place": place}
    a.commitments.append({**entry, "with": b.id})
    b.commitments.append({**entry, "with": a.id})


def _just_talked(a, b, hours_ago: float = 0.5):
    a.last_chat[b.id] = b.last_chat[a.id] = NOW - timedelta(hours=hours_ago)


async def test_without_an_arrangement_the_cooldown_still_applies(tmp_path):
    engine = await _engine(tmp_path)
    a, b = engine.agents["isolde"], engine.agents["tobin"]
    a.mode = b.mode = "acting"
    _just_talked(a, b)
    assert not engine._may_react(a, b, NOW)


async def test_an_arrangement_due_about_now_lifts_the_cooldown_for_that_pair_only(tmp_path):
    engine = await _engine(tmp_path)
    a, b, c = engine.agents["isolde"], engine.agents["tobin"], engine.agents["nib"]
    c.mode = "acting"
    _arrange(a, b, NOW)
    _just_talked(a, b)
    a.last_chat[c.id] = c.last_chat[a.id] = NOW - timedelta(hours=0.5)
    assert engine._may_react(a, b, NOW)
    assert engine._may_react(b, a, NOW)
    assert not engine._may_react(a, c, NOW)  # an arrangement with Tobin says nothing about Nib


async def test_an_arrangement_far_from_now_does_not_lift_it(tmp_path):
    engine = await _engine(tmp_path)
    a, b = engine.agents["isolde"], engine.agents["tobin"]
    _arrange(a, b, NOW + timedelta(hours=2))
    _just_talked(a, b)
    assert not engine._may_react(a, b, NOW)
    assert engine._may_react(a, b, NOW + timedelta(hours=1, minutes=45))  # inside the window of its hour


async def test_starting_the_arranged_talk_uses_the_arrangement_up(tmp_path):
    engine = await _engine(tmp_path)
    a, b = engine.agents["isolde"], engine.agents["tobin"]
    _arrange(a, b, NOW)
    _just_talked(a, b)
    engine.now = NOW
    engine._start_conversation(a, b, "to keep our arrangement", NOW)
    assert a.convo_id is not None and a.convo_id == b.convo_id
    assert all(c.get("met") for c in a.commitments + b.commitments)
    # the talk ends; moments later nothing but the plain cooldown stands between them
    engine._end_conversation(engine.conversations[a.convo_id], "goodbye")
    assert not engine._may_react(a, b, NOW + timedelta(minutes=5))


async def test_a_second_talk_cannot_be_opened_on_the_same_arrangement(tmp_path):
    engine = await _engine(tmp_path)
    a, b = engine.agents["isolde"], engine.agents["tobin"]
    _arrange(a, b, NOW)
    _just_talked(a, b)
    engine.now = NOW
    engine._start_conversation(a, b, "first", NOW)
    first = a.convo_id
    engine._end_conversation(engine.conversations[first], "goodbye")
    engine._start_conversation(a, b, "second", NOW + timedelta(minutes=10))
    assert a.convo_id is None


async def test_the_reaction_prompt_names_the_arrangement(tmp_path):
    engine = await _engine(tmp_path)
    a, b = engine.agents["isolde"], engine.agents["tobin"]
    text = prompts.react(a, engine.ctx.names, NOW, "the Weeping Stones", "waits", b.name, "waits", [],
                         arranged="meet at the weeping stones")
    assert "arranged" in text and "meet at the weeping stones" in text
    assert "arranged" not in prompts.react(a, engine.ctx.names, NOW, "the Weeping Stones", "waits", b.name,
                                           "waits", [])


async def test_the_arranged_reaction_hands_the_arrangement_to_the_prompt(tmp_path):
    engine = await _engine(tmp_path)
    a, b = engine.agents["isolde"], engine.agents["tobin"]
    _arrange(a, b, NOW)
    seen: list[str] = []
    real = engine.ctx.llm.structured

    async def spy(system, user, schema, **kw):
        seen.append(user)
        return await real(system, user, schema, **kw)

    engine.ctx.llm.structured = spy
    await engine._react(a, b, NOW)
    assert any("meet at the weeping stones" in u for u in seen)


async def test_a_meeting_just_kept_does_not_block_the_next_promise_of_the_hour(tmp_path):
    """Regression (found replaying Isolde and Nib): the arrangement a talk had just cashed in stayed on the
    list for two more hours and, under 'one promise per hour', refused every new 'let us go on from here'."""
    engine = await _engine(tmp_path)
    a, b = engine.agents["isolde"], engine.agents["tobin"]
    _arrange(a, b, NOW)
    _just_talked(a, b)
    engine.now = NOW
    engine._start_conversation(a, b, "to keep our arrangement", NOW)
    follow_up = {"date": NOW.date().isoformat(), "time": (NOW + timedelta(minutes=40)).strftime("%H:%M"),
                 "activity": "walk on together", "place": "riverbank", "with": b.id}
    assert a._acceptable(follow_up, engine.world, NOW + timedelta(minutes=25))
    # an unkept promise in the same hour still blocks, as before
    other = {**follow_up, "activity": "something else", "met": False}
    a.commitments.append(other)
    assert not a._acceptable({**follow_up, "time": (NOW + timedelta(minutes=50)).strftime("%H:%M")},
                             engine.world, NOW + timedelta(minutes=25))


# ---------------------------------------------------------------- the goblin's side of it

from types import SimpleNamespace  # noqa: E402


def _goblin_approaches(engine, nib, target):
    """Nib's own choice says: approach `target`."""
    async def choice(ctx, now, seen, talking_with, hidden, cover=None):
        return (SimpleNamespace(action="approach", whom=target.name, thought="she is near", reason="to see her"),
                {target.name: target})
    nib.goblin_choice = choice


async def test_the_goblin_may_approach_inside_the_cooldown_for_an_arranged_meeting(tmp_path):
    """The human path got this exception, the goblin's own approach still demanded the full cooldown: a meeting
    arranged with Nib stayed silent."""
    engine = await _engine(tmp_path)
    nib, isolde = engine.agents["nib"], engine.agents["isolde"]
    _arrange(nib, isolde, NOW, "meet at the weeping stones", "standing_stones")
    _just_talked(nib, isolde)
    _goblin_approaches(engine, nib, isolde)
    engine.now = NOW
    await engine._goblin_chooses(nib, [isolde], None, None, NOW)
    assert nib.convo_id is not None and nib.convo_id == isolde.convo_id


async def test_the_goblin_still_waits_out_the_cooldown_without_an_arrangement(tmp_path):
    engine = await _engine(tmp_path)
    nib, isolde = engine.agents["nib"], engine.agents["isolde"]
    nib.mode = isolde.mode = "acting"
    _just_talked(nib, isolde)
    _goblin_approaches(engine, nib, isolde)
    engine.now = NOW
    await engine._goblin_chooses(nib, [isolde], None, None, NOW)
    assert nib.convo_id is None


def test_where_the_goblin_can_go():
    from hollowmere.agents.persona import Persona
    from hollowmere.world.town import build_town
    world = build_town()
    base = dict(age=30, color="#fff", traits="t", background="b", currently="c", lifestyle="l", work="w",
                start="x/y", home_object="ho")
    from hollowmere.agents.agent import Agent
    nib = Agent(Persona(id="nib", name="Nib", sex="male", species="goblin", home="goblin_den", **{**base, "work": "ruin"}),
                None, world)
    human = Agent(Persona(id="h", name="H", sex="female", home="tavern", **base), None, world)
    assert nib.can_go("riverbank/reeds", world) and nib.can_go("standing_stones", world)  # outdoors
    assert nib.can_go("goblin_den/nib_nest", world) and nib.can_go("ruin/idols", world)  # his own
    assert not nib.can_go("tavern/bar", world) and not nib.can_go("mill", world)  # indoors that are not his
    assert human.can_go("tavern/bar", world) and human.can_go("mill", world)


async def test_a_human_is_only_offered_meeting_places_the_goblin_can_go_to(tmp_path):
    """Isolde proposed a meeting at the mill, Nib's side refused it (he does not walk indoors), and she kept a
    promise that he did not: the two ended up planning past each other."""
    import json
    engine = await _engine(tmp_path)
    nib, isolde = engine.agents["nib"], engine.agents["isolde"]
    isolde.learn_area("goblin_den", engine.world)
    seen = []
    fake = engine.ctx.llm

    class Spy:
        stats = fake.stats

        async def structured(self, system, user, schema, *, label, temperature=None):
            if label == "digest":
                seen.append(json.dumps(schema.model_json_schema()))
            return await fake.structured(system, user, schema, label=label, temperature=temperature)

    engine.ctx.llm = Spy()
    await isolde.digest_chat(engine.ctx, NOW, nib, [(isolde.name, "Meet me?"), (nib.name, "Aye.")])
    assert seen
    offered = seen[0]
    assert '"tavern' not in offered and '"mill' not in offered
    assert '"riverbank' in offered and '"goblin_den' in offered
    # a talk between two humans still offers the whole town
    seen.clear()
    await isolde.digest_chat(engine.ctx, NOW, engine.agents["tobin"], [(isolde.name, "Meet me?"), ("Tobin", "Aye.")])
    assert '"tavern' in seen[0] and '"mill' in seen[0]
