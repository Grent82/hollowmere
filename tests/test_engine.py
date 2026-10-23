"""Run a full simulated day on the offline fake brain."""

from collections import Counter
from datetime import datetime, time

from hollowmere.config import Config
from hollowmere.llm.fake import FakeLLM
from hollowmere.memory.embed import HashingEmbedder
from hollowmere.sim.setup import build_engine


async def test_a_day_in_hollowmere(tmp_path):
    cfg = Config()
    cfg.sim.save_path = str(tmp_path / "run.db")
    cfg.sim.tick_seconds = 120
    engine = await build_engine(cfg, llm=FakeLLM(latency=0), embedder=HashingEmbedder(128))
    modes: Counter = Counter()
    for _ in range(24 * 30):  # one game day
        await engine.tick()
        modes.update(a.mode for a in engine.agents.values())
    kinds = Counter(e["kind"] for e in engine.events)
    assert kinds["plan"] >= len(engine.agents)
    assert kinds["chat"] > 0, kinds
    assert modes["walking"] > 0 and modes["acting"] > 0 and modes["asleep"] > 0
    for a in engine.agents.values():
        assert a.memory.items, a.id
        assert engine.world.walkable(a.x, a.y)
    engine.save()

    # resume from the save
    cfg2 = cfg.model_copy(deep=True)
    resumed = await build_engine(cfg2, llm=FakeLLM(latency=0), embedder=HashingEmbedder(128))
    assert resumed.now == engine.now
    assert {a.id: (a.x, a.y) for a in resumed.agents.values()} == {a.id: (a.x, a.y) for a in engine.agents.values()}


class Scripted(FakeLLM):
    """The fake brain, except that the goblin's choices come from a script (an exception fails the call)."""

    def __init__(self, choices):
        super().__init__(latency=0)
        self.choices, self.asked = list(choices), []

    async def structured(self, system, user, schema, *, label, temperature=None):
        if label != "goblin_choice":
            return await super().structured(system, user, schema, label=label, temperature=temperature)
        self.asked.append(user)
        action, whom = self.choices.pop(0)
        if isinstance(action, Exception):
            raise action
        return schema.model_validate({"thought": "Hm.", "action": action, "whom": whom, "reason": "bread"})


async def _goblin_scene(tmp_path, name, choices):
    engine = await _town(tmp_path, name)
    engine.ctx.llm = Scripted(choices)
    for a in engine.agents.values():
        a.x, a.y = engine.world.target_tile("tavern")  # indoors: out of everyone's sight
    wx, wy = engine.world.target_tile("square/well")

    def place(a, dx):
        a.mode, a.hiding_until, a.path, a.convo_id, a.thinking = "acting", None, [], None, None
        a.x, a.y = (wx + dx, wy) if dx is not None else engine.world.target_tile("tavern")
    return engine, place


async def test_the_goblin_asks_the_model_what_to_do(tmp_path):
    from hollowmere.llm.client import LLMError
    engine, place = await _goblin_scene(tmp_path, "gob.db", [
        ("watch", "nobody"), ("hide", "Silas Mourne"), (LLMError("down"), None),
        ("approach", "Ros Mourne"), ("flee", "Mara Hale")])
    llm = engine.ctx.llm
    nib, ros, mara, silas = (engine.agents[k] for k in ("nib", "ros", "mara", "silas"))
    place(nib, 0), place(ros, 2), place(mara, -2)
    # a group: no rule, one question to the model, which sees both and how he feels about them
    await engine._stealth(engine.now)
    assert len(llm.asked) == 1 and "Ros Mourne" in llm.asked[0] and "Mara Hale" in llm.asked[0]
    assert nib.hiding_until is None and engine.events[-1]["kind"] == "watch"
    await engine._stealth(engine.now)  # the same people: not asked again
    assert len(llm.asked) == 1
    place(silas, 3)  # someone new: asked again
    await engine._stealth(engine.now)
    assert len(llm.asked) == 2 and nib.task.activity == "slinks off to hide"
    assert "(wary" in llm.asked[1]  # his starting stance toward Silas reaches the prompt
    # the model fails: he hides, the cautious default
    engine._situations.clear()
    place(nib, 0)
    await engine._stealth(engine.now)
    assert nib.hiding_until is not None
    # one person alone, and he chooses to approach
    engine._situations.clear()
    place(nib, 0), place(mara, None), place(silas, None)
    await engine._stealth(engine.now)
    c = next(iter(engine.conversations.values()))
    assert (c.a, c.b) == (nib, ros)
    # someone comes by during the talk, and he flees: the talk is cut short
    place(mara, -2)
    await engine._stealth(engine.now)
    assert not engine.conversations and nib.task.activity == "bolts for cover"


async def test_the_goblin_does_not_instantly_re_approach_after_a_talk(tmp_path):
    engine, place = await _goblin_scene(tmp_path, "reapproach.db", [
        ("approach", "Ros Mourne"), ("approach", "Ros Mourne")])
    nib, ros = engine.agents["nib"], engine.agents["ros"]
    place(nib, 0), place(ros, 1)
    await engine._stealth(engine.now)
    c = next(iter(engine.conversations.values()))
    assert (c.a, c.b) == (nib, ros)
    engine._end_conversation(c, "goodbye")
    engine._situations.clear()  # the sighting looks fresh again a moment later
    await engine._stealth(engine.now)
    assert not engine.conversations  # he just talked to her: no instant re-approach
    assert engine.events[-1]["kind"] == "watch"


async def test_humans_who_know_the_goblin_see_him_plainly(tmp_path):
    engine, place = await _goblin_scene(tmp_path, "plain.db", [("watch", "nobody")] * 5)
    nib, ros = engine.agents["nib"], engine.agents["ros"]
    place(nib, 0), place(ros, 1)
    ros.bonds["nib"] = {"stance": "fond", "note": "a sad little thing", "talks": 1, "last": None}
    await engine._perceive(engine.now)
    assert not [e for e in engine.events if e["kind"] == "sighting"]
    assert any("Nib" in m.text for m in ros.memory.items)


async def test_a_human_who_knows_the_goblin_may_also_be_the_one_to_approach(tmp_path):
    engine = await _town(tmp_path, "known_goblin_react.db")
    isolde, nib, now = engine.agents["isolde"], engine.agents["nib"], engine.now
    isolde.bonds["nib"] = {"stance": "fond", "affinity": 4.0, "note": "a sad little thing", "talks": 1, "last": None}
    isolde.mode, isolde.hiding_until, isolde.convo_id, isolde.thinking = "acting", None, None, None
    nib.mode, nib.hiding_until, nib.convo_id, nib.thinking = "acting", None, None, None
    isolde.x, isolde.y = engine.world.target_tile("riverbank")
    nib.x, nib.y = isolde.x, isolde.y
    await engine._perceive(now)
    assert "nib" in isolde.last_react  # a known, non-hiding goblin is also a candidate for HER reaction


async def test_the_goblin_creeps_after_whom_he_chose_to_seek(tmp_path):
    from hollowmere.agents.agent import Block
    from datetime import timedelta
    engine, place = await _goblin_scene(tmp_path, "seek.db", [])
    nib, wenna, now = engine.agents["nib"], engine.agents["wenna"], engine.now
    place(nib, 0)
    wenna.mode = "acting"
    wenna.x, wenna.y = engine.world.target_tile("riverbank")
    nib.plan = [Block(now - timedelta(hours=1), now + timedelta(hours=2), "looks for Wenna", "riverbank")]
    nib.seek, nib.seek_block = "wenna", 0
    engine._seek(now)
    assert nib.path and abs(nib.path[-1][0] - wenna.x) + abs(nib.path[-1][1] - wenna.y) <= 4
    assert engine.events[-1]["kind"] == "seek"
    engine._sought["wenna"] = now  # they have met: no more creeping after her for a while
    nib.path = []
    engine._seek(now)
    assert not nib.path


async def test_a_human_who_seeks_someone_walks_right_in_instead_of_waiting_outside(tmp_path):
    from hollowmere.agents.agent import Block
    from datetime import timedelta
    engine = await _town(tmp_path, "human_seek.db")
    wat, elsbeth, now = engine.agents["wat"], engine.agents["elsbeth"], engine.now
    assert engine.world.areas["bakery"].indoor
    elsbeth.mode = "acting"
    elsbeth.x, elsbeth.y = engine.world.target_tile("bakery")
    wat.mode, wat.hiding_until, wat.convo_id = "acting", None, None
    wat.plan = [Block(now - timedelta(hours=1), now + timedelta(hours=2), "goes to find Elsbeth", "mill")]
    wat.seek, wat.seek_block = "elsbeth", 0
    engine._seek(now)
    assert wat.path  # unlike the goblin, he does not stop at the bakery door
    assert engine.events[-1]["kind"] == "visit"


async def test_a_human_stops_seeking_someone_right_after_talking_to_them(tmp_path):
    from hollowmere.agents.agent import Block
    from datetime import timedelta
    engine = await _town(tmp_path, "human_seek_cooldown.db")
    isolde, nib, now = engine.agents["isolde"], engine.agents["nib"], engine.now
    nib.x, nib.y = engine.world.target_tile("riverbank")
    nib.mode, nib.hiding_until, nib.convo_id = "acting", None, None
    isolde.mode, isolde.hiding_until, isolde.convo_id = "acting", None, None
    isolde.plan = [Block(now - timedelta(hours=1), now + timedelta(hours=2), "goes to find Nib", "tavern")]
    isolde.seek, isolde.seek_block = "nib", 0
    engine._seek(now)
    assert isolde.path  # she is on her way to him
    engine._start_conversation(isolde, nib, "to see him", now)
    c = next(iter(engine.conversations.values()))
    engine._end_conversation(c, "goodbye")  # they just talked, however briefly
    isolde.path = []
    engine._seek(now)
    assert not isolde.path  # ... so she does not immediately set off looking for him again


def test_a_conversation_that_repeats_itself_ends():
    from hollowmere.memory.stream import repeats
    said = ["I'll have them done by dusk. But I'd rather be forging ploughshares than weapons for the watch."]
    assert repeats("I'll have them done by dusk, but I'd rather be forging ploughshares than weapons for the watch.", said)
    assert not repeats("The steward wants forty spearheads by week's end.", said)


async def test_the_goblin_does_not_bolt_from_someone_still_deciding(tmp_path):
    cfg = Config()
    cfg.sim.save_path = str(tmp_path / "judge.db")
    engine = await build_engine(cfg, llm=FakeLLM(latency=0), embedder=HashingEmbedder(128))
    nib, silas = engine.agents["nib"], engine.agents["silas"]
    for a in engine.agents.values():
        a.x, a.y = engine.world.target_tile("tavern")  # everyone else out of sight
    wx, wy = engine.world.target_tile("square/well")
    for a, dx in ((nib, 0), (silas, 2)):
        a.mode, a.hiding_until, a.path, a.convo_id = "acting", None, [], None
        a.x, a.y = wx + dx, wy
    engine._judging.add(("nib", "silas"))
    engine._judged(silas, nib)("protect")
    assert ("nib", "silas") not in engine._judging and "silas" in nib.allies
    engine.ctx.llm = Scripted([])
    await engine._stealth(engine.now)  # only a protector in sight: nothing to decide
    assert engine.ctx.llm.asked == []


def test_a_sentence_lifted_from_the_other_speaker_counts_as_a_repeat():
    from hollowmere.memory.stream import repeats
    said = ["The blood on the shirt... it is not the only thing that feels wrong. "
            "The well is the source, and I fear it is poisoning us all."]
    assert repeats("The well is the source, and I fear it is poisoning us all.", said)


async def test_a_reflection_is_drawn_once_and_shown_once(tmp_path):
    cfg = Config()
    cfg.sim.save_path = str(tmp_path / "refl.db")
    engine = await build_engine(cfg, llm=FakeLLM(latency=0), embedder=HashingEmbedder(128))
    odo, now = engine.agents["odo"], engine.now
    await odo._add_reflection("The goblin is a sign of the greater evil come upon Hollowmere.", now, 7)
    await odo._add_reflection("The goblin is a sign of the greater evil come upon Hollowmere!", now, 7)
    assert sum(m.kind == "reflection" for m in odo.memory.items) == 1
    engine._background(now)
    engine._background(now)
    assert [e["kind"] for e in engine.events].count("reflection") == 1


async def test_a_speaker_keeps_private_thoughts_apart_and_knows_the_townsfolk(tmp_path):
    from hollowmere.agents import prompts
    cfg = Config()
    cfg.sim.save_path = str(tmp_path / "talk.db")
    engine = await build_engine(cfg, llm=FakeLLM(latency=0), embedder=HashingEmbedder(128))
    roderic, ctx, now = engine.agents["roderic"], engine.ctx, engine.now
    seen = await roderic.memory.add("observation", "Silas Mourne is at the graveyard, digging.", now, 2)
    secret = await roderic.memory.add("thought", "I need someone to blame for the plague.", now, 6)
    roderic.note_topic("Agnes", "ask whether the herb-wife has been seen at night")
    text = prompts.converse(roderic, ctx.names, now, "the square", "Silas Mourne", "digging", "ask about graves",
                            [], [seen, secret], topics=list(roderic.topics), known_as=ctx.known_as)
    heard, private = text.split("WHAT YOU HAVE SEEN AND HEARD")[1].split("YOUR PRIVATE THOUGHTS")
    assert seen.text in heard and secret.text not in heard and secret.text in private
    assert "Mara Hale: the weaver" in text and "Nib" not in text  # nobody in town knows the goblin
    assert "the herb-wife has been seen at night" in text  # shown before opening, to vary the topic
    later = prompts.converse(roderic, ctx.names, now, "the square", "Silas Mourne", "digging", "",
                             [("Captain Roderic Graves", "Evening, Silas.")], [], topics=list(roderic.topics),
                             known_as=ctx.known_as)
    assert "the herb-wife has been seen at night" not in later


async def test_the_report_reads_a_run(tmp_path):
    from hollowmere.report import Report
    cfg = Config()
    cfg.sim.save_path = str(tmp_path / "rep.db")
    cfg.sim.tick_seconds = 120
    engine = await build_engine(cfg, llm=FakeLLM(latency=0), embedder=HashingEmbedder(128))
    for _ in range(24 * 30):
        await engine.tick()
    engine.save()
    text = Report(engine.store, engine.world, [a.persona for a in engine.agents.values()]).build()
    for section in ("MOVEMENT", "CONVERSATIONS", "REFLECTIONS", "PLANS AND PROMISES", "THE GOBLIN", "LLM CALLS"):
        assert section in text
    assert "no samples" not in text and "no conversations recorded" not in text
    assert "plan_day" in text  # LLM counters were saved
    assert engine.store.get("llm_stats")["calls"]["plan_day"] >= len(engine.agents)


async def _town(tmp_path, name):
    cfg = Config()
    cfg.sim.save_path = str(tmp_path / name)
    engine = await build_engine(cfg, llm=FakeLLM(latency=0), embedder=HashingEmbedder(128))
    for _ in range(3):
        await engine.tick()  # everyone has a plan for the day
    return engine


async def test_only_real_promises_are_kept(tmp_path):
    engine = await _town(tmp_path, "promise.db")
    brannoc, nib, world, now = engine.agents["brannoc"], engine.agents["nib"], engine.world, engine.now
    day = now.date().isoformat()
    p = lambda t, place="graveyard", what="keep watch": {"date": day, "time": t, "activity": what,  # noqa: E731
                                                          "place": place, "with": None}
    assert not brannoc._acceptable(p("04:30"), world, now)  # already past
    assert not brannoc._acceptable(p("18:00", "tavern", "report to Brannoc"), world, now)
    assert brannoc._acceptable(p("18:00"), world, now)
    await brannoc._commit(p("18:00"), world, now)
    assert not brannoc._acceptable(p("18:00"), world, now)                 # the same again
    assert not brannoc._acceptable(p("18:30", "weaver_cottage"), world, now)  # elsewhere in the same hour
    assert brannoc._acceptable(p("20:00", "tavern"), world, now)
    assert not nib._acceptable(p("20:00", "tavern"), world, now)           # the goblin in the tavern
    assert any(b.activity == "keep watch" for b in brannoc.plan)


async def test_both_sides_of_a_meeting_end_up_at_the_same_time_and_place(tmp_path):
    from hollowmere.agents.agent import Digest
    from hollowmere.sim.engine import Conversation
    engine = await _town(tmp_path, "meet.db")
    isolde, tobin, world, now = engine.agents["isolde"], engine.agents["tobin"], engine.world, engine.now
    day = now.date().isoformat()
    hers = {"date": day, "time": "16:00", "activity": "meet Tobin", "place": "riverbank", "with": "tobin"}
    his = {"date": day, "time": "18:30", "activity": "meet Isolde", "place": "riverbank/fishing_rock", "with": "isolde"}
    await isolde._commit(hers, world, now)
    await tobin._commit(his, world, now)
    plan_before = len(tobin.plan)
    c = Conversation("c99", isolde, tobin, "see Tobin", now, transcript=[("Isolde", "Meet me at four?")])
    engine._digested(c, isolde, tobin)(Digest(together=[hers]))
    engine._digested(c, tobin, isolde)(Digest(together=[his]))
    assert (his["time"], his["place"]) == ("16:00", "riverbank")
    assert [e["kind"] for e in engine.events].count("agreement") == 1
    assert not any(b.activity == "meet Isolde" and b.start.hour == 18 for b in tobin.plan)
    for a, b in zip(tobin.plan, tobin.plan[1:]):
        assert a.end == b.start  # moving the meeting left no hole in his day
    assert len(tobin.plan) >= plan_before - 1


async def test_an_old_save_loses_its_duplicate_promises(tmp_path):
    engine = await _town(tmp_path, "old.db")
    roderic = engine.agents["roderic"]
    snap = roderic.snapshot()
    day = engine.now.date().isoformat()
    snap["commitments"] = [{"date": day, "time": "22:00", "activity": "curfew patrol", "place": "square"}] * 12 + [
        {"date": day, "time": "22:30", "activity": "curfew patrol", "place": "north_gate"},
        {"date": day, "time": "14:00", "activity": "talk with Agnes Vell", "place": "tavern"}]
    roderic.restore(snap)
    assert [(c["time"], c["place"]) for c in roderic.commitments] == [("22:00", "square"), ("14:00", "tavern")]


async def test_a_repeated_line_gets_one_more_try(tmp_path):
    from hollowmere.agents.schemas import Utterance
    from hollowmere.sim.engine import Conversation
    engine = await _town(tmp_path, "retry.db")
    ros, agnes = engine.agents["ros"], engine.agents["agnes"]
    asked = []

    async def say(ctx, now, other, purpose, transcript, avoid=""):
        asked.append(avoid)
        return Utterance(says="The river took another one." if not avoid else "Did you hear the bell?", end=False)

    agnes.say = say
    c = Conversation("c1-x", ros, agnes, "gossip", engine.now, talking=True,
                     transcript=[("Ros Mourne", "The river took another one.")])
    engine.conversations[c.id] = c
    ros.convo_id = agnes.convo_id = c.id
    await engine._turn(c, engine.now)
    assert asked == ["", "The river took another one."]
    assert c.transcript[-1] == ("Agnes Vell", "Did you hear the bell?") and not c.done


async def test_a_resumed_run_forgets_what_happened_after_the_last_autosave(tmp_path):
    engine = await _town(tmp_path, "trim.db")
    engine.save()
    saved = engine.now
    for _ in range(12):
        await engine.tick()  # recorded, but lost on a crash before the next autosave
    assert engine.store.db.execute("SELECT MAX(time) FROM samples").fetchone()[0] > saved.isoformat()
    cfg = Config()
    cfg.sim.save_path = str(tmp_path / "trim.db")
    resumed = await build_engine(cfg, llm=FakeLLM(latency=0), embedder=HashingEmbedder(128))
    assert resumed.now == saved
    latest = resumed.store.db.execute("SELECT MAX(time) FROM samples").fetchone()[0]
    assert latest is None or latest <= saved.isoformat()


async def test_one_failing_tick_does_not_stop_the_town(tmp_path, monkeypatch):
    import asyncio

    import hollowmere.sim.engine as engine_mod
    engine = await _town(tmp_path, "crash.db")
    real_tick, ticks = engine.tick, []

    async def flaky_tick():
        ticks.append(1)
        if len(ticks) == 2:
            raise RuntimeError("embeddings failed after 4 attempts")
        if len(ticks) == 4:
            engine.stop()
        await real_tick()

    async def no_wait(_):
        return None
    engine.tick = flaky_tick
    engine.cfg.tick_interval = 0
    monkeypatch.setattr(engine_mod.asyncio, "sleep", no_wait)
    await asyncio.wait_for(engine.run(), timeout=30)
    assert len(ticks) == 4  # it carried on after the failure
    assert any(e["kind"] == "error" for e in engine.events)


async def test_a_promise_an_hour_later_at_the_same_place_is_still_scheduled(tmp_path):
    from hollowmere.agents.agent import Block
    engine = await _town(tmp_path, "late.db")
    wenna, now = engine.agents["wenna"], engine.now
    day = now.date()
    at = lambda h: datetime.combine(day, time(h))  # noqa: E731
    wenna.plan = [Block(at(10), at(12), "brews", "wenna_hut"), Block(at(12), at(13), "eats", "wenna_hut"),
                  Block(at(13), at(15), "treats the sick", "almshouse/sickbeds")]
    wenna._insert_commitment({"time": "12:00", "activity": "bring the tincture", "place": "almshouse"}, day)
    assert [(b.start.hour, b.activity) for b in wenna.plan if b.start.hour == 12] == [(12, "bring the tincture")]


async def test_digest_drops_their_errands_unanswered_demands_and_invented_times(tmp_path):
    from hollowmere.agents.agent import Ctx
    engine = await _town(tmp_path, "digest.db")
    agnes, odo = engine.agents["agnes"], engine.agents["odo"]
    talk = [("Odo", "Tell Isolde to bring the chalice tomorrow at dusk."),
            ("Agnes", "Leave a silver mark in the till. I'll be at the vigil tomorrow at dusk myself.")]
    base = {"day": "tomorrow", "time": "17:30", "place": "chapel"}
    commitments = [
        {**base, "when_said": "tomorrow at dusk", "who": "them", "accepted": True, "activity": "bring the chalice"},
        {**base, "when_said": "now", "who": "them", "accepted": False, "activity": "leave a silver mark",
         "place": "tavern"},
        {**base, "when_said": "at noon", "who": "me", "accepted": True, "activity": "bless the well",
         "day": "today", "time": "12:00"},
        {**base, "when_said": "tomorrow at dusk", "who": "me", "accepted": True, "activity": "attend the vigil"},
    ]

    class Reply:  # a digest lists at most two, so the four arrive in two talks
        async def structured(self, system, user, schema, **kw):
            return schema.model_validate({"summary": "We talked.", "importance": 4, "commitments": batches.pop(0),
                                          "feeling": "fond", "view_of_them": "He keeps his word."})

    batches = [commitments[:2], commitments[2:]]
    ctx = Ctx(Reply(), engine.world, engine.ctx.names, {})
    for _ in range(2):
        await agnes.digest_chat(ctx, engine.now, odo, talk)
    assert [c["activity"] for c in agnes.commitments] == ["attend the vigil"]
    # two "fond" talks in a row nudge a neutral stance up to fond, not an instant jump
    assert agnes.bonds["odo"]["stance"] == "fond" and agnes.bonds["odo"]["talks"] == 2
    from hollowmere.agents import prompts
    line = prompts.bond_line(agnes, "odo")
    assert agnes.persona.relationships["odo"] in line and "He keeps his word. (fond, 2 talks)" in line
    snap = agnes.snapshot()
    agnes.bonds = {}
    agnes.restore(snap)
    assert agnes.bonds["odo"]["note"] == "He keeps his word."
    nib = engine.agents["nib"]
    old = nib.snapshot()
    del old["bonds"]  # a save from before bonds: the persona's starting stances are kept
    nib.restore(old)
    assert nib.bonds["roderic"]["stance"] == "fear" and nib.bonds["wenna"]["stance"] == "fond"


async def test_day_review_sets_the_agents_own_drive_and_it_survives_a_restore(tmp_path):
    from hollowmere.agents.agent import Ctx

    engine = await _town(tmp_path, "drive.db")
    mara = engine.agents["mara"]
    assert mara.drive == ""

    class Reply:
        async def structured(self, system, user, schema, **kw):
            return schema.model_validate({
                "summary": "A quiet day, but Roderic's name came up again.", "importance": 5,
                "intentions": ["speak with Roderic"],
                "drive": "to finally demand the truth from Roderic",
            })

    ctx = Ctx(Reply(), engine.world, engine.ctx.names, {})
    await mara.review_day(ctx, engine.now)
    assert mara.drive == "to finally demand the truth from Roderic"

    snap = mara.snapshot()
    mara.drive = ""
    mara.restore(snap)
    assert mara.drive == "to finally demand the truth from Roderic"


async def test_day_review_works_with_the_offline_fake_llm(tmp_path):
    """The engine's background-task wrapper swallows exceptions from review_day(), so a schema the
    fake brain cannot satisfy fails silently in sim; call it directly so a broken schema shows up here."""
    from hollowmere.agents.agent import Ctx

    engine = await _town(tmp_path, "fakedrive.db")
    mara = engine.agents["mara"]
    ctx = Ctx(engine.ctx.llm, engine.world, engine.ctx.names, {})
    await mara.review_day(ctx, engine.now)
    assert mara.drive != ""


async def test_switching_embedders_rebuilds_the_stored_vectors(tmp_path):
    class Other(HashingEmbedder):  # same vector size, different model: vectors must not be mixed
        def __init__(self):
            super().__init__(128)
            self.signature = "server-some-other-model"

        def embed_one(self, text):
            return -super().embed_one(text)

    cfg = Config()
    cfg.sim.save_path = str(tmp_path / "switch.db")
    first = await build_engine(cfg, llm=FakeLLM(latency=0), embedder=HashingEmbedder(128))
    text = first.agents["ros"].memory.items[0].text
    first.save()
    first.store.close()
    other = Other()
    second = await build_engine(cfg, llm=FakeLLM(latency=0), embedder=other)
    ros = second.agents["ros"].memory
    assert (ros._vecs[0] == other.embed_one(text)).all()  # rebuilt with the new model, not kept from the old
    assert second.store.get("embedder") == "server-some-other-model"


async def test_a_human_can_warm_to_the_goblin_over_several_talks(tmp_path):
    from hollowmere.agents import prompts
    cfg = Config()
    cfg.sim.save_path = str(tmp_path / "regard.db")
    engine = await build_engine(cfg, llm=FakeLLM(latency=0), embedder=HashingEmbedder(128))
    nib, ros = engine.agents["nib"], engine.agents["ros"]
    for step in ("drive off", "undecided", "tolerate", "tolerate", "feed"):
        engine._judged(ros, nib)(step)
    assert nib.regard["ros"]["step"] == "feed" and nib.regard["ros"]["rebuffs"] == 1
    assert "ros" not in nib.allies
    kinds = [e["kind"] for e in engine.events if "ros" in e["agents"]]
    assert kinds == ["spurn", "warm", "warm"]  # tolerating twice is not news
    assert "after the last talk" in prompts.regard_line(nib, "ros", engine.now)
    assert "Ros Mourne (feed it)" in prompts.identity(nib, engine.ctx.names)
    engine._judged(ros, nib)("protect")
    assert "ros" in nib.allies
