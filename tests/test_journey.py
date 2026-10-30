"""What befalls those who set out from Hollowmere is told by the model; the code only bounds it and carries it out."""

from datetime import datetime, timedelta

from hollowmere.config import Config
from hollowmere.llm.client import LLMError
from hollowmere.llm.fake import FakeLLM
from hollowmere.memory.embed import HashingEmbedder
from hollowmere.sim.setup import build_engine

NOW = datetime(1347, 11, 5, 12)
EVENING = datetime(1347, 11, 5, 22, 0)


class Road:
    """A model that tells the road as scripted, one answer per call, and leaves every other call to the fake."""

    def __init__(self, *tellings):
        self.tellings, self.fake, self.asked = list(tellings), FakeLLM(latency=0), []
        self.stats = self.fake.stats

    async def structured(self, system, user, schema, **kw):
        if schema.__name__ != "Journey":
            return await self.fake.structured(system, user, schema, **kw)
        self.asked.append(user)
        told = self.tellings.pop(0)
        if isinstance(told, Exception):
            raise told
        return schema.model_validate(told)


def fate(who, how="returns", days=2, memory="", state=""):
    return {"who": who, "fate": how, "after_days": days, "memory": memory, "state": state}


def telling(*fates, story="the ford had flooded", rumor="", children=""):
    return {"story": story, "fates": list(fates), "rumor": rumor, "children": children}


async def _engine(tmp_path, llm):
    cfg = Config()
    cfg.sim.save_path = str(tmp_path / "run.db")
    engine = await build_engine(cfg, llm=llm, embedder=HashingEmbedder(128))
    engine.now = NOW
    return engine


async def _set_out(engine, *who):
    engine.schedule_departure([engine.agents[w] for w in who], NOW)
    await engine._absences(EVENING)


async def _baby(engine):
    preg = {"mother_id": "isolde", "father_id": "tobin", "start_day": "1347-11-01", "due_day": "1347-11-05",
            "resolved": False}
    engine.pregnancies.append(preg)
    await engine._give_birth(preg, NOW)
    return next(a for a in engine.agents.values() if "isolde" in a.persona.parents)


def _until(agent):
    return datetime.fromisoformat(agent.absence["until"])


async def test_given_a_departure_when_the_model_tells_the_road_then_story_and_days_are_theirs(tmp_path):
    # Given two who set out, and a model that sends Isolde home in five days and Tobin on, to be looked at in nine
    road = Road(telling(fate("Isolde", days=5), fate("Tobin", "stays_away", days=9)))
    engine = await _engine(tmp_path, road)
    # When they set out
    await _set_out(engine, "isolde", "tobin")
    # Then each has the day the model gave, and the story is the reason
    isolde, tobin = engine.agents["isolde"], engine.agents["tobin"]
    assert _until(isolde) == datetime(1347, 11, 10, 17) and _until(tobin) == datetime(1347, 11, 14, 17)
    assert isolde.absence["reason"] == "the ford had flooded" and tobin.absence.get("stays")
    assert not isolde.absence.get("stays")


async def test_given_two_fates_when_they_differ_then_the_pair_is_split_and_the_baby_follows_its_mother(tmp_path):
    # Given a baby of Isolde and Tobin, who part on the road
    road = Road(telling(fate("Isolde", days=5), fate("Tobin", "stays_away", days=9)))
    engine = await _engine(tmp_path, road)
    child = await _baby(engine)
    # When they set out
    await _set_out(engine, "isolde", "tobin")
    # Then the baby is with Isolde, and Tobin is alone
    assert engine.agents["isolde"].absence["with"] == ["isolde", child.id]
    assert child.absence["with"] == ["isolde", child.id] and _until(child) == _until(engine.agents["isolde"])
    assert engine.agents["tobin"].absence["with"] == ["tobin"]


async def test_given_a_failed_call_when_they_set_out_then_the_old_chance_decides(tmp_path):
    # Given a model that cannot be reached
    road = Road(LLMError("down"))
    engine = await _engine(tmp_path, road)
    # When they set out
    await _set_out(engine, "isolde", "tobin")
    # Then they are away for the days the config allows, with a reason from the old list
    for who in ("isolde", "tobin"):
        a = engine.agents[who]
        assert a.is_away and 1 <= (_until(a).date() - EVENING.date()).days <= engine.cfg.absence_max_days
        assert a.absence["reason"] and not a.absence.get("stays")


async def test_given_the_model_leaves_one_out_when_it_answers_then_that_one_goes_by_the_old_chance(tmp_path):
    # Given a model that tells nothing of Tobin
    road = Road(telling(fate("Isolde", days=3)))
    engine = await _engine(tmp_path, road)
    # When they set out
    await _set_out(engine, "isolde", "tobin")
    # Then Tobin still goes by the old chance and nobody is lost
    assert engine.agents["tobin"].is_away and _until(engine.agents["tobin"]) > EVENING
    assert _until(engine.agents["isolde"]) == datetime(1347, 11, 8, 17)


async def test_given_someone_who_stays_when_the_day_comes_then_the_model_is_asked_again_and_may_send_them_home(tmp_path):
    # Given Tobin who stays away, looked at in nine days
    road = Road(telling(fate("Isolde", days=2), fate("Tobin", "stays_away", days=9)),
                telling(fate("Tobin", "returns", days=1, memory="The north road ended at a burned farm, and I came home.",
                             state="hollowed out"), story="the north road ended at a burned farm"))
    engine = await _engine(tmp_path, road)
    await _set_out(engine, "isolde", "tobin")
    tobin = engine.agents["tobin"]
    # When the day comes and the model sends him home in a day
    await engine._absences(_until(tobin))
    assert tobin.is_away and _until(tobin) == datetime(1347, 11, 15, 17)
    await engine._absences(_until(tobin))
    # Then he is home, with the memory the model gave and the state it named
    assert not tobin.is_away and tobin.mode == "acting"
    assert any(m.text == "The north road ended at a burned farm, and I came home." and m.importance == 8
               for m in tobin.memory.items)
    assert any(s["text"] == "hollowed out" for s in tobin.states)
    assert "9 days" in road.asked[1] or "nine days" in road.asked[1]


async def test_given_someone_who_stays_when_the_day_comes_and_the_model_fails_then_they_come_home(tmp_path):
    # Given Tobin who stays, and a model that fails at the second call
    road = Road(telling(fate("Isolde", days=2), fate("Tobin", "stays_away", days=9)), LLMError("down"))
    engine = await _engine(tmp_path, road)
    await _set_out(engine, "isolde", "tobin")
    tobin = engine.agents["tobin"]
    # When the day comes
    await engine._absences(_until(tobin))
    # Then nobody stays lost: he is home
    assert not tobin.is_away


async def test_given_a_short_visit_when_they_are_home_then_they_set_out_again_after_two_days(tmp_path):
    # Given Isolde who is back for a short while
    road = Road(telling(fate("Isolde", "returns_briefly", days=1), fate("Tobin", days=1)))
    engine = await _engine(tmp_path, road)
    await _set_out(engine, "isolde", "tobin")
    isolde = engine.agents["isolde"]
    back = _until(isolde)
    # When she comes home
    await engine._absences(back)
    # Then she is home and a new departure waits for her
    assert not isolde.is_away
    again = [q for q in engine._queue if q["do"] == "depart" and q["ids"] == ["isolde"]]
    assert again and datetime.fromisoformat(again[0]["at"]) == datetime.combine(back.date() + timedelta(days=2),
                                                                           datetime.min.time().replace(hour=22))


# ------------------------------------------------------------------ the town misses them


async def _gone_long(tmp_path, *tellings):
    engine = await _engine(tmp_path, Road(*tellings))
    await _set_out(engine, "isolde", "tobin")
    return engine


def _heard(agent, needle):
    return [m for m in agent.memory.items if needle in m.text]


async def test_given_two_gone_three_days_when_the_morning_comes_then_the_town_hears_they_are_missed(tmp_path):
    # Given Isolde and Tobin who have not come home
    engine = await _gone_long(tmp_path, telling(fate("Isolde", "stays_away", days=14), fate("Tobin", "stays_away", days=14)))
    # When the third morning after they left comes
    await engine._absences(datetime(1347, 11, 8, 6))
    # Then the feed says so, and their circle remembers it
    assert any(e["kind"] == "missed" and "3 days" in e["text"] for e in engine.events)
    agnes = engine.agents["agnes"]
    assert _heard(agnes, "Isolde and Tobin have been gone from Hollowmere for 3 days")


async def test_given_the_town_misses_them_when_a_week_goes_by_then_it_misses_them_again(tmp_path):
    # Given the third-day news is out
    engine = await _gone_long(tmp_path, telling(fate("Isolde", "stays_away", days=20), fate("Tobin", "stays_away", days=20)))
    await engine._absences(datetime(1347, 11, 8, 6))
    # When a week passes and they are still away
    await engine._absences(datetime(1347, 11, 15, 6))
    # Then it is said again with the new count
    assert _heard(engine.agents["agnes"], "for 10 days")


async def test_given_they_are_home_when_the_news_falls_due_then_nothing_is_said(tmp_path):
    # Given both come home on the second day
    engine = await _gone_long(tmp_path, telling(fate("Isolde", days=2), fate("Tobin", days=2)))
    await engine._absences(datetime(1347, 11, 7, 17))
    # When the third morning comes
    await engine._absences(datetime(1347, 11, 8, 6))
    # Then nobody is missed
    assert not _heard(engine.agents["agnes"], "have been gone from Hollowmere")


async def test_given_a_rumor_when_the_morning_comes_then_the_circle_hears_it_as_a_rumor(tmp_path):
    # Given a telling that sends a rumor into the town
    engine = await _gone_long(tmp_path, telling(
        fate("Isolde", "stays_away", days=14), fate("Tobin", "stays_away", days=14),
        rumor="Two figures were seen crossing the north ford at dawn."))
    # When the morning after they left comes
    await engine._absences(datetime(1347, 11, 7, 6))
    # Then those close to them hold it as word, not as sight
    heard = _heard(engine.agents["agnes"], "north ford")
    assert heard and heard[0].source == "word"
    assert not _heard(engine.agents["silas"], "north ford") or engine.agents["silas"] in engine.circle(
        [engine.agents["isolde"], engine.agents["tobin"]])


async def test_given_a_child_on_the_road_when_the_model_tells_it_then_the_child_carries_what_became_of_it(tmp_path):
    # Given Isolde and Tobin with their baby on the road
    road = Road(telling(fate("Isolde", "stays_away", days=9), fate("Tobin", "stays_away", days=9),
                        children="The baby thrived on goat's milk from a shepherd's wife."))
    engine = await _engine(tmp_path, road)
    child = await _baby(engine)
    # When they set out
    await _set_out(engine, "isolde", "tobin")
    # Then the telling was asked to think of the child's growth, and the child holds what was told
    assert "grow on the road" in road.asked[0] and child.name in road.asked[0]
    assert child.absence["kids"] == "The baby thrived on goat's milk from a shepherd's wife."
