"""The evening collections: what the gods wish for, and what the townsfolk would change in the world."""

from datetime import datetime

from hollowmere.config import Config
from hollowmere.llm.fake import FakeLLM
from hollowmere.memory.embed import HashingEmbedder
from hollowmere.sim import reports
from hollowmere.sim.setup import build_engine

EVENING = datetime(1347, 11, 5, 23, 10)


class Scripted(FakeLLM):
    """Answers the two report calls from a script and remembers what it was asked."""

    def __init__(self, wishes=None, ideas=None):
        super().__init__(latency=0)
        self.wishes, self.ideas, self.asked = wishes or [], ideas or [], {"god_wishes": [], "world_ideas": []}

    async def structured(self, system, user, schema, *, label, temperature=None):
        if label in self.asked:
            self.asked[label].append(user)
        return await super().structured(system, user, schema, label=label, temperature=temperature)

    def _godwishes(self, js: dict, user: str) -> dict:
        return {"wishes": self.wishes}

    def _worldideas(self, js: dict, user: str) -> dict:
        return {"ideas": self.ideas}


async def _engine(tmp_path, llm=None):
    cfg = Config()
    cfg.sim.save_path = str(tmp_path / "run.db")
    engine = await build_engine(cfg, llm=llm or Scripted(), embedder=HashingEmbedder(128))
    engine.now = EVENING
    return engine


def _pray(engine, agent_id="odo", god="yevra", text="A warmth settles over you."):
    engine._event("prayer", [agent_id], f"{agent_id} prays to {god}: {text}", form="sign", god=god, clarity="none")


WISH = {"wish": "I long to walk among them and give them love.", "towards": "the world",
        "why": "They pray to me only in pain."}


async def test_given_a_prayer_to_yevra_when_the_evening_report_runs_then_her_wish_is_written_to_the_file(tmp_path):
    # Given Yevra has been prayed to, and wishes to walk among mortals
    engine = await _engine(tmp_path, Scripted(wishes=[WISH]))
    _pray(engine)
    # When the evening reports run
    await reports.run(engine, EVENING)
    # Then her wish is in the file, under her name and the day
    text = (reports.folder(engine) / "god_wishes.md").read_text(encoding="utf-8")
    assert "Yevra" in text and "walk among them" in text and "1347-11-05" in text


async def test_given_a_god_nobody_prayed_to_when_the_report_runs_then_that_god_is_not_asked(tmp_path):
    # Given only Yevra has been prayed to
    llm = Scripted(wishes=[WISH])
    engine = await _engine(tmp_path, llm)
    _pray(engine)
    # When the evening reports run
    await reports.run(engine, EVENING)
    # Then only one god was asked
    assert len(llm.asked["god_wishes"]) == 1


async def test_given_earlier_wishes_when_the_report_runs_again_then_the_god_is_shown_them(tmp_path):
    # Given a wish already collected
    llm = Scripted(wishes=[WISH])
    engine = await _engine(tmp_path, llm)
    _pray(engine)
    await reports.run(engine, EVENING)
    # When the reports run the next evening
    await reports.run(engine, datetime(1347, 11, 6, 23, 10))
    # Then the second ask contains the first wish, to add to it and not repeat it
    assert "walk among them" in llm.asked["god_wishes"][1]


async def test_given_ideas_when_the_report_runs_then_only_those_with_a_line_from_the_day_are_kept(tmp_path):
    # Given Clemence remembered a line today, and the model offers two ideas, one with a made-up quote
    ideas = [{"kind": "missing_item", "item": "lantern", "where": "tavern", "what": "a physician's lantern",
              "evidence": "the physician's lantern was dark"},
             {"kind": "change", "item": "roof", "where": "tavern", "what": "a roof on the ward",
              "evidence": "nobody ever said this"}]
    engine = await _engine(tmp_path, Scripted(ideas=ideas))
    await engine.agents["clemence"].observe("I saw that the physician's lantern was dark on the ward.", EVENING, 5)
    # When the evening reports run
    await reports.run(engine, EVENING)
    # Then only the idea backed by a line is written
    text = (reports.folder(engine) / "world_ideas.md").read_text(encoding="utf-8")
    assert "physician's lantern" in text and "roof on the ward" not in text and "Clemence" in text


async def test_given_the_day_when_the_world_is_scanned_then_the_prompt_has_the_objects_the_game_has(tmp_path):
    # Given a day with something remembered
    llm = Scripted()
    engine = await _engine(tmp_path, llm)
    await engine.agents["clemence"].observe("Something happened on the ward.", EVENING, 5)
    # When the world is scanned
    await reports.run(engine, EVENING)
    # Then the prompt lists what exists in the game, so the model can tell what is missing
    some = next(iter(engine.world.objects.values())).name
    assert some in llm.asked["world_ideas"][0] and "Something happened on the ward" in llm.asked["world_ideas"][0]


async def test_given_the_clock_when_asked_then_the_report_is_due_once_after_23_and_not_again_that_day(tmp_path):
    # Given the afternoon, and then the late evening
    engine = await _engine(tmp_path)
    afternoon = datetime(1347, 11, 5, 15)
    # Then it is not due early, due late, and not due again once started
    assert not engine._report_due(afternoon)
    assert engine._report_due(EVENING)
    engine._reported = EVENING.date()
    assert not engine._report_due(EVENING)


async def test_given_a_failing_model_when_the_reports_run_then_the_simulation_is_not_hurt(tmp_path):
    # Given a model that always fails
    from hollowmere.llm.client import LLMError

    class Broken(FakeLLM):
        async def structured(self, *a, **kw):
            raise LLMError("down")

    engine = await _engine(tmp_path, Broken(latency=0))
    _pray(engine)
    # When the reports run
    await engine._run_reports(EVENING)
    # Then nothing is raised, and no file is made from nothing
    assert not (reports.folder(engine) / "god_wishes.md").exists()


async def test_given_two_gods_with_wishes_when_the_report_runs_then_the_day_has_one_heading(tmp_path):
    # Given two gods that were prayed to and both wish for something
    engine = await _engine(tmp_path, Scripted(wishes=[WISH]))
    _pray(engine, god="yevra")
    _pray(engine, god="caedros")
    # When the evening reports run
    await reports.run(engine, EVENING)
    # Then the file has the day once, with both gods under it
    text = (reports.folder(engine) / "god_wishes.md").read_text(encoding="utf-8")
    assert text.count("## 1347-11-05") == 1 and "Yevra" in text and "Caedros" in text


async def test_given_a_missing_item_the_game_already_has_when_the_report_runs_then_it_is_not_written(tmp_path):
    # Given the town has a market stall and the model calls one missing, and also names a truly missing lamp
    ideas = [{"kind": "missing_item", "item": "market stall", "where": "tavern", "what": "a market stall for turnips",
              "evidence": "bare stall"},
             {"kind": "missing_item", "item": "bucket", "where": "tavern", "what": "a bucket for the ward",
              "evidence": "bucket leaks"}]
    engine = await _engine(tmp_path, Scripted(ideas=ideas))
    assert any(o.name == "market stall" for o in engine.world.objects.values())
    await engine.agents["clemence"].observe("The bare stall stood empty and the bucket leaks.", EVENING, 5)
    # When the evening reports run
    await reports.run(engine, EVENING)
    # Then only the bucket, which the game lacks, is written
    text = (reports.folder(engine) / "world_ideas.md").read_text(encoding="utf-8")
    assert "bucket for the ward" in text and "market stall" not in text


async def test_given_an_idea_with_a_place_when_the_report_runs_then_the_file_names_the_item_and_where_it_belongs(tmp_path):
    # Given Wat remembers sifting at the mill, and the model says a sieve belongs at the mill
    ideas = [{"kind": "missing_item", "item": "sieve", "where": "mill", "what": "a sieve for the meal",
              "evidence": "sifts the meal through a sieve"}]
    engine = await _engine(tmp_path, Scripted(ideas=ideas))
    await engine.agents["wat"].observe("Wat sifts the meal through a sieve.", EVENING, 5)
    # When the evening reports run
    await reports.run(engine, EVENING)
    # Then the entry reads as Wat's sieve at the old mill
    text = (reports.folder(engine) / "world_ideas.md").read_text(encoding="utf-8")
    assert "**Wat** [missing_item] sieve (the old mill)" in text


async def test_given_the_world_scan_when_the_prompt_is_written_then_it_lists_the_places_to_choose_from(tmp_path):
    # Given any day with a memory
    llm = Scripted()
    engine = await _engine(tmp_path, llm)
    await engine.agents["wat"].observe("Wat works at the mill.", EVENING, 5)
    # When the world is scanned
    await reports.run(engine, EVENING)
    # Then the places are named, so `where` can be one of them
    assert "mill: the old mill" in llm.asked["world_ideas"][0]
