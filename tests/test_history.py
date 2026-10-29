"""Lasting facts of a life: what has happened to a person that stays true, kept apart from mood and temperament."""

from datetime import datetime

from hollowmere.agents import prompts
from hollowmere.agents.agent import Ctx
from hollowmere.config import Config
from hollowmere.llm.fake import FakeLLM
from hollowmere.memory.embed import HashingEmbedder
from hollowmere.sim.setup import build_engine

NOW = datetime(1347, 11, 9, 21)
BIRTH = "Isolde bore Nib's child in the tavern garret after a night of pain."


async def _engine(tmp_path):
    cfg = Config()
    cfg.sim.save_path = str(tmp_path / "run.db")
    return await build_engine(cfg, llm=FakeLLM(latency=0), embedder=HashingEmbedder(128))


class Reply:
    """Answers the day review with nothing much and the life-facts call from a script; remembers what it was asked."""

    def __init__(self, facts=None, evidence=""):
        self.facts, self.evidence, self.asked = facts or [], evidence, []

    async def structured(self, system, user, schema, *, label, temperature=None):
        if label == "life_facts":
            self.asked.append(user)
            return schema.model_validate({"facts": self.facts, "evidence": self.evidence})
        return schema.model_validate({"summary": "A day.", "importance": 4, "intentions": [], "drive": "to rest",
                                      "states": []})


async def _review(engine, agent, reply, text=BIRTH, importance=9):
    await agent.observe(text, NOW, importance)
    await agent.review_day(Ctx(reply, engine.world, engine.ctx.names, {}), NOW)


def _facts(agent):
    return [f["text"] for f in agent.history]


async def test_given_a_person_with_lasting_facts_when_the_identity_is_written_then_they_follow_the_background(tmp_path):
    # Given Isolde with one lasting fact
    engine = await _engine(tmp_path)
    isolde = engine.agents["isolde"]
    isolde.history = [{"text": "Bore Nib's child.", "since": "1347-11-09"}]
    # When her identity is written
    text = prompts.identity(isolde, engine.ctx.names)
    # Then the fact stands in its own block after her background, apart from her states
    assert "Since the story began: Bore Nib's child." in text
    assert text.index(isolde.persona.background[:30]) < text.index("Since the story began")


async def test_given_a_very_important_day_when_reviewed_then_a_backed_fact_is_kept_with_its_date(tmp_path):
    # Given Isolde remembered the birth as a very important thing today
    engine = await _engine(tmp_path)
    isolde = engine.agents["isolde"]
    reply = Reply(["Bore Nib's child."], evidence="bore Nib's child in the tavern garret")
    # When the day is reviewed
    await _review(engine, isolde, reply)
    # Then the fact is kept, dated today
    assert _facts(isolde) == ["Bore Nib's child."] and isolde.history[0]["since"] == "1347-11-09"


async def test_given_an_unremarkable_day_when_reviewed_then_the_facts_are_not_even_asked_for(tmp_path):
    # Given a day with only a middling memory
    engine = await _engine(tmp_path)
    reply = Reply(["Invented."], evidence="x")
    # When the day is reviewed
    await _review(engine, engine.agents["isolde"], reply, text="She swept the floor.", importance=5)
    # Then no life-facts call was made
    assert reply.asked == [] and engine.agents["isolde"].history == []


async def test_given_a_fact_the_day_does_not_show_when_reviewed_then_it_is_dropped(tmp_path):
    # Given a model that offers a fact with a quote that is nowhere in the day
    engine = await _engine(tmp_path)
    isolde = engine.agents["isolde"]
    reply = Reply(["Is secretly a witch."], evidence="she cast a spell at midnight")
    # When the day is reviewed
    await _review(engine, isolde, reply)
    # Then nothing is added
    assert isolde.history == []


async def test_given_no_new_fact_when_the_model_omits_the_old_ones_then_none_is_lost(tmp_path):
    # Given Isolde with two facts, and a model that answers with an empty list
    engine = await _engine(tmp_path)
    isolde = engine.agents["isolde"]
    isolde.history = [{"text": "A", "since": "1347-11-01"}, {"text": "B", "since": "1347-11-02"}]
    # When the day is reviewed
    await _review(engine, isolde, Reply([], evidence=""))
    # Then both stay
    assert _facts(isolde) == ["A", "B"]


async def test_given_two_new_facts_when_reviewed_then_only_the_first_is_taken(tmp_path):
    # Given a model that offers two new facts, both backed
    engine = await _engine(tmp_path)
    isolde = engine.agents["isolde"]
    reply = Reply(["Bore Nib's child.", "Lives in the tavern garret."], evidence="bore Nib's child")
    # When the day is reviewed
    await _review(engine, isolde, reply)
    # Then one fact is added per day
    assert _facts(isolde) == ["Bore Nib's child."]


async def test_given_a_full_list_when_a_fact_is_added_then_a_dropped_one_makes_room_or_the_oldest_goes(tmp_path):
    # Given six facts, F1 the oldest
    engine = await _engine(tmp_path)
    isolde = engine.agents["isolde"]
    isolde.history = [{"text": f"F{i}", "since": f"1347-11-0{i}"} for i in range(1, 7)]
    # When the model adds one and leaves out F3 (merged into the new one)
    reply = Reply(["F1", "F2", "F4", "F5", "F6", "Bore Nib's child."], evidence="bore Nib's child")
    await _review(engine, isolde, reply)
    # Then F3 went, and the rest stayed
    assert _facts(isolde) == ["F1", "F2", "F4", "F5", "F6", "Bore Nib's child."]
    # And when the model keeps all six, the oldest makes room for the next one
    isolde.reviewed_date = None
    reply2 = Reply(_facts(isolde) + ["Fled once."], evidence="bore Nib's child")
    await _review(engine, isolde, reply2, text="Isolde bore Nib's child again in a dream.")
    assert len(isolde.history) == 6 and "F1" not in _facts(isolde) and _facts(isolde)[-1] == "Fled once."


async def test_given_facts_when_saved_and_restored_then_they_return_and_an_old_save_has_none(tmp_path):
    # Given Isolde with a fact
    engine = await _engine(tmp_path)
    isolde = engine.agents["isolde"]
    isolde.history = [{"text": "Bore Nib's child.", "since": "1347-11-09"}]
    snap = isolde.snapshot()
    # When she is restored from it, and from a save made before this existed
    isolde.history = []
    isolde.restore(snap)
    restored = _facts(isolde)
    old = {k: v for k, v in snap.items() if k != "history"}
    isolde.restore(old)
    # Then the fact comes back, and an old save leaves her with none
    assert restored == ["Bore Nib's child."] and _facts(isolde) == []


async def test_given_facts_when_the_detail_is_built_for_the_ui_then_they_are_sent(tmp_path):
    # Given Isolde with a fact
    engine = await _engine(tmp_path)
    engine.agents["isolde"].history = [{"text": "Bore Nib's child.", "since": "1347-11-09"}]
    # When the detail is built
    detail = engine.agent_detail("isolde")
    # Then it carries the fact
    assert detail["history"][0]["text"] == "Bore Nib's child."


async def test_given_the_life_facts_prompt_when_written_then_it_shows_the_old_facts_and_rules_out_moods(tmp_path):
    # Given Isolde with a fact and an important memory
    engine = await _engine(tmp_path)
    isolde = engine.agents["isolde"]
    isolde.history = [{"text": "Fled once.", "since": "1347-11-08"}]
    await isolde.observe(BIRTH, NOW, 9)
    mems = [m for m in isolde.memory.items if m.importance >= 8]
    # When the prompt is written
    text = prompts.life_facts(isolde, engine.ctx.names, NOW, mems)
    # Then the old fact is named, and a mood is ruled out
    assert "Fled once." in text and "mood" in text and "gone in a week" in text
