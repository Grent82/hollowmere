"""Passing states of mind (grieving, exhausted) kept apart from the temperament that stays."""

from datetime import date, datetime

from hollowmere.agents import prompts
from hollowmere.agents.agent import Ctx
from hollowmere.config import Config
from hollowmere.llm.fake import FakeLLM
from hollowmere.memory.embed import HashingEmbedder
from hollowmere.sim.setup import build_engine

NOW = datetime(1347, 11, 5, 21)


async def _engine(tmp_path):
    cfg = Config()
    cfg.sim.save_path = str(tmp_path / "run.db")
    return await build_engine(cfg, llm=FakeLLM(latency=0), embedder=HashingEmbedder(128))


async def _review(engine, agent, states, now=NOW):
    """Run the agent's day review with a model that answers with `states`."""
    seen = []

    class Reply:
        async def structured(self, system, user, schema, **kw):
            seen.append(user)
            return schema.model_validate({
                "summary": "A day.", "importance": 4, "intentions": ["rest"], "drive": "to rest", "states": states})

    await agent.observe("Something happened in the lane.", now, 5)
    await agent.review_day(Ctx(Reply(), engine.world, engine.ctx.names, {}), now)
    return seen[0]


async def test_given_elsbeth_when_the_town_starts_then_her_grief_is_a_state_and_not_a_trait(tmp_path):
    # Given Elsbeth and Clemence as the persona file writes them
    engine = await _engine(tmp_path)
    elsbeth, clemence = engine.agents["elsbeth"], engine.agents["clemence"]
    # Then grieving and exhausted are passing states, and no longer part of the temperament
    assert "grieving" not in elsbeth.persona.traits and "stubborn" in elsbeth.persona.traits
    assert "exhausted" not in clemence.persona.traits
    assert [s["text"] for s in elsbeth.states] == ["grieving for her child"]
    assert [s["text"] for s in clemence.states] == ["exhausted"]


async def test_given_a_state_when_the_identity_is_written_then_it_stands_apart_from_the_temperament(tmp_path):
    # Given Elsbeth grieving
    engine = await _engine(tmp_path)
    elsbeth = engine.agents["elsbeth"]
    # When her identity is written
    text = prompts.identity(elsbeth, engine.ctx.names)
    # Then the state has its own line, apart from "Temperament:"
    assert "Right now: grieving for her child" in text
    assert "Temperament: proud, stubborn" in text and "grieving" not in text.split("Temperament:")[1].split("\n")[0]


async def test_given_a_review_without_the_grief_when_the_day_is_reviewed_then_the_state_is_gone(tmp_path):
    # Given Elsbeth, who has been grieving
    engine = await _engine(tmp_path)
    elsbeth = engine.agents["elsbeth"]
    # When the review ends her grief and notes a lighter state
    await _review(engine, elsbeth, ["a little lighter after the baking"])
    # Then only the new state remains
    assert [s["text"] for s in elsbeth.states] == ["a little lighter after the baking"]


async def test_given_a_kept_state_when_reviewed_then_it_keeps_its_age_and_a_new_one_starts_today(tmp_path):
    # Given Elsbeth with a grief that began on the first day
    engine = await _engine(tmp_path)
    elsbeth = engine.agents["elsbeth"]
    elsbeth.states[0]["since"] = "1347-10-20"
    # When the review keeps it, in other letters, and adds one
    await _review(engine, elsbeth, ["Grieving for her child", "proud of the loaf"])
    # Then the grief is still from the 20th, and the new state is from today
    by_text = {s["text"].lower(): s for s in elsbeth.states}
    assert by_text["grieving for her child"]["since"] == "1347-10-20"
    assert by_text["proud of the loaf"]["since"] == "1347-11-05"


async def test_given_too_many_states_when_reviewed_then_at_most_three_remain_and_the_held_one_comes_first(tmp_path):
    # Given Elsbeth grieving, and a model that piles on one dark mood after another
    engine = await _engine(tmp_path)
    elsbeth = engine.agents["elsbeth"]
    # When the review lists five, the held grief somewhere in the middle
    await _review(engine, elsbeth, [
        "afraid of the Watch", "grieving for her child", "furious at the miller", "bitter about the harvest",
        "glad of the warm oven"])
    # Then three remain: the grief she already held, then the new ones in the order given
    assert [s["text"] for s in elsbeth.states] == ["grieving for her child", "afraid of the Watch", "furious at the miller"]


async def test_given_states_when_saved_and_restored_then_they_come_back_and_old_saves_keep_the_seed(tmp_path):
    # Given Clemence with a changed state
    engine = await _engine(tmp_path)
    clemence = engine.agents["clemence"]
    await _review(engine, clemence, ["rested at last"])
    snap = clemence.snapshot()
    # When she is restored from it, and from a save made before states
    clemence.states = []
    clemence.restore(snap)
    restored = [s["text"] for s in clemence.states]
    old = {k: v for k, v in snap.items() if k != "states"}
    clemence.states = [{"text": "seed", "since": None}]
    clemence.restore(old)
    # Then the saved ones return, and the old save leaves the seed alone
    assert restored == ["rested at last"] and clemence.states[0]["text"] == "seed"


async def test_given_the_day_review_prompt_when_written_then_it_shows_ages_and_asks_for_bright_states_too(tmp_path):
    # Given Elsbeth with a grief that is twelve days old
    engine = await _engine(tmp_path)
    elsbeth = engine.agents["elsbeth"]
    elsbeth.states[0]["since"] = "1347-10-24"
    # When the day review prompt is written
    text = prompts.day_review(elsbeth, engine.ctx.names, NOW, [])
    # Then the age is named, and the bright kinds of state are asked for as much as the dark
    assert "grieving for her child (12 days)" in text
    assert "relieved" in text and "at most three" in text


async def test_given_a_state_when_the_agent_is_shown_in_the_ui_then_it_is_sent(tmp_path):
    # Given Elsbeth grieving
    engine = await _engine(tmp_path)
    # When the detail for the UI is built
    detail = engine.agent_detail("elsbeth")
    # Then the state is sent with its text
    assert detail["states"][0]["text"] == "grieving for her child"
    assert "grieving" not in detail["traits"]
