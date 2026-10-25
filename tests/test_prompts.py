from datetime import datetime

from hollowmere.agents import prompts
from hollowmere.config import Config
from hollowmere.llm.fake import FakeLLM
from hollowmere.memory.embed import HashingEmbedder
from hollowmere.sim.setup import build_engine

NOW = datetime(1347, 10, 22, 19, 0)


async def _engine(tmp_path):
    cfg = Config()
    cfg.sim.save_path = str(tmp_path / "run.db")
    return await build_engine(cfg, llm=FakeLLM(latency=0), embedder=HashingEmbedder(128))


async def test_system_prompt_tells_model_to_derive_voice_per_persona(tmp_path):
    engine = await _engine(tmp_path)
    for agent_id in ("nib", "roderic", "mara"):
        text = prompts.system_prompt(engine.agents[agent_id])
        assert "frightened townsperson" not in text
        assert "No two people in Hollowmere think or talk alike" in text


async def test_converse_drops_shared_narrator_override(tmp_path):
    engine = await _engine(tmp_path)
    nib, roderic = engine.agents["nib"], engine.agents["roderic"]
    text = prompts.converse(nib, engine.ctx.names, NOW, "riverbank", roderic.name, "patrolling the square",
                            "", [], [], town_places=["riverbank"], known_as=engine.ctx.known_as)
    assert "the way a frightened townsperson in 1347 talks" not in text
    assert "not a shared narrator's voice" in text
    # the persona-specific facts that should drive the voice are still there
    assert nib.persona.traits in text


async def test_reflection_prompts_also_ask_for_the_personas_own_voice(tmp_path):
    engine = await _engine(tmp_path)
    mara = engine.agents["mara"]
    assert "not a shared narrator's" in prompts.focal_questions(mara, engine.ctx.names, NOW, [])
    assert "not a shared narrator's" in prompts.insights(mara, engine.ctx.names, NOW, ["why?"], [])
    assert "not a shared narrator's" in prompts.day_review(mara, engine.ctx.names, NOW, [])


async def test_insights_prompt_does_not_push_the_model_toward_danger_only(tmp_path):
    engine = await _engine(tmp_path)
    mara = engine.agents["mara"]
    text = prompts.insights(mara, engine.ctx.names, NOW, ["why?"], [])
    assert "dangers" in text  # danger is still a valid topic...
    assert "Fear is not the only thing worth concluding" in text  # ...but no longer the default one


async def test_identity_states_the_agents_sex(tmp_path):
    engine = await _engine(tmp_path)
    mara = engine.agents["mara"]
    assert "31, female" in prompts.identity(mara, engine.ctx.names)


async def test_identity_shows_the_agents_own_current_drive_when_set(tmp_path):
    engine = await _engine(tmp_path)
    mara = engine.agents["mara"]
    assert "wants for themself" not in prompts.identity(mara, engine.ctx.names)  # nothing set yet
    mara.drive = "to finally stand up to Roderic"
    text = prompts.identity(mara, engine.ctx.names)
    assert "to finally stand up to Roderic" in text


async def test_system_prompt_falls_back_to_the_seed_town_state_until_reviewed(tmp_path):
    engine = await _engine(tmp_path)
    mara = engine.agents["mara"]
    assert mara.town_state == ""
    text = prompts.system_prompt(mara)
    assert "the plague has returned" in text.lower()


async def test_system_prompt_uses_the_agents_own_town_state_once_set(tmp_path):
    engine = await _engine(tmp_path)
    mara = engine.agents["mara"]
    mara.town_state = "The plague has lifted at last, though the harvest still fails."
    text = prompts.system_prompt(mara)
    assert "The plague has lifted at last" in text
    assert "the plague has returned" not in text.lower()  # the seed line is gone, not just appended


async def test_system_prompt_keeps_the_fixed_world_rules_no_matter_what_the_town_state_says(tmp_path):
    engine = await _engine(tmp_path)
    mara = engine.agents["mara"]
    mara.town_state = "A season of plenty and peace has somehow come to Hollowmere."
    text = prompts.system_prompt(mara)
    assert "Yevra" in text and "only rumour and faith" in text


def test_narrator_system_prompt_speaks_for_no_one_in_particular():
    text = prompts.narrator_system_prompt()
    assert "Yevra" in text  # the fixed rules still bind the narrator
    assert "mind of" not in text  # unlike system_prompt(), this is nobody's personal voice


def test_town_review_shows_the_old_state_and_what_has_happened_since():
    text = prompts.town_review(
        "The plague has returned, the harvest has failed, and Lord Maelor has not left his keep in months.",
        NOW, ["Agnes and Odo agree to meet at the tavern at 18:00.", "Wenna has sworn to protect the goblin."])
    assert "The plague has returned" in text
    assert "Agnes and Odo agree to meet at the tavern at 18:00." in text
    assert "Wenna has sworn to protect the goblin." in text
