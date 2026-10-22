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
