from datetime import datetime, timedelta

from hollowmere.agents import prompts
from hollowmere.config import Config
from hollowmere.llm.fake import FakeLLM
from hollowmere.memory.embed import HashingEmbedder
from hollowmere.memory.stream import Memory
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


async def test_converse_does_not_force_private_thoughts_into_suspicions_and_fears(tmp_path):
    """Regression: private thoughts were always introduced as 'suspicions, schemes and fears', even
    when the stored memory itself was warm or neutral. A live save audited on 2026-10-25 showed almost
    every talk reading as guarded or combative regardless of the speakers' actual stance - this blanket
    label was one of the reasons why: it told the model every private thought is, by its nature,
    something to guard against, before the model ever sees what the thought actually is."""
    engine = await _engine(tmp_path)
    nib, roderic = engine.agents["nib"], engine.agents["roderic"]
    text = prompts.converse(nib, engine.ctx.names, NOW, "riverbank", roderic.name, "patrolling the square",
                            "", [], [], town_places=["riverbank"], known_as=engine.ctx.known_as)
    assert "suspicions, schemes and fears" not in text
    assert "private thoughts" in text.lower()


async def test_converse_lets_an_existing_fondness_show_instead_of_only_guardedness(tmp_path):
    """Regression: even when two people already felt fond or better toward each other, converse() gave
    the model no instruction to let that warmth shape the talk - only the grim world and private
    suspicions were ever named - so dialogue defaulted to friction and chat_digest then (correctly,
    given the words actually spoken) judged it wary, eroding bonds that should have been growing."""
    engine = await _engine(tmp_path)
    nib, roderic = engine.agents["nib"], engine.agents["roderic"]

    nib.bonds[roderic.id] = {"stance": "trust", "affinity": 5.0, "note": "", "talks": 3, "last": None}
    warm_text = prompts.converse(nib, engine.ctx.names, NOW, "riverbank", roderic.name, "patrolling the square",
                                 "", [], [], town_places=["riverbank"], known_as=engine.ctx.known_as)
    assert "let that" in warm_text.lower()

    nib.bonds[roderic.id] = {"stance": "wary", "affinity": 2.0, "note": "", "talks": 3, "last": None}
    wary_text = prompts.converse(nib, engine.ctx.names, NOW, "riverbank", roderic.name, "patrolling the square",
                                 "", [], [], town_places=["riverbank"], known_as=engine.ctx.known_as)
    assert "let that" not in wary_text.lower()


async def test_converse_names_making_love_as_an_option_once_eligible(tmp_path):
    """Without an explicit nudge, the model never writes dialogue that goes there at all, so
    chat_digest's strict, transcript-only made_love judgment can never find anything to confirm -
    affinity can climb as high as it likes and the mechanism still never fires."""
    engine = await _engine(tmp_path)
    nib, isolde = engine.agents["nib"], engine.agents["isolde"]

    not_yet = prompts.converse(nib, engine.ctx.names, NOW, "riverbank", isolde.name, "walking",
                               "", [], [], town_places=["riverbank"], known_as=engine.ctx.known_as,
                               romance_eligible=False)
    assert "make love" not in not_yet.lower()

    eligible = prompts.converse(nib, engine.ctx.names, NOW, "riverbank", isolde.name, "walking",
                                "", [], [], town_places=["riverbank"], known_as=engine.ctx.known_as,
                                romance_eligible=True)
    assert "make love" in eligible.lower()


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


async def test_identity_shows_the_situation_in_place_of_the_seed_currently(tmp_path):
    engine = await _engine(tmp_path)
    mara = engine.agents["mara"]
    assert f"Lately: {mara.persona.currently}" in prompts.identity(mara, engine.ctx.names)  # until reviewed
    mara.situation = "Mara has taken the shroud to Father Odo."
    text = prompts.identity(mara, engine.ctx.names)
    assert "Lately: Mara has taken the shroud to Father Odo." in text
    assert mara.persona.currently not in text


async def test_day_review_prompt_hands_over_the_situation_to_be_carried_forward(tmp_path):
    engine = await _engine(tmp_path)
    mara = engine.agents["mara"]
    mara.situation = "Mara has taken the shroud to Father Odo."
    text = prompts.day_review(mara, engine.ctx.names, engine.now, [])
    assert "situation" in text.lower() and "Mara has taken the shroud to Father Odo." in text


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


def test_when_keeps_the_clock_time_for_old_memories_only_if_asked():
    now = datetime(1347, 10, 25, 19, 0)
    old = datetime(1347, 10, 22, 8, 15)  # 3 days before `now`
    assert prompts.when(old, now) == "3 days ago"
    assert prompts.when(old, now, keep_time=True) == "3 days ago at 08:15"


def test_memories_keep_the_clock_time_for_old_observations_but_not_other_kinds():
    now = datetime(1347, 10, 25, 19, 0)
    old = datetime(1347, 10, 22, 8, 15)
    obs = Memory(1, "observation", "Isolde tends the bar at the tavern.", old, old, 3, [])
    thought = Memory(2, "thought", "Isolde seems happier than usual lately.", old, old, 3, [])
    text = prompts.memories([obs, thought], now)
    assert "(3 days ago at 08:15) Isolde tends the bar at the tavern." in text
    assert "(3 days ago) Isolde seems happier than usual lately." in text


async def test_plan_day_includes_what_the_whole_town_knows_of_other_people(tmp_path):
    engine = await _engine(tmp_path)
    nib = engine.agents["nib"]
    text = prompts.plan_day(nib, engine.ctx.world, engine.ctx.names, NOW, [], [], known_as=engine.ctx.known_as)
    assert "THE TOWNSFOLK" in text
    assert engine.ctx.known_as["isolde"] in text


async def test_plan_day_tells_the_model_to_stay_uncertain_without_grounding(tmp_path):
    engine = await _engine(tmp_path)
    nib = engine.agents["nib"]
    text = prompts.plan_day(nib, engine.ctx.world, engine.ctx.names, NOW, [], [], known_as=engine.ctx.known_as)
    assert "only" in text and "stay uncertain" in text


async def test_chat_digest_leaves_the_feeling_rule_to_the_schema_not_the_prose(tmp_path):
    """The grounding rule for 'feeling' now lives once, in chat_digest_schema's field description -
    duplicating it in the free-form prose too just competes with itself for no benefit."""
    engine = await _engine(tmp_path)
    nib, ros = engine.agents["nib"], engine.agents["ros"]
    text = prompts.chat_digest(nib, engine.ctx.names, NOW, ros.name, [("Nib", "hello"), ("Ros", "hello")])
    assert "general dread" not in text.lower()
    assert "touches everyone alike" not in text.lower()


async def test_chat_digest_hides_the_partner_s_own_prior_stance_label(tmp_path):
    """Regression: identity() showed the partner's current stance label ('now: wary, 3 talks') right
    above the transcript, right before asking the model to judge a fresh feeling toward that same
    person - a model shown its own old verdict tends to just repeat it. The specific partner being
    digested must not see that label reflected back; everyone else's still can, that's useful context."""
    engine = await _engine(tmp_path)
    nib, ros, mara = engine.agents["nib"], engine.agents["ros"], engine.agents["mara"]
    nib.bonds["ros"] = {"stance": "wary", "affinity": 2.0, "note": "a loud washerwoman", "talks": 3, "last": None}
    nib.bonds["mara"] = {"stance": "fond", "affinity": 4.0, "note": "quietly kind", "talks": 2, "last": None}
    text = prompts.chat_digest(nib, engine.ctx.names, NOW, ros.name, [("Nib", "hello"), ("Ros", "hello")])
    ros_line = next(line for line in text.splitlines() if "Ros Mourne:" in line)
    assert "wary" not in ros_line  # her own prior label, hidden since she's the one being judged
    assert "a loud washerwoman" in ros_line  # the rest of what Nib knows of her is still shown
    mara_line = next(line for line in text.splitlines() if "Mara Hale:" in line)
    assert "fond" in mara_line  # Mara is not the partner in this talk; her label is unaffected


async def test_chat_digest_shows_what_is_already_owed_to_someone_else(tmp_path):
    """Regression: a new commitment within an hour of an existing one is silently dropped by
    _acceptable()'s PROMISE_GAP check, with no way for the model to know that will happen."""
    engine = await _engine(tmp_path)
    nib, ros = engine.agents["nib"], engine.agents["ros"]
    text = prompts.chat_digest(nib, engine.ctx.names, NOW, ros.name, [("Nib", "hello"), ("Ros", "hello")],
                                existing=["18:00 — tends the shrine at the ruin"])
    assert "18:00 — tends the shrine at the ruin" in text
    text_empty = prompts.chat_digest(nib, engine.ctx.names, NOW, ros.name, [("Nib", "hello"), ("Ros", "hello")])
    assert "none" in text_empty.lower()


async def test_chat_digest_warns_a_goblin_never_to_agree_to_an_indoor_meeting(tmp_path):
    """Regression: _acceptable() silently drops any goblin commitment to an indoor place other than
    home or work, but nothing told the model that rule exists."""
    engine = await _engine(tmp_path)
    nib, wenna = engine.agents["nib"], engine.agents["wenna"]
    text = prompts.chat_digest(nib, engine.ctx.names, NOW, wenna.name, [("Nib", "hi"), ("Wenna", "hi")])
    assert nib.persona.home in text and nib.persona.work in text
    human_text = prompts.chat_digest(wenna, engine.ctx.names, NOW, nib.name, [("Wenna", "hi"), ("Nib", "hi")])
    assert wenna.persona.home not in human_text


async def test_chat_digest_covers_a_time_renegotiated_within_the_same_talk(tmp_path):
    engine = await _engine(tmp_path)
    nib, ros = engine.agents["nib"], engine.agents["ros"]
    text = prompts.chat_digest(nib, engine.ctx.names, NOW, ros.name, [("Nib", "hi"), ("Ros", "hi")])
    assert "changed your minds" in text.lower() or "finally settled" in text.lower()


async def test_chat_digest_treats_a_vague_maybe_as_unagreed(tmp_path):
    engine = await _engine(tmp_path)
    nib, ros = engine.agents["nib"], engine.agents["ros"]
    text = prompts.chat_digest(nib, engine.ctx.names, NOW, ros.name, [("Nib", "hi"), ("Ros", "hi")])
    assert "maybe" in text.lower()


async def test_chat_digest_moves_hour_disambiguation_out_of_the_prose(tmp_path):
    """That rule now lives in chat_digest_schema's `time` field description instead - one compound,
    double-negative sentence buried mid-paragraph is exactly the kind of instruction a smaller model
    is likely to drop."""
    engine = await _engine(tmp_path)
    nib, ros = engine.agents["nib"], engine.agents["ros"]
    text = prompts.chat_digest(nib, engine.ctx.names, NOW, ros.name, [("Nib", "hi"), ("Ros", "hi")])
    assert "the opposite half of it" not in text.lower()


async def test_system_prompt_requires_suspicion_to_be_earned_not_assumed_from_the_ages_dread(tmp_path):
    """Regression: 'suspicion and rumour are fine' was an unscoped, system-level licence to distrust
    anyone for no specific reason, issued before any per-talk corrective in chat_digest ever gets a
    chance to apply - a blanket permission at that authority level is hard for a later, narrower
    instruction to outweigh."""
    engine = await _engine(tmp_path)
    text = prompts.system_prompt(engine.agents["mara"])
    assert "earned" in text.lower()
    assert "suspicion and rumour are as fine to feel as affection and desire" not in text


async def test_god_system_prompt_carries_that_gods_own_temperament_and_voice_and_the_boundary_line(tmp_path):
    from hollowmere.agents.gods import GODS
    for god in GODS:
        text = prompts.god_system_prompt(god)
        assert god.name in text and god.temperament in text and god.voice in text
        assert "only the one person praying to you" in text


async def test_identity_shows_the_agents_current_favor_when_set(tmp_path):
    engine = await _engine(tmp_path)
    mara = engine.agents["mara"]
    assert "lingers on" not in prompts.identity(mara, engine.ctx.names)  # nothing set yet
    mara.favor = {"god": "yevra", "text": "a flush of warmth", "expires": NOW.isoformat()}
    text = prompts.identity(mara, engine.ctx.names)
    assert "Yevra" in text and "a flush of warmth" in text


async def test_identity_names_the_god_of_a_favor_only_when_the_mortal_suspects_or_knows(tmp_path):
    engine = await _engine(tmp_path)
    mara = engine.agents["mara"]
    for clarity, named in (("none", False), ("doubt", False), ("suspected", True), ("certain", True)):
        mara.favor = {"god": "yevra", "text": "a flush of warmth", "expires": NOW.isoformat(), "clarity": clarity}
        text = prompts.identity(mara, engine.ctx.names)
        assert "a flush of warmth" in text, clarity
        assert ("Yevra" in text) is named, clarity


async def test_plan_day_prompt_explains_the_pray_to_field(tmp_path):
    engine = await _engine(tmp_path)
    nib = engine.agents["nib"]
    text = prompts.plan_day(nib, engine.ctx.world, engine.ctx.names, NOW, [], [], known_as=engine.ctx.known_as)
    assert "pray_to" in text and '"none"' in text


async def test_divine_prayer_prompt_shows_the_offering_and_addresses_the_god_directly(tmp_path):
    engine = await _engine(tmp_path)
    from hollowmere.agents.gods import GOD_BY_ID
    odo, god = engine.agents["odo"], GOD_BY_ID["caedros"]
    text = prompts.divine_prayer(odo, god, NOW, "prays for the dying", [])
    assert "prays for the dying" in text and god.name in text


def test_town_review_shows_the_old_state_and_what_has_happened_since():
    text = prompts.town_review(
        "The plague has returned, the harvest has failed, and Lord Maelor has not left his keep in months.",
        NOW, ["Agnes and Odo agree to meet at the tavern at 18:00.", "Wenna has sworn to protect the goblin."])
    assert "The plague has returned" in text
    assert "Agnes and Odo agree to meet at the tavern at 18:00." in text
    assert "Wenna has sworn to protect the goblin." in text


async def test_planning_prompts_forbid_writing_what_other_people_do(tmp_path):
    """Regression: Nib's plan read 'Climbs the rubble, pulling Isolde up after him' while Isolde was nowhere
    near. A step is what the planner does alone; others are only ever hoped for or promised."""
    engine = await _engine(tmp_path)
    nib = engine.agents["nib"]
    day = prompts.plan_day(nib, engine.world, engine.ctx.names, NOW, [], ["19:00 - meet Isolde at the stones"])
    assert "never what another person does" in day and "promised to meet them" in day
    steps = prompts.decompose(nib, engine.world, engine.ctx.names, NOW, "escorts Isolde", NOW,
                              NOW + timedelta(hours=2), "the ruin", [])
    assert "only what Nib does on their own" in steps
    assert "not what happens once they are together" in steps
