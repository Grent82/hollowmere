"""Making love is decided where the model can decide it: in the line itself, by both of them."""

from datetime import datetime, timedelta

from hollowmere.agents import prompts
from hollowmere.agents.schemas import Utterance, utterance_schema
from hollowmere.config import Config
from hollowmere.llm.fake import FakeLLM
from hollowmere.memory.embed import HashingEmbedder
from hollowmere.sim.engine import MAX_TURNS
from hollowmere.sim.setup import build_engine

NOW = datetime(1347, 11, 3, 23, 30)
# Unlike one another, so that the talk is not cut short as a run of repeats before it reaches the limit.
LINES = ["The reeds are cold tonight.", "Hold my hand a moment.", "Somewhere a dog is barking.",
         "I fear the Watch will come.", "Your hair smells of smoke.", "Tell me about your mother.",
         "The bells have stopped at last.", "Nobody saw us leave the tavern.", "My feet ache from the lane.",
         "There is bread in my pocket.", "Do you hear the river running?", "Tomorrow I must carry water.",
         "A star is falling over the keep.", "Stay until the lantern burns out."]
HALF = MAX_TURNS // 2  # the first speaker has the odd lines, the second the even ones: the last line is the second's HALF-th


async def _engine(tmp_path, **sim):
    cfg = Config()
    cfg.sim.save_path = str(tmp_path / "run.db")
    for k, v in sim.items():
        setattr(cfg.sim, k, v)
    return await build_engine(cfg, llm=FakeLLM(latency=0), embedder=HashingEmbedder(128))


class Scripted:
    """Answers every line of a conversation; `flags` says, per speaker name, which of their lines are 'intimate'."""

    def __init__(self, flags: dict[str, list[bool]]):
        self.flags = {k: list(v) for k, v in flags.items()}
        self.asked_for = []
        self.rest = FakeLLM(latency=0)  # the digests after the talk run for real, as they do in the game

    async def structured(self, system, user, schema, *, label, temperature=None):
        if label != "converse":
            return await self.rest.structured(system, user, schema, label=label, temperature=temperature)
        speaker = next(n for n in self.flags if f"You are {n}." in user)
        self.asked_for.append("intimate" in schema.model_fields)
        flag = self.flags[speaker].pop(0) if self.flags[speaker] else False
        fields = {"says": LINES[(len(self.asked_for) - 1) % len(LINES)], "end": False}
        if "intimate" in schema.model_fields:
            fields["intimate"] = flag
        return schema(**fields)

    async def aclose(self):
        pass


async def _talk(engine, a, b, llm, lines=6):
    engine.ctx.llm = llm
    a.mode = b.mode = "acting"
    a.last_chat.clear(); b.last_chat.clear()
    engine.now = NOW
    engine._start_conversation(a, b, "to be with her", NOW)
    c = engine.conversations[a.convo_id]
    c.talking = True
    t = NOW
    while not c.done:
        await engine._turn(c, t)
        t += timedelta(minutes=1)
    return c


def test_the_schema_offers_the_intimate_field_only_to_a_pair_that_may():
    assert utterance_schema(False) is Utterance
    assert "intimate" not in Utterance.model_fields
    romantic = utterance_schema(True)
    assert issubclass(romantic, Utterance) and "intimate" in romantic.model_fields
    assert romantic.model_fields["intimate"].is_required()


async def test_the_converse_prompt_explains_when_to_set_intimate(tmp_path):
    engine = await _engine(tmp_path)
    a = engine.agents["isolde"]
    base = dict(names=engine.ctx.names, now=NOW, place="the reeds", other_name="Nib", other_activity="waits",
                purpose="", transcript=[], mems=[])
    eligible = prompts.converse(a, romance_eligible=True, **base)
    assert "intimate to true" in eligible
    assert "intimate" not in prompts.converse(a, romance_eligible=False, **base).lower()


async def test_both_saying_yes_in_their_own_lines_confirms_it_and_ends_the_talk(tmp_path):
    engine = await _engine(tmp_path)
    a, b = engine.agents["isolde"], engine.agents["nib"]
    llm = Scripted({a.name: [False, True], b.name: [False, True]})
    c = await _talk(engine, a, b, llm)
    await _settle(engine)
    assert len(c.transcript) == 4  # the talk ended on the second yes, not at the line limit
    assert a.bonds[b.id]["intimate"] and b.bonds[a.id]["intimate"]
    assert a.bonds[b.id]["stance"] == "close" or a.bonds[b.id]["stance"] == "love"
    assert any(e["kind"] == "pregnancy" for e in engine.events)  # male + female, conception_chance 1.0
    assert any("made love" in m.text for m in a.memory.items)


async def test_one_side_alone_confirms_nothing(tmp_path):
    engine = await _engine(tmp_path)
    a, b = engine.agents["isolde"], engine.agents["nib"]
    llm = Scripted({a.name: [True, True, True], b.name: []})
    await _talk(engine, a, b, llm)
    await _settle(engine)
    assert not a.bonds[b.id].get("intimate") and not b.bonds[a.id].get("intimate")
    assert not any(e["kind"] == "pregnancy" for e in engine.events)


async def test_the_same_speaker_saying_yes_twice_is_not_both(tmp_path):
    engine = await _engine(tmp_path)
    a, b = engine.agents["isolde"], engine.agents["nib"]
    llm = Scripted({a.name: [True, True, True], b.name: [False, False, False]})
    c = await _talk(engine, a, b, llm)
    assert not c.intimate - {a.id}


async def test_a_pair_that_may_not_is_never_offered_the_field(tmp_path):
    engine = await _engine(tmp_path)
    a, b = engine.agents["isolde"], engine.agents["tobin"]
    a.bonds[b.id]["stance"] = b.bonds[a.id]["stance"] = "wary"
    a.bonds[b.id]["affinity"] = b.bonds[a.id]["affinity"] = 2.0
    llm = Scripted({a.name: [True] * 6, b.name: [True] * 6})
    await _talk(engine, a, b, llm)
    assert not any(llm.asked_for)


async def test_the_speech_path_and_the_digest_path_confirm_it_only_once(tmp_path):
    engine = await _engine(tmp_path)
    a, b = engine.agents["isolde"], engine.agents["nib"]
    llm = Scripted({a.name: [True], b.name: [True]})
    c = await _talk(engine, a, b, llm)
    await _settle(engine)
    from hollowmere.agents.agent import Digest
    engine._settle(c, {a.id: Digest(made_love=True), b.id: Digest(made_love=True)})  # the digest agrees as well
    await _settle(engine)
    assert sum(1 for e in engine.events if e["kind"] == "pregnancy") == 1
    assert sum(1 for m in a.memory.items if "made love" in m.text) == 1


async def test_given_no_proposal_when_the_talk_runs_on_then_it_ends_at_the_line_limit(tmp_path):
    # Given a pair that may, and nobody proposing anything
    engine = await _engine(tmp_path)
    a, b = engine.agents["isolde"], engine.agents["nib"]
    # When they keep talking
    c = await _talk(engine, a, b, Scripted({a.name: [], b.name: []}))
    # Then the talk is cut at the limit
    assert len(c.transcript) == MAX_TURNS


async def test_given_a_proposal_in_the_last_line_when_the_limit_is_reached_then_the_other_may_still_answer(tmp_path):
    # Given the last line is a proposal and the answer to it is yes
    engine = await _engine(tmp_path)
    a, b = engine.agents["isolde"], engine.agents["nib"]
    llm = Scripted({a.name: [False] * HALF + [True], b.name: [False] * (HALF - 1) + [True]})
    # When the talk reaches the limit
    c = await _talk(engine, a, b, llm)
    await _settle(engine)
    # Then one more line is spoken, and both said yes
    assert len(c.transcript) == MAX_TURNS + 1
    assert a.bonds[b.id]["intimate"] and b.bonds[a.id]["intimate"]


async def test_given_a_proposal_in_the_last_line_when_the_answer_is_no_then_the_talk_ends_after_that_one_line(tmp_path):
    # Given the last line is a proposal that is turned aside
    engine = await _engine(tmp_path)
    a, b = engine.agents["isolde"], engine.agents["nib"]
    llm = Scripted({a.name: [False] * (HALF + 1), b.name: [False] * (HALF - 1) + [True]})
    # When the talk reaches the limit
    c = await _talk(engine, a, b, llm)
    await _settle(engine)
    # Then exactly one more line is allowed, and nothing is confirmed
    assert len(c.transcript) == MAX_TURNS + 1
    assert not a.bonds[b.id].get("intimate")


async def test_given_a_proposal_earlier_when_the_limit_is_reached_then_it_is_not_stretched(tmp_path):
    # Given a proposal in line six that was answered with no, and nothing pending at the last line
    engine = await _engine(tmp_path)
    a, b = engine.agents["isolde"], engine.agents["nib"]
    llm = Scripted({a.name: [False] * HALF, b.name: [False, False, True]})
    # When the talk reaches the limit
    c = await _talk(engine, a, b, llm)
    # Then it ends at the limit
    assert len(c.transcript) == MAX_TURNS


async def _settle(engine):
    import asyncio
    await asyncio.gather(*[t for t in engine._bg.values() if not t.done()])


async def test_the_prompt_tells_the_listener_when_the_other_has_just_proposed(tmp_path):
    engine = await _engine(tmp_path)
    a = engine.agents["isolde"]
    base = dict(names=engine.ctx.names, now=NOW, place="the reeds", other_name="Nib", other_activity="waits",
                purpose="", transcript=[("Nib", "Come away with me.")], mems=[], romance_eligible=True)
    invited = prompts.converse(a, invited=True, **base)
    assert "has just made clear" in invited and "intimate to true" in invited
    assert "has just made clear" not in prompts.converse(a, invited=False, **base)


async def test_the_answer_is_asked_for_only_after_a_proposal(tmp_path):
    engine = await _engine(tmp_path)
    a, b = engine.agents["isolde"], engine.agents["nib"]
    seen: list[tuple[str, bool]] = []

    class Spy(Scripted):
        async def structured(self, system, user, schema, *, label, temperature=None):
            if label == "converse":
                speaker = next(n for n in self.flags if f"You are {n}." in user)
                seen.append((speaker, "has just made clear" in user))
            return await super().structured(system, user, schema, label=label, temperature=temperature)

    # Isolde opens, Nib proposes on his first line, Isolde answers yes on her second
    llm = Spy({a.name: [False, True], b.name: [True, False]})
    c = await _talk(engine, a, b, llm)
    await _settle(engine)
    assert seen[0] == (a.name, False)       # nothing proposed yet
    assert seen[1] == (b.name, False)       # the proposer is not asked to answer himself
    assert seen[2] == (a.name, True)        # she is told, and answers
    assert len(c.transcript) == 3 and c.intimate == {a.id, b.id}
    assert a.bonds[b.id].get("intimate") and b.bonds[a.id].get("intimate")


async def test_a_proposal_nobody_answers_is_not_repeated_to_the_proposer(tmp_path):
    engine = await _engine(tmp_path)
    a, b = engine.agents["isolde"], engine.agents["nib"]
    told: list[str] = []

    class Spy(Scripted):
        async def structured(self, system, user, schema, *, label, temperature=None):
            if label == "converse" and "has just made clear" in user:
                told.append(next(n for n in self.flags if f"You are {n}." in user))
            return await super().structured(system, user, schema, label=label, temperature=temperature)

    llm = Spy({a.name: [False, False, False], b.name: [True, False, False]})
    await _talk(engine, a, b, llm)
    assert set(told) == {a.name}  # only the one who has not said it herself is ever told
