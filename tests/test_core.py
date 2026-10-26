from datetime import date, datetime

import numpy as np
import pytest

from hollowmere.agents.agent import is_sleep, normalise_plan, parse_hhmm
from hollowmere.agents.persona import load_personas
from hollowmere.agents.schemas import day_plan_schema
from hollowmere.llm.client import extract_json
from hollowmere.memory.embed import HashingEmbedder
from hollowmere.memory.store import Store
from hollowmere.memory.stream import MemoryStream
from hollowmere.world.town import build_town


def test_every_persona_has_a_sex():
    for p in load_personas():
        assert p.sex in ("male", "female"), p.id


def test_every_object_and_home_is_reachable():
    w = build_town()
    start = w.target_tile("square/well")
    for o in w.objects.values():
        assert w.walkable(o.x, o.y), o.id
        assert (o.x, o.y) == start or w.find_path(start, (o.x, o.y)), o.id
    for p in load_personas():
        assert p.home in w.areas and p.work in w.areas
        assert w.resolve_place(p.bed)[1] is not None, p.bed
        assert w.target_tile(p.start) is not None, p.start


def test_every_light_has_a_source_object():
    w = build_town()
    for light in w.lights:
        x, y = int(light.x), int(light.y)
        assert any(o.x == x and o.y == y for o in w.objects.values()), (light.x, light.y)


def test_extract_json_tolerates_think_and_fences():
    reply = '<think>hmm {"no": 1}</think>\nSure!\n```json\n{"a": 1, "b": [2]}\n```'
    assert extract_json(reply) == {"a": 1, "b": [2]}
    with pytest.raises(ValueError):
        extract_json("<think>never finished {")


def test_place_enum_rejects_unknown_places():
    schema = day_plan_schema(("tavern", "smithy/anvil"))
    ok = {"morning_thought": "x", "wake_up": "06:00",
          "blocks": [{"start": "06:00", "end": "07:00", "activity": "a", "pray_to": "none", "place": "tavern"}] * 3}
    schema.model_validate(ok)
    bad = {**ok, "blocks": [{**ok["blocks"][0], "place": "castle"}] * 3}
    with pytest.raises(Exception):
        schema.model_validate(bad)


def test_day_plan_block_requires_pray_to_naming_a_known_god_or_none():
    schema = day_plan_schema(("chapel",))
    ok = {"morning_thought": "x", "wake_up": "06:00",
          "blocks": [{"start": "06:00", "end": "07:00", "activity": "prays",
                      "pray_to": "caedros", "place": "chapel"}] * 3}
    schema.model_validate(ok)
    bad = {**ok, "blocks": [{**ok["blocks"][0], "pray_to": "thor"}] * 3}
    with pytest.raises(Exception):
        schema.model_validate(bad)


def test_divine_answer_form_is_constrained_to_the_six_known_kinds():
    from typing import get_args

    from hollowmere.agents.schemas import DivineAnswer
    assert set(get_args(DivineAnswer.model_fields["form"].annotation)) == {
        "dream", "sign", "omen", "coincidence", "buff", "silence"}


def test_normalise_plan_fills_gaps_and_ends_in_bed():
    day = date(1347, 11, 2)
    wake = parse_hhmm(day, "06:00")
    blocks = normalise_plan(day, wake, [
        ("07:00", "12:00", "works", "smithy", None),
        ("11:00", "13:00", "eats", "tavern", None),      # overlaps
        ("15:00", "01:00", "drinks", "tavern", None),    # after a gap, past midnight
    ], "brannoc_room/brannoc_bed")
    assert blocks[0].start == wake
    for a, b in zip(blocks, blocks[1:]):
        assert a.end == b.start
    assert blocks[2].end.day == 3 and blocks[2].end.hour == 1
    assert is_sleep(blocks[-1].activity) and blocks[-1].place == "brannoc_room/brannoc_bed"


async def test_retrieval_prefers_relevant_and_important():
    store = Store(":memory:")
    mem = MemoryStream("t", store, HashingEmbedder(256))
    now = datetime(1347, 11, 2, 12)
    await mem.add("observation", "Brannoc is hammering at the anvil", now, 2)
    await mem.add("observation", "The well water smells foul", now, 7)
    await mem.add("observation", "Agnes is wiping the bar counter", now, 2)
    top = await mem.retrieve(["is the well water safe to drink"], now, k=1)
    assert top[0].text == "The well water smells foul"
    # survives a reload with embeddings intact
    again = MemoryStream("t", store, HashingEmbedder(256))
    await again.load()
    assert len(again.items) == 3 and again._vecs.shape == (3, 256)
    assert np.allclose(np.linalg.norm(again._vecs, axis=1), 1, atol=1e-5)


def test_times_use_a_pattern_llama_cpp_can_enforce():
    import re

    from hollowmere.agents.schemas import HHMM
    assert r"\d" not in HHMM  # llama.cpp's grammar does not enforce \d, so "24:00" got through
    assert re.match(HHMM, "06:00") and re.match(HHMM, "23:59") and not re.match(HHMM, "24:00")


def test_a_commitment_to_meet_someone_is_never_filed_as_only_their_errand():
    """Regression for a real bug: Isolde demanded Nib meet her at the flat rock, her own digest
    filed the commitment as who="them" ("his errand"), so her own code skipped scheduling her -
    she never showed up to the meeting she herself proposed. A meeting needs both people present,
    so the schema must say so explicitly instead of leaving it to the model to infer from grammar."""
    from typing import get_args

    from hollowmere.agents.schemas import chat_digest_schema
    schema = chat_digest_schema(("tavern",))
    commitment = get_args(schema.model_fields["commitments"].annotation)[0]
    who_description = commitment.model_fields["who"].description
    assert "meeting" in who_description.lower()
    assert "both" in who_description.lower()


def test_the_feeling_judgment_must_follow_the_talk_not_the_towns_general_dread():
    """Regression: almost every relationship in a real run converged to exactly 'wary', because the
    model judges 'feeling' from the town's ambient suspicion rather than what was actually said in
    that specific talk. The schema must say so explicitly, and as a positive default rather than only
    a negation - smaller models follow "do X" more reliably than "don't default to Y"."""
    from hollowmere.agents.schemas import chat_digest_schema
    schema = chat_digest_schema(("tavern",))
    feeling_description = schema.model_fields["feeling"].description.lower()
    assert "this talk" in feeling_description
    assert "dread" in feeling_description or "suspicion" in feeling_description or "fear" in feeling_description
    assert "neutral: an ordinary exchange" in feeling_description


def test_the_feeling_judgment_names_what_each_stance_looks_like_in_a_talk():
    """Replaying 40 stored talks through the real model: with a bare label list, 62% of judgments were
    'wary' even for talks an outside reader found friendly; naming the concrete behaviour behind each
    stance cut that to 45% and tripled 'trust'. The 'dread is not wariness' sentence alone changed
    nothing, so the anchors are what has to stay."""
    from hollowmere.agents.schemas import chat_digest_schema
    feeling_description = chat_digest_schema(("tavern",)).model_fields["feeling"].description.lower()
    for anchor in ("wary: this talk gave you a specific reason", "fond: you enjoyed their company",
                   "trust: they confided in you"):
        assert anchor in feeling_description


def test_commitment_day_and_importance_fields_carry_their_own_guidance():
    """Regression: `day` had no Field(description=...) at all (unlike every sibling field), and
    `importance` had none either - both left the model with nothing to anchor a judgement on."""
    from typing import get_args

    from hollowmere.agents.schemas import chat_digest_schema
    schema = chat_digest_schema(("tavern",))
    assert schema.model_fields["importance"].description
    commitment = get_args(schema.model_fields["commitments"].annotation)[0]
    assert commitment.model_fields["day"].description


def test_the_time_field_carries_the_vague_hour_rule_instead_of_the_prose():
    """Moved out of chat_digest()'s prose: a schema field description is the more reliable home for
    a per-field rule, and it was previously one compound, double-negative sentence buried mid-prompt."""
    from typing import get_args

    from hollowmere.agents.schemas import chat_digest_schema
    schema = chat_digest_schema(("tavern",))
    commitment = get_args(schema.model_fields["commitments"].annotation)[0]
    time_description = commitment.model_fields["time"].description.lower()
    assert "vague" in time_description or "dusk" in time_description


def test_commitments_field_explains_the_cap_and_how_to_handle_a_maybe():
    from typing import get_args

    from hollowmere.agents.schemas import chat_digest_schema
    schema = chat_digest_schema(("tavern",))
    description = schema.model_fields["commitments"].description.lower()
    assert "matter most" in description
    assert "maybe" in description


def test_the_goblin_cannot_plan_to_go_indoors_in_town():
    from hollowmere.agents.agent import Agent
    from hollowmere.agents.persona import load_personas
    from hollowmere.world.town import build_town
    world = build_town()
    nib = Agent(next(p for p in load_personas() if p.is_goblin), None, world)
    assert "tavern" not in nib.known_areas and "bakery" not in nib.known_areas
    assert "goblin_den" in nib.known_areas and "graveyard" in nib.known_areas


async def test_retrieval_survives_memories_added_while_it_waits_for_embeddings(tmp_path):
    import asyncio

    from hollowmere.memory.embed import HashingEmbedder
    from hollowmere.memory.store import Store
    from hollowmere.memory.stream import MemoryStream

    class SlowEmbedder(HashingEmbedder):
        async def embed(self, texts):  # a network round trip: other coroutines run meanwhile
            await asyncio.sleep(0.05 if texts == ["river"] else 0.01)  # the add finishes first
            return await super().embed(texts)

    from datetime import datetime
    now = datetime(1347, 11, 2, 9)
    stream = MemoryStream("ros", Store(str(tmp_path / "race.db")), SlowEmbedder(64))
    for i in range(5):
        await stream.add("observation", f"the river is high {i}", now, 3)
    found, _ = await asyncio.gather(stream.retrieve(["river"], now, k=3),
                                    stream.add("observation", "a crow on the wall", now, 3))
    assert len(found) == 3


def test_store_events_since_filters_by_time_and_kind():
    store = Store(":memory:")
    t0 = datetime(1347, 11, 2, 5, 0)
    store.add_event(t0, "chat", [], "small talk", {})
    store.add_event(t0.replace(hour=6), "ally", ["a", "b"], "a protects b", {})
    store.add_event(t0.replace(hour=7), "spurn", ["a", "b"], "a drives b off", {})
    assert store.events_since(t0, {"ally", "spurn"}) == ["a protects b", "a drives b off"]
    assert store.events_since(t0.replace(hour=6, minute=30), {"ally", "spurn"}) == ["a drives b off"]


def test_chat_log_for_includes_only_this_agents_talks_with_speaker_partner_and_convo():
    store = Store(":memory:")
    t0 = datetime(1347, 11, 2, 14, 0)
    store.add_event(t0, "chat", ["agnes", "odo"], "Good morrow, Odo.", {"speaker": "agnes", "convo": "c1"})
    store.add_event(t0.replace(minute=1), "chat", ["agnes", "odo"], "And to you.",
                    {"speaker": "odo", "convo": "c1"})
    store.add_event(t0.replace(minute=2), "chat_end", ["agnes", "odo"], "Agnes and Odo part ways.",
                    {"convo": "c1", "reason": "goodbye", "purpose": ""})
    store.add_event(t0.replace(hour=16), "chat", ["wenna", "odo"], "a talk agnes has no part in",
                    {"speaker": "wenna", "convo": "c2"})

    log = store.chat_log_for("agnes")
    assert [e["text"] for e in log] == ["Good morrow, Odo.", "And to you.", "Agnes and Odo part ways."]
    assert [e["with"] for e in log] == ["odo", "odo", "odo"]
    assert [e["convo"] for e in log] == ["c1", "c1", "c1"]
    assert log[0]["speaker"] == "agnes" and log[1]["speaker"] == "odo"
    assert log[2]["kind"] == "chat_end"


def test_chat_log_for_caps_at_the_given_number_of_lines_keeping_the_most_recent():
    store = Store(":memory:")
    t0 = datetime(1347, 11, 2, 14, 0)
    for i in range(5):
        store.add_event(t0.replace(minute=i), "chat", ["agnes", "odo"], f"line {i}",
                        {"speaker": "agnes", "convo": "c1"})
    log = store.chat_log_for("agnes", limit=2)
    assert [e["text"] for e in log] == ["line 3", "line 4"]


def test_prayers_for_returns_only_this_agents_prayers_newest_first_without_the_name_prefix():
    store = Store(":memory:")
    t0 = datetime(1347, 11, 2, 5, 0)
    store.add_event(t0, "prayer", ["odo"], "Odo prays to Caedros: a quiet dream: of ash",
                    {"form": "dream", "god": "caedros"})
    store.add_event(t0.replace(hour=6), "prayer", ["wenna"], "Wenna prays to Sevreth: a crow",
                    {"form": "sign", "god": "sevreth"})
    store.add_event(t0.replace(hour=7), "prayer", ["odo"], "Odo prays to Yevra: warmth",
                    {"form": "buff", "god": "yevra"})
    store.add_event(t0.replace(hour=8), "chat", ["odo", "wenna"], "not a prayer", {"speaker": "odo", "convo": "c1"})

    prayers = store.prayers_for("odo")
    assert [(p["god"], p["form"], p["text"]) for p in prayers] == [
        ("yevra", "buff", "warmth"), ("caedros", "dream", "a quiet dream: of ash")]
    assert prayers[0]["time"] > prayers[1]["time"]


def test_prayers_for_caps_at_the_given_limit_keeping_the_most_recent():
    store = Store(":memory:")
    t0 = datetime(1347, 11, 2, 5, 0)
    for i in range(5):
        store.add_event(t0.replace(minute=i), "prayer", ["odo"], f"Odo prays to Caedros: answer {i}",
                        {"form": "sign", "god": "caedros"})
    assert [p["text"] for p in store.prayers_for("odo", limit=2)] == ["answer 4", "answer 3"]
