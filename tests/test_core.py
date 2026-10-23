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
          "blocks": [{"start": "06:00", "end": "07:00", "activity": "a", "place": "tavern"}] * 3}
    schema.model_validate(ok)
    bad = {**ok, "blocks": [{**ok["blocks"][0], "place": "castle"}] * 3}
    with pytest.raises(Exception):
        schema.model_validate(bad)


def test_normalise_plan_fills_gaps_and_ends_in_bed():
    day = date(1347, 11, 2)
    wake = parse_hhmm(day, "06:00")
    blocks = normalise_plan(day, wake, [
        ("07:00", "12:00", "works", "smithy"),
        ("11:00", "13:00", "eats", "tavern"),      # overlaps
        ("15:00", "01:00", "drinks", "tavern"),    # after a gap, past midnight
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
