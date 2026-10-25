from hollowmere.agents.gods import GOD_BY_ID, GODS
from hollowmere.agents.prompts import WORLD_FACTS


def test_exactly_the_four_known_gods_are_present():
    assert {g.id for g in GODS} == {"caedros", "yevra", "agrath", "sevreth"}


def test_every_god_has_all_the_prompt_steering_fields():
    for g in GODS:
        assert g.domain and g.temperament and g.response_bias and g.voice, g.id


def test_gods_json_does_not_contradict_world_facts():
    for g in GODS:
        assert g.name in WORLD_FACTS, g.id


def test_god_by_id_is_keyed_by_id():
    assert GOD_BY_ID["yevra"].domain == "love, lust and childbearing"
