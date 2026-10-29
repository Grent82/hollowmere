"""Things the townsfolk kept reaching for in the evening reports, now in the town where they plainly belong."""

import pytest

from hollowmere.world.town import build_town

WANTED = [
    ("mill_hopper", "hopper", "mill"),
    ("whetstone", "whetstone", "smithy"),
    ("mortar", "mortar and pestle", "wenna_hut"),
    ("ward_lamp", "lamp", "almshouse"),
    ("leather_strop", "leather strop", "smithy"),
    ("linen_flour_sack", "linen flour sack", "mill"),
    ("mill_stove", "mill stove", "mill"),
    ("stew_pot", "stew pot", "bakery"),
    ("wool_bales", "bales of grey wool", "weaver_cottage"),
]


@pytest.mark.parametrize("obj_id,name,area", WANTED)
def test_given_the_town_when_built_then_the_wanted_thing_stands_where_it_belongs(obj_id, name, area):
    # Given the town as built
    world = build_town()
    # Then the object is there, named, in its place
    obj = world.objects[obj_id]
    assert obj.name == name and obj.area == area


def test_given_the_new_things_when_the_town_is_built_then_none_blocks_a_door_or_shares_a_tile():
    # Given the town as built
    world = build_town()
    tiles = [(o.x, o.y) for o in world.objects.values()]
    # Then no two objects stand on one tile
    assert len(tiles) == len(set(tiles))
