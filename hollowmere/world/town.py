"""Hollowmere: a plague-haunted river town under a dying lord.

Coordinates are tiles, x to the right, y down. Rectangles are inclusive.
"""

from __future__ import annotations

import random

from hollowmere.world.world import Area, Light, Tile, World, WorldObject

W, H = 90, 58

TORCH = "#ff9a3c"
CANDLE = "#ffc66b"
HEARTH = "#ff7a2a"
FORGE = "#ff5a1f"
CAULDRON = "#7dff9a"
GOBLIN_FIRE = "#ff8a3c"


class _Builder:
    def __init__(self) -> None:
        self.w = World(W, H)

    def fill(self, x0: int, y0: int, x1: int, y1: int, tile: Tile) -> None:
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                self.w.tiles[y][x] = tile

    def set(self, x: int, y: int, tile: Tile) -> None:
        self.w.tiles[y][x] = tile

    def building(self, x0: int, y0: int, x1: int, y1: int, floor: Tile, doors: list[tuple[int, int]]) -> None:
        self.fill(x0, y0, x1, y1, Tile.WALL)
        self.fill(x0 + 1, y0 + 1, x1 - 1, y1 - 1, floor)
        for d in doors:
            self.set(*d, Tile.DOOR)

    def hwall(self, x0: int, x1: int, y: int, gaps: list[int], gap_tile: Tile = Tile.DOOR) -> None:
        for x in range(x0, x1 + 1):
            self.set(x, y, gap_tile if x in gaps else Tile.WALL)

    def vwall(self, x: int, y0: int, y1: int, gaps: list[int], gap_tile: Tile = Tile.DOOR) -> None:
        for y in range(y0, y1 + 1):
            self.set(x, y, gap_tile if y in gaps else Tile.WALL)

    def fence(self, x0: int, y0: int, x1: int, y1: int, gates: list[tuple[int, int]], ground: Tile) -> None:
        self.fill(x0, y0, x1, y1, Tile.FENCE)
        self.fill(x0 + 1, y0 + 1, x1 - 1, y1 - 1, ground)
        for g in gates:
            self.set(*g, ground)

    def area(self, id: str, name: str, desc: str, rect: tuple[int, int, int, int], *,
             public: bool, indoor: bool) -> None:
        a = self.w.areas.setdefault(id, Area(id, name, desc, public, indoor))
        x0, y0, x1, y1 = rect
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                prev = self.w.area_grid[y][x]
                if prev and prev != id:
                    self.w.areas[prev].tiles.remove((x, y))
                self.w.area_grid[y][x] = id
                a.tiles.append((x, y))

    def obj(self, id: str, name: str, area: str, x: int, y: int, kind: str,
            light: tuple[float, str] | None = None) -> None:
        assert self.w.area_grid[y][x] == area, f"{id} at {(x, y)} is not inside {area}"
        self.w.objects[id] = WorldObject(id, name, area, x, y, kind)
        if light:
            self.light(x + 0.5, y + 0.5, *light)

    def light(self, x: float, y: float, radius: float, color: str, flicker: bool = True) -> None:
        self.w.lights.append(Light(x, y, radius, color, flicker))


def build_town(seed: int = 1347) -> World:
    b = _Builder()
    rng = random.Random(seed)

    # --- terrain -----------------------------------------------------------
    b.fill(0, 0, W - 1, H - 1, Tile.GRASS)
    for y in range(H):
        for x in range(W):
            edge = min(x, y, W - 1 - x, H - 1 - y)
            if edge < 2 or (edge < 4 and rng.random() < 0.35):
                b.set(x, y, Tile.TREE)
    for _ in range(700):  # the Witchwood beyond the river
        x, y = rng.randint(60, W - 3), rng.randint(2, H - 3)
        b.set(x, y, Tile.TREE)
    for _ in range(40):
        x, y = rng.randint(3, 20), rng.randint(36, H - 3)
        b.set(x, y, Tile.TREE)
    for _ in range(140):  # worn mud patches
        x, y = rng.randint(3, 57), rng.randint(3, H - 4)
        if b.w.tiles[y][x] == Tile.GRASS:
            b.set(x, y, Tile.DIRT)

    # River Mere and the bridge
    b.fill(58, 0, 59, H - 1, Tile.WATER)
    b.fill(58, 22, 59, 23, Tile.BRIDGE)

    # Roads
    b.fill(2, 22, 57, 23, Tile.COBBLE)       # King's Road, east-west
    b.fill(30, 2, 31, 44, Tile.COBBLE)       # Gate Street, north-south
    b.fill(24, 18, 38, 27, Tile.COBBLE)      # the square
    b.fill(60, 15, 60, 30, Tile.DIRT)        # path along the far bank
    b.fill(60, 22, 61, 23, Tile.DIRT)

    # --- the Drowned Lantern (tavern) ---------------------------------------
    b.building(5, 6, 20, 19, Tile.WOOD, doors=[(12, 19), (13, 19)])
    b.hwall(6, 19, 11, gaps=[9, 16])
    b.vwall(13, 7, 10, gaps=[])
    b.fill(12, 20, 13, 21, Tile.COBBLE)
    b.area("tavern", "the Drowned Lantern", "a smoky tavern; the town's only warm room after dark",
           (6, 12, 19, 18), public=True, indoor=True)
    b.area("agnes_room", "Agnes's chamber above the bar", "the innkeeper's cramped bedroom",
           (6, 7, 12, 10), public=False, indoor=True)
    b.area("isolde_room", "Isolde's garret", "a narrow room with one window facing the road",
           (14, 7, 19, 10), public=False, indoor=True)
    b.obj("bar", "bar counter", "tavern", 8, 14, "counter")
    b.obj("tavern_hearth", "hearth", "tavern", 19, 15, "hearth", (6, HEARTH))
    b.obj("long_table", "long table", "tavern", 12, 16, "table", (3, CANDLE))
    b.obj("corner_table", "corner table", "tavern", 16, 13, "table")
    b.obj("casks", "ale casks", "tavern", 6, 17, "casks")
    b.obj("agnes_bed", "bed", "agnes_room", 7, 8, "bed")
    b.obj("agnes_strongbox", "strongbox", "agnes_room", 11, 8, "chest")
    b.obj("isolde_bed", "bed", "isolde_room", 18, 8, "bed")
    b.obj("isolde_window", "window", "isolde_room", 15, 7, "window")

    # --- the smithy ----------------------------------------------------------
    b.building(40, 6, 52, 17, Tile.STONE, doors=[(46, 17), (47, 17)])
    b.hwall(41, 51, 11, gaps=[43, 49])
    b.vwall(47, 7, 10, gaps=[])
    b.fill(46, 18, 47, 21, Tile.COBBLE)
    b.area("smithy", "Hale's smithy", "a soot-black forge that rings from dawn",
           (41, 12, 51, 16), public=True, indoor=True)
    b.area("brannoc_room", "Brannoc's quarters", "a bare room with an old soldier's kit on the wall",
           (41, 7, 46, 10), public=False, indoor=True)
    b.area("tobin_loft", "the apprentice's loft", "a cot among scrap iron and charcoal sacks",
           (48, 7, 51, 10), public=False, indoor=True)
    b.obj("forge", "forge", "smithy", 42, 13, "forge", (7, FORGE))
    b.obj("anvil", "anvil", "smithy", 44, 14, "anvil")
    b.obj("quench_barrel", "quenching barrel", "smithy", 46, 13, "barrel")
    b.obj("weapon_rack", "weapon rack", "smithy", 50, 13, "rack")
    b.obj("workbench", "workbench", "smithy", 50, 15, "table")
    b.obj("brannoc_bed", "bed", "brannoc_room", 42, 8, "bed")
    b.obj("war_chest", "old war chest", "brannoc_room", 45, 8, "chest")
    b.obj("tobin_cot", "cot", "tobin_loft", 50, 8, "bed")

    # --- the watchhouse ------------------------------------------------------
    b.building(22, 6, 28, 14, Tile.STONE, doors=[(28, 9)])
    b.hwall(23, 27, 11, gaps=[25])
    b.fill(29, 9, 29, 9, Tile.COBBLE)
    b.area("watchhouse", "the watchhouse", "the Lord's guardroom, smelling of oil and old fear",
           (23, 7, 27, 10), public=True, indoor=True)
    b.area("roderic_room", "the captain's quarters", "a tidy cell of a room behind the guardroom",
           (23, 12, 27, 13), public=False, indoor=True)
    b.obj("duty_desk", "duty desk", "watchhouse", 26, 9, "table")
    b.obj("arms_rack", "arms rack", "watchhouse", 24, 8, "rack")
    b.obj("brazier", "brazier", "watchhouse", 23, 10, "brazier", (5, HEARTH))
    b.obj("roderic_bed", "bed", "roderic_room", 24, 13, "bed")
    b.obj("lockbox", "lockbox of tax rolls", "roderic_room", 27, 12, "chest")

    # --- the north gate --------------------------------------------------------
    b.fill(26, 2, 35, 4, Tile.COBBLE)
    b.area("north_gate", "the north gate", "the barred town gate on the road to the Lord's keep",
           (26, 2, 35, 4), public=True, indoor=False)
    b.obj("gate", "gate", "north_gate", 30, 2, "gate")
    b.light(28.5, 3.5, 4, TORCH)
    b.light(33.5, 3.5, 4, TORCH)

    # --- the square ----------------------------------------------------------------
    b.area("square", "the market square", "a muddy cobbled square ringed by shuttered stalls",
           (24, 18, 38, 27), public=True, indoor=False)
    b.obj("well", "well", "square", 27, 20, "well")
    b.obj("gallows", "gallows", "square", 36, 26, "gallows")
    b.obj("market_stall", "market stall", "square", 34, 19, "stall")
    b.obj("notice_post", "notice post", "square", 25, 25, "post")
    for tx, ty in [(24.5, 18.5), (38.5, 18.5), (24.5, 27.5), (38.5, 27.5)]:
        b.light(tx, ty, 4.5, TORCH)
    for tx in (8.5, 18.5, 44.5, 54.5):
        b.light(tx, 21.5, 3.5, TORCH)

    # --- the chapel of the Pale Saint ----------------------------------------------
    b.building(36, 27, 48, 39, Tile.STONE, doors=[(41, 27), (42, 27), (48, 31)])
    b.hwall(37, 47, 35, gaps=[39, 45])
    b.vwall(42, 36, 38, gaps=[])
    b.fill(41, 24, 42, 26, Tile.COBBLE)
    b.area("chapel", "the chapel of the Pale Saint", "cold stone, guttering candles, too many names read aloud",
           (37, 28, 47, 34), public=True, indoor=True)
    b.area("odo_cell", "the priest's cell", "a cell with a straw pallet and a cracked icon",
           (37, 36, 41, 38), public=False, indoor=True)
    b.area("crypt", "the crypt stair", "steps down to the founders' tombs; the air tastes of earth",
           (43, 36, 47, 38), public=False, indoor=True)
    b.obj("altar", "altar", "chapel", 42, 33, "altar", (4, CANDLE))
    b.obj("candle_rack", "votive candles", "chapel", 45, 33, "candles", (3, CANDLE))
    b.obj("pews", "pews", "chapel", 40, 30, "bench")
    b.obj("odo_bed", "pallet", "odo_cell", 38, 37, "bed")
    b.obj("icon", "cracked icon", "odo_cell", 40, 36, "icon", (2, CANDLE))
    b.obj("tomb", "founder's tomb", "crypt", 46, 37, "tomb")

    # --- graveyard and the gravedigger's shack --------------------------------------
    b.fence(50, 26, 57, 37, gates=[(50, 31), (53, 37)], ground=Tile.GRAVE_SOIL)
    b.set(49, 31, Tile.DIRT)
    b.area("graveyard", "the graveyard", "crooked stones and fresh mounds; the plague pits lie at the back",
           (51, 27, 56, 36), public=True, indoor=False)
    b.obj("fresh_graves", "fresh graves", "graveyard", 53, 29, "grave")
    b.obj("plague_pit", "plague pit", "graveyard", 55, 34, "pit")
    b.obj("old_stones", "old headstones", "graveyard", 52, 33, "grave")
    b.obj("dead_yew", "dead yew", "graveyard", 56, 27, "tree")
    b.building(50, 39, 56, 43, Tile.WOOD, doors=[(53, 39)])
    b.set(53, 38, Tile.DIRT)
    b.area("silas_shack", "the gravedigger's shack", "a lean-to of tarred planks, spades by the door",
           (51, 40, 55, 42), public=False, indoor=True)
    b.obj("silas_cot", "cot", "silas_shack", 51, 41, "bed")
    b.obj("spade_rack", "spades", "silas_shack", 55, 40, "rack")
    b.obj("stove", "iron stove", "silas_shack", 55, 42, "stove", (3, HEARTH))

    # --- across the river: Wenna's hut and herb garden ------------------------------
    b.building(61, 13, 67, 19, Tile.WOOD, doors=[(61, 16)])
    b.area("wenna_hut", "the herb-wife's hut", "bundles of drying roots, a black kettle, smoke that smells sweet",
           (62, 14, 66, 18), public=False, indoor=True)
    b.obj("cauldron", "cauldron", "wenna_hut", 64, 16, "cauldron", (4, CAULDRON))
    b.obj("wenna_bed", "bed", "wenna_hut", 66, 14, "bed")
    b.obj("drying_herbs", "drying herbs", "wenna_hut", 62, 18, "herbs")
    b.fence(61, 24, 67, 30, gates=[(61, 26)], ground=Tile.FIELD)
    b.area("herb_garden", "the herb garden", "rows of nightshade, feverfew and things with no common name",
           (62, 25, 66, 29), public=True, indoor=False)
    b.obj("herb_beds", "herb beds", "herb_garden", 64, 27, "plants")
    b.obj("beehives", "beehives", "herb_garden", 66, 29, "hive")
    b.area("riverbank", "the riverbank", "reeds and black water under the bridge",
           (54, 10, 57, 20), public=True, indoor=False)
    b.obj("fishing_rock", "flat rock", "riverbank", 57, 14, "rock")

    # --- fields and the standing stones ----------------------------------------------
    b.fence(4, 26, 20, 34, gates=[(12, 26)], ground=Tile.FIELD)
    b.fill(12, 24, 12, 25, Tile.DIRT)
    b.area("fields", "the barley fields", "blighted barley no one has the hands to harvest",
           (5, 27, 19, 33), public=True, indoor=False)
    b.obj("scarecrow", "scarecrow", "fields", 12, 30, "scarecrow")
    b.fill(6, 37, 14, 42, Tile.GRASS)
    b.fill(12, 35, 12, 36, Tile.DIRT)
    b.fill(12, 34, 12, 34, Tile.FENCE)  # keep the field closed; the stones are reached from the west
    b.fill(3, 35, 12, 36, Tile.DIRT)
    b.fill(3, 24, 3, 35, Tile.DIRT)
    b.area("standing_stones", "the Weeping Stones", "a ring of old stones the priest calls heathen",
           (6, 37, 14, 42), public=True, indoor=False)
    b.obj("stone_ring", "stone ring", "standing_stones", 10, 39, "stones")

    # --- the bakery, off Gate Street ----------------------------------------------------
    b.building(21, 29, 28, 37, Tile.STONE, doors=[(28, 31)])
    b.hwall(22, 27, 34, gaps=[24])
    b.set(29, 31, Tile.COBBLE)
    b.area("bakery", "the bakery", "a warm shop smelling of rye and ash; the shelves are mostly bare",
           (22, 30, 27, 33), public=True, indoor=True)
    b.area("elsbeth_room", "the baker's back room", "a narrow bed beside the flour sacks",
           (22, 35, 27, 36), public=False, indoor=True)
    b.obj("bread_oven", "bread oven", "bakery", 22, 30, "oven", (5, HEARTH))
    b.obj("kneading_table", "kneading table", "bakery", 25, 31, "table")
    b.obj("bread_shelf", "bread shelf", "bakery", 27, 33, "shelf")
    b.obj("elsbeth_bed", "bed", "elsbeth_room", 23, 36, "bed")
    b.obj("flour_sacks", "flour sacks", "elsbeth_room", 27, 35, "sacks")

    # --- the south lane: weaver, almshouse, wash house -------------------------------------
    b.fill(24, 44, 57, 44, Tile.DIRT)
    b.fill(30, 44, 31, 44, Tile.COBBLE)

    b.building(20, 45, 27, 51, Tile.WOOD, doors=[(24, 45)])
    b.area("weaver_cottage", "the weaver's cottage", "a loom, bolts of grey wool, and a stack of finished shrouds",
           (21, 46, 26, 50), public=True, indoor=True)
    b.obj("loom", "loom", "weaver_cottage", 22, 47, "loom")
    b.obj("spinning_wheel", "spinning wheel", "weaver_cottage", 25, 47, "wheel")
    b.obj("shrouds", "stack of shrouds", "weaver_cottage", 26, 50, "cloth")
    b.obj("mara_bed", "bed", "weaver_cottage", 21, 50, "bed", (2, CANDLE))

    b.building(33, 45, 44, 53, Tile.STONE, doors=[(38, 45)])
    b.hwall(34, 43, 50, gaps=[36])
    b.area("almshouse", "the almshouse", "the plague ward: straw cots, vinegar, coughing in the dark",
           (34, 46, 43, 49), public=True, indoor=True)
    b.area("clemence_cell", "the sister's cell", "a bare cell with a book hidden under the pallet",
           (34, 51, 43, 52), public=False, indoor=True)
    b.obj("sickbeds", "sickbeds", "almshouse", 37, 47, "sickbeds", (3, CANDLE))
    b.obj("water_basin", "basin", "almshouse", 34, 49, "basin")
    b.obj("remedy_table", "remedy table", "almshouse", 42, 47, "table")
    b.obj("clemence_bed", "pallet", "clemence_cell", 35, 52, "bed")
    b.obj("clemence_book", "hidden book", "clemence_cell", 41, 51, "book")

    b.building(48, 46, 55, 51, Tile.WOOD, doors=[(51, 46), (55, 48)])
    b.area("ros_room", "the wash house", "steaming tubs and a cot in the corner",
           (49, 47, 54, 50), public=False, indoor=True)
    b.obj("wash_tubs", "wash tubs", "ros_room", 50, 48, "tubs")
    b.obj("ros_bed", "cot", "ros_room", 54, 50, "bed", (2, HEARTH))
    b.fill(56, 45, 57, 53, Tile.DIRT)
    b.area("washing_place", "the washing stones", "flat stones at the water's edge where the town's linen is beaten clean",
           (56, 45, 57, 53), public=True, indoor=False)
    b.obj("wash_stones", "washing stones", "washing_place", 57, 49, "rock")
    b.obj("drying_lines", "drying lines", "washing_place", 56, 46, "lines")
    for tx in (30.5, 46.5):
        b.light(tx, 44.5, 3.5, TORCH)

    # --- the Witchwood and the ruin beyond it ----------------------------------------------
    # A trail from the far bank winds east to a ruined hill-fort the townsfolk call
    # the Hollow Keep. Beneath its rubble is the burrow of the last goblin.
    b.fill(62, 22, 66, 22, Tile.DIRT)
    b.fill(66, 21, 70, 21, Tile.DIRT)
    b.fill(70, 22, 73, 22, Tile.DIRT)
    b.fill(71, 15, 86, 33, Tile.GRASS)
    for y in range(16, 33):
        for x in range(73, 87):
            edge = x in (73, 86) or y in (16, 32)
            if edge:
                if rng.random() < 0.7:
                    b.set(x, y, Tile.WALL)
            elif rng.random() < 0.6:
                b.set(x, y, Tile.STONE)
    b.set(73, 22, Tile.DIRT)
    b.building(81, 17, 85, 21, Tile.STONE, doors=[(81, 19)])
    b.building(75, 26, 79, 30, Tile.DIRT, doors=[(79, 28)])
    b.area("ruin", "the Hollow Keep", "a ruined hill-fort in the Witchwood; the townsfolk say it is haunted",
           (74, 17, 85, 31), public=True, indoor=False)
    b.area("goblin_den", "the burrow under the rubble", "a dry hollow lined with stolen rags, smelling of smoke and musk",
           (76, 27, 78, 29), public=False, indoor=True)
    b.set(75, 19, Tile.STONE)
    b.obj("idols", "carved goblin idols", "ruin", 75, 19, "idols")
    b.set(79, 23, Tile.DIRT)
    b.obj("fire_pit", "fire pit", "ruin", 79, 23, "firepit", (3.5, GOBLIN_FIRE))
    b.obj("broken_altar", "broken altar", "ruin", 83, 19, "altar")
    b.obj("nib_nest", "nest of rags", "goblin_den", 76, 29, "bed")
    b.obj("hoard", "hoard of trinkets", "goblin_den", 78, 27, "chest")
    b.obj("tribe_shrine", "skulls of the tribe", "goblin_den", 76, 27, "shrine", (1.5, GOBLIN_FIRE))

    # hiding places on the edges of town
    b.fill(60, 31, 60, 37, Tile.DIRT)
    b.fill(61, 37, 66, 37, Tile.DIRT)
    b.area("witchwood", "the edge of the Witchwood", "black pines and bramble where the townsfolk do not go",
           (62, 33, 70, 41), public=True, indoor=False)
    b.set(66, 36, Tile.GRASS)
    b.obj("bramble_thicket", "bramble thicket", "witchwood", 66, 36, "thicket")
    b.set(55, 18, Tile.GRASS)
    b.obj("reeds", "tall reeds", "riverbank", 55, 18, "reeds")
    b.set(7, 41, Tile.GRASS)
    b.obj("brambles", "brambles", "standing_stones", 7, 41, "thicket")

    return b.w
