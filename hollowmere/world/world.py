"""Tile world: terrain grid, named areas, interactable objects, and pathfinding.

The original used a Tiled export plus five parallel CSV layers to encode the
world:sector:arena:object hierarchy. Here the world is built in code
(see town.py) into one grid plus two small dictionaries.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from enum import IntEnum


class Tile(IntEnum):
    GRASS = 0
    DIRT = 1
    COBBLE = 2
    WOOD = 3
    STONE = 4
    WALL = 5
    WATER = 6
    TREE = 7
    DOOR = 8
    FENCE = 9
    GRAVE_SOIL = 10
    BRIDGE = 11
    FIELD = 12


WALKABLE = frozenset({Tile.GRASS, Tile.DIRT, Tile.COBBLE, Tile.WOOD, Tile.STONE, Tile.DOOR,
                      Tile.GRAVE_SOIL, Tile.BRIDGE, Tile.FIELD})


@dataclass
class Area:
    id: str
    name: str
    description: str
    public: bool
    indoor: bool
    tiles: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class WorldObject:
    id: str
    name: str
    area: str
    x: int
    y: int
    kind: str
    state: str = "idle"

    @property
    def place_id(self) -> str:
        return f"{self.area}/{self.id}"


@dataclass
class Light:
    x: float
    y: float
    radius: float
    color: str
    flicker: bool = True


class World:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.tiles: list[list[Tile]] = [[Tile.GRASS] * width for _ in range(height)]
        self.area_grid: list[list[str | None]] = [[None] * width for _ in range(height)]
        self.areas: dict[str, Area] = {}
        self.objects: dict[str, WorldObject] = {}
        self.lights: list[Light] = []
        self.retreats: list[str] = []  # areas two people can withdraw to; never offered to ordinary planning

    def extend_south(self, rows: int) -> None:
        """Append `rows` rows of grass below the map, leaving every existing tile where it was."""
        self.tiles.extend([Tile.GRASS] * self.width for _ in range(rows))
        self.area_grid.extend([None] * self.width for _ in range(rows))
        self.height += rows

    # ------------------------------------------------------------------ queries

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def walkable(self, x: int, y: int) -> bool:
        return self.in_bounds(x, y) and self.tiles[y][x] in WALKABLE

    def area_at(self, x: int, y: int) -> str | None:
        return self.area_grid[y][x] if self.in_bounds(x, y) else None

    def objects_in(self, area_id: str) -> list[WorldObject]:
        return [o for o in self.objects.values() if o.area == area_id]

    def place_name(self, place_id: str) -> str:
        """Human-readable name for "area" or "area/object"."""
        area_id, _, obj_id = place_id.partition("/")
        area = self.areas.get(area_id)
        if area is None:
            return place_id
        if obj_id and obj_id in self.objects:
            return f"the {self.objects[obj_id].name} in {area.name}"
        return area.name

    def resolve_place(self, place_id: str) -> tuple[str, WorldObject | None] | None:
        area_id, _, obj_id = place_id.partition("/")
        if area_id not in self.areas:
            return None
        obj = self.objects.get(obj_id) if obj_id else None
        if obj is not None and obj.area != area_id:
            obj = None
        return area_id, obj

    def target_tile(self, place_id: str, salt: int = 0) -> tuple[int, int] | None:
        """Tile an agent should walk to for a place: the object's tile, or a stable spot in the area."""
        resolved = self.resolve_place(place_id)
        if resolved is None:
            return None
        area_id, obj = resolved
        if obj is not None:
            return obj.x, obj.y
        tiles = [t for t in self.areas[area_id].tiles if self.walkable(*t)]
        return tiles[(salt * 7919) % len(tiles)] if tiles else None

    # ------------------------------------------------------------- pathfinding

    def find_path(self, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
        """A* over 4-connected walkable tiles. Returns the steps after `start` (empty if unreachable)."""
        if start == goal:
            return []
        if not self.walkable(*goal):
            return []
        gx, gy = goal
        open_heap: list[tuple[int, int, tuple[int, int]]] = [(0, 0, start)]
        came: dict[tuple[int, int], tuple[int, int]] = {}
        cost = {start: 0}
        while open_heap:
            _, g, cur = heapq.heappop(open_heap)
            if cur == goal:
                path = [cur]
                while path[-1] in came:
                    path.append(came[path[-1]])
                return path[-2::-1]
            if g > cost[cur]:
                continue
            x, y = cur
            for nxt in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if not self.walkable(*nxt):
                    continue
                # Roads are slightly cheaper so agents prefer them, like real townsfolk.
                step = 2 if self.tiles[nxt[1]][nxt[0]] == Tile.COBBLE else 3
                ng = g + step
                if ng < cost.get(nxt, 1 << 30):
                    cost[nxt] = ng
                    came[nxt] = cur
                    h = 2 * (abs(nxt[0] - gx) + abs(nxt[1] - gy))
                    heapq.heappush(open_heap, (ng + h, ng, nxt))
        return []

    # ---------------------------------------------------------- serialisation

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "tiles": [int(t) for row in self.tiles for t in row],
            "tileNames": {t.value: t.name.lower() for t in Tile},
            "areas": [{"id": a.id, "name": a.name, "indoor": a.indoor, "public": a.public,
                       "tiles": a.tiles} for a in self.areas.values()],
            "objects": [{"id": o.id, "name": o.name, "area": o.area, "x": o.x, "y": o.y,
                         "kind": o.kind, "state": o.state} for o in self.objects.values()],
            "lights": [{"x": l.x, "y": l.y, "radius": l.radius, "color": l.color, "flicker": l.flicker}
                       for l in self.lights],
        }
