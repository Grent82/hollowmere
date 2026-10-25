"""God seeds: the fixed, structured facts about the four gods, read by prompts and schemas."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from hollowmere.agents.persona import DATA


class God(BaseModel):
    id: str
    name: str
    domain: str
    temperament: str
    response_bias: str
    voice: str


def load_gods(path: Path | None = None) -> list[God]:
    path = path or DATA / "gods.json"
    return [God.model_validate(g) for g in json.loads(path.read_text(encoding="utf-8"))]


GODS: list[God] = load_gods()
GOD_BY_ID: dict[str, God] = {g.id: g for g in GODS}
GOD_IDS: tuple[str, ...] = tuple(GOD_BY_ID)
