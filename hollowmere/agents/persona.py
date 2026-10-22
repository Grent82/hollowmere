"""Persona seeds: the fixed identity each agent starts with."""

from __future__ import annotations

import json
from pathlib import Path

from typing import Literal

from pydantic import BaseModel, Field

DATA = Path(__file__).resolve().parent.parent / "data"


Stance = Literal["fear", "dislike", "wary", "neutral", "fond", "trust"]
STANCES: tuple[str, ...] = Stance.__args__


class Persona(BaseModel):
    id: str
    name: str
    known_as: str = ""  # what anyone in town knows of them; empty for someone the town does not know
    age: int
    color: str
    traits: str                  # temperament only; it is read as character, never as looks
    appearance: str = ""         # what others see (and smell) of them
    background: str
    currently: str
    lifestyle: str
    home: str
    home_object: str
    work: str
    relationships: dict[str, str] = Field(default_factory=dict)
    stances: dict[str, Stance] = Field(default_factory=dict)  # how they feel about those people at the start
    start: str
    species: str = "human"
    abilities: str = ""          # gifts and weaknesses the agent knows it has
    resists_charm: bool = False  # sees through goblin musk

    @property
    def is_goblin(self) -> bool:
        return self.species == "goblin"

    @property
    def first_name(self) -> str:
        parts = self.name.split()
        return parts[1] if parts[0] in {"Father", "Captain", "Mother", "Ser", "Sister"} and len(parts) > 1 else parts[0]

    @property
    def bed(self) -> str:
        return f"{self.home}/{self.home_object}"


def load_personas(path: Path | None = None) -> list[Persona]:
    path = path or DATA / "personas.json"
    return [Persona.model_validate(p) for p in json.loads(path.read_text(encoding="utf-8"))]
