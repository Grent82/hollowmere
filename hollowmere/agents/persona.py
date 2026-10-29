"""Persona seeds: the fixed identity each agent starts with."""

from __future__ import annotations

import json
from pathlib import Path

from typing import Literal

from pydantic import BaseModel, Field

DATA = Path(__file__).resolve().parent.parent / "data"


Stance = Literal["fear", "dislike", "wary", "neutral", "fond", "trust", "close", "love"]
STANCES: tuple[str, ...] = Stance.__args__
RANK: dict[str, int] = {s: i for i, s in enumerate(STANCES)}

# How much of the gap between the current affinity and a freshly judged feeling closes in one
# talk. Trust is earned slowly; fear or betrayal can close the gap at once. Both are also scaled
# by that talk's importance (1-10), so a trivial chat barely moves the needle either way.
AFFINITY_STEP_UP = 0.5
AFFINITY_STEP_DOWN = 0.55


def starting_affinity(stance: str) -> float:
    """The affinity score a bare stance label (no history yet) corresponds to."""
    return float(RANK[stance])


def update_affinity(current: float, feeling: Stance, importance: int) -> float:
    """Move the continuous affinity score (0 = fear .. len(STANCES)-1 = trust) toward `feeling`,
    by at most the full remaining gap, so a single talk can never overshoot past what was judged."""
    target = RANK[feeling]
    delta = target - current
    step = AFFINITY_STEP_DOWN if delta < 0 else AFFINITY_STEP_UP
    applied = step * (0.5 + importance / 10) * delta
    if abs(applied) > abs(delta):
        applied = delta
    return max(0.0, min(len(STANCES) - 1, current + applied))


def affinity_label(affinity: float) -> Stance:
    # Python's round() breaks exact .5 ties toward even, which would make the tie-break direction
    # depend on which rank it lands between; round half up instead, consistently.
    return STANCES[min(len(STANCES) - 1, int(affinity + 0.5))]


Sex = Literal["male", "female"]
CHILD_AGE = 3  # what a newborn becomes when it grows into a child
MINOR_AGE = 14  # below this: dependent on a parent, no love, no leaving alone


class Persona(BaseModel):
    id: str
    name: str
    known_as: str = ""  # what anyone in town knows of them; empty for someone the town does not know
    age: int
    sex: Sex
    color: str
    traits: str                  # temperament only; it is read as character, never as looks
    states: list[str] = Field(default_factory=list)  # passing states ("grieving") they start with; the day review keeps them up
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
    # family, as ids. Write a tie on one side only: Kinship reads the other side from it
    parents: list[str] = Field(default_factory=list)  # mother and father; also set for anyone born in the run
    siblings: list[str] = Field(default_factory=list)
    spouse: str | None = None
    aunt_or_uncle: list[str] = Field(default_factory=list)  # their parents' siblings, when the parents are not people

    @property
    def infant(self) -> bool:
        return self.age < 1

    @property
    def child(self) -> bool:
        """A small child: past the newborn, still kept close to whoever looks after it."""
        return 1 <= self.age < MINOR_AGE

    @property
    def dependent(self) -> bool:
        """A newborn or a child: carried or led along, never on its own, no part in love or leaving."""
        return self.age < MINOR_AGE

    @property
    def is_goblin(self) -> bool:
        """Goblin stealth (hiding, being spotted) applies to grown goblins; a baby or child in its parent's care is no secret."""
        return self.species == "goblin" and not self.dependent

    @property
    def is_human(self) -> bool:
        """Anyone who is not of the goblin kind. A goblin baby is no secret (see is_goblin), yet it is kin to
        goblins, never a stranger who could give one away."""
        return self.species != "goblin"

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
