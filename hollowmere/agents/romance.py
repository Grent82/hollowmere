"""Romance and pregnancy: newborn generation and the eligibility rule for "making love"."""

from __future__ import annotations

import random
from datetime import date
from typing import TYPE_CHECKING

from hollowmere.agents.persona import Persona, RANK, starting_affinity

if TYPE_CHECKING:
    from hollowmere.agents.agent import Agent

MALE_NAMES = ("Ned", "Fenwick", "Garrick", "Aldous", "Wymer")
FEMALE_NAMES = ("Maude", "Thea", "Rosamund", "Enid", "Lyra")
GOBLIN_NAMES = ("Grik", "Snurt", "Tam", "Pell", "Mossa")


def eligible_for_love(self_sex: str, other_sex: str, bond: dict | None) -> bool:
    """Whether this agent's own point of view makes "making love" with `other` a reachable outcome
    of a conversation - opposite sex, and either already intimate or already fond or better."""
    if bond is None or self_sex == other_sex:
        return False
    if bond.get("intimate"):
        return True
    return RANK[bond["stance"]] >= RANK["fond"]


def generate_child_persona(mother: Persona, father: Persona, born_on: date, rng: random.Random) -> Persona:
    """A brand-new Persona for a newborn: the father's species, the mother's home, both parents
    named as family. Deterministic given `rng`'s state - callers pass the engine's one shared,
    ever-advancing rng, so two births never actually draw from the same state."""
    sex = rng.choice(["male", "female"])
    species = father.species
    if species == "goblin":
        first = rng.choice(GOBLIN_NAMES)
        name = first
    else:
        first = rng.choice(MALE_NAMES if sex == "male" else FEMALE_NAMES)
        name = f"{first} {father.name.split()[-1]}"
    child_id = f"{first.lower()}-{rng.randrange(1_000_000):06d}"
    return Persona(
        id=child_id, name=name, age=0, sex=sex, color=mother.color,
        traits="too young yet to show any temperament.",
        background=f"{name} is the child of {mother.name} and {father.name}, born on {born_on.isoformat()}.",
        currently="a newborn, who has yet to do anything but exist.",
        lifestyle="sleeps most of the day, waking only to feed and cry.",
        home=mother.home, home_object=mother.home_object, work="",
        start=f"{mother.home}/{mother.home_object}",
        relationships={mother.id: "my mother", father.id: "my father"},
        stances={mother.id: "fond", father.id: "fond"},
        species=species,
    )


def recognize_child(mother: "Agent", father: "Agent", child: "Agent") -> None:
    """Make both parents' own bonds mechanically aware of the child - Persona.relationships isn't
    persisted across restarts, only Agent.bonds is, so this has to be a direct write to that dict."""
    for parent in (mother, father):
        parent.bonds[child.id] = {"stance": "fond", "affinity": starting_affinity("fond"),
                                  "note": "", "talks": 0, "last": None}
