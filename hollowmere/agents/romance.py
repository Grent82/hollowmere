"""Romance and pregnancy: newborn generation and the eligibility rule for "making love"."""

from __future__ import annotations

import random
import re
from datetime import date
from typing import TYPE_CHECKING

from hollowmere.agents.persona import Persona, RANK, starting_affinity

if TYPE_CHECKING:
    from hollowmere.agents.agent import Agent

MALE_NAMES = (
    "Ned", "Fenwick", "Garrick", "Aldous", "Wymer", "Alden", "Bram", "Cuthbert", "Dunstan", "Edric",
    "Godwin", "Hob", "Osric", "Percival", "Rupert", "Tobias", "Wilfred", "Corwin", "Hadrian", "Jory",
)
FEMALE_NAMES = (
    "Maude", "Thea", "Rosamund", "Enid", "Lyra", "Alys", "Bryony", "Cecily", "Dilys", "Edith",
    "Gwenna", "Hilda", "Ismay", "Linnet", "Nell", "Odette", "Philippa", "Sybil", "Verity", "Wynne",
)
GOBLIN_TRAITS = ("horny", "lustful", "driven to breed")  # every goblin has these; the rest comes from the parents
GOBLIN_NAMES = (
    "Grik", "Snurt", "Tam", "Pell", "Mossa", "Bligg", "Dribb", "Fizzle", "Gnash", "Hobb",
    "Krumb", "Lurk", "Nitt", "Pogg", "Quill", "Skrit", "Tarn", "Wick", "Yarg", "Zib",
)


def eligible_for_love(self_sex: str, other_sex: str, bond: dict | None, kin: bool = False) -> bool:
    """Whether this agent's own point of view makes "making love" with `other` a reachable outcome
    of a conversation - opposite sex, and either already intimate or already fond or better."""
    if bond is None or self_sex == other_sex or kin:
        return False
    if bond.get("intimate"):
        return True
    return RANK[bond["stance"]] >= RANK["fond"]


def generate_child_persona(mother: Persona, father: Persona, born_on: date, rng: random.Random) -> Persona:
    """A brand-new Persona for a newborn: the father's species, the mother's home, both parents
    named as family. Deterministic given `rng`'s state - callers pass the engine's one shared,
    ever-advancing rng, so two births never actually draw from the same state."""
    sex = rng.choice(["male", "female"])
    species = "goblin" if "goblin" in (mother.species, father.species) else father.species  # a goblin's child is one
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
        parents=[mother.id, father.id],
        stances={mother.id: "fond", father.id: "fond"},
        species=species,
    )


GROWN_ONLY = ("customer", "money", "desire", "preach")  # a child shows "braver than she lets on", not "warm to paying customers"


def growth_pool(mother: Persona, father: Persona) -> tuple[str, ...]:
    """The temperament words a child can inherit: those of both parents, without the three every goblin has."""
    words: list[str] = []
    for p in (mother, father):
        for w in p.traits.split(","):
            w = w.strip()
            if w and w not in words and w not in GOBLIN_TRAITS and not any(g in w for g in GROWN_ONLY):
                words.append(w)
    return tuple(words)


def born_on(persona: Persona) -> date | None:
    """The day a generated child was born, as its background says."""
    m = re.search(r"born on (\d{4}-\d{2}-\d{2})", persona.background)
    return date.fromisoformat(m.group(1)) if m else None


async def write_child_persona(ctx, base: Persona, mother: Persona, father: Persona, born_on: date,
                              keep_name: str = "") -> Persona:
    """The newborn as the model writes it from its parents: name, looks, background. What the code fixed on `base`
    (sex, species, parents, age, home, a temperament too young to show) stays; a failed call, or a name that is
    already taken, leaves the generated persona as it was."""
    from hollowmere.agents import prompts
    from hollowmere.agents.schemas import ChildDraft
    from hollowmere.llm.client import LLMError

    firsts = sorted({n.split()[0] for n in ctx.names.values() if n.split()})
    taken = {n.lower() for n in firsts} | {n.lower() for n in ctx.names.values()}
    refused, out = "", None
    for _ in range(2):  # once more if the name it chose is already someone's
        try:
            out = await ctx.llm.structured(
                prompts.child_draft_system_prompt(),
                prompts.child_draft(mother, father, base.species, base.sex, born_on, firsts, refused, keep_name),
                ChildDraft, label="child_persona", temperature=0.8)
        except LLMError:
            return base
        name = keep_name or out.name.strip()
        if keep_name or (name and name.split()[0].lower() not in taken and name.lower() not in taken):
            break
        refused = name
    else:
        return base
    return base.model_copy(update={
        "name": name, "appearance": out.appearance.strip(),
        "background": f"{out.background.strip()} {name} was born on {born_on.isoformat()}."})


def birth_memory(is_mother: bool, child: Persona, other: Persona) -> str:
    word = "son" if child.sex == "male" else "daughter"
    if is_mother:
        return f"I gave birth to {child.name}, {other.name}'s and my {word}."
    return f"{child.name} was born, {other.name}'s and my {word}."


def recognize_child(mother: "Agent", father: "Agent", child: "Agent") -> None:
    """Make both parents' own bonds mechanically aware of the child - Persona.relationships isn't
    persisted across restarts, only Agent.bonds is, so this has to be a direct write to that dict."""
    for parent in (mother, father):
        parent.bonds[child.id] = {"stance": "fond", "affinity": starting_affinity("fond"),
                                  "note": "", "talks": 0, "last": None,
                                  "kin": f"your newborn {'son' if child.persona.sex == 'male' else 'daughter'}"}
