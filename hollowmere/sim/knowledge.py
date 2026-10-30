"""Who knows what, and how they came to know it: read off the memories (their text and where they came from).

No model is asked. The engine knows what is true (a pregnancy, a birth, two lovers, a flight, an arrest, a goblin
in the hills); for each such fact every memory is read that names the people it is about and says the thing, and the
earliest one tells how that person came to know it (saw it, lived it, was told by someone, had word of it, or
worked it out). It is a view for the player, and changes nothing in the town."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

from hollowmere.agents.agent import mentions
from hollowmere.memory.stream import Memory


@dataclass
class Fact:
    kind: str
    text: str
    names: list[str]  # ids a memory must name to be about it (all of them)
    pattern: re.Pattern  # and one of these words
    involved: set[str] = field(default_factory=set)  # those who lived it, or the thing is about
    about: list[str] = field(default_factory=list)  # whom it is said of: where a whisper of it belongs
    after: datetime | None = None  # nothing before this moment can be knowledge of it


def _words(*words: str) -> re.Pattern:
    """Any of the words, at the start of a word ("son" does not match "person")."""
    return re.compile(r"\b(?:" + "|".join(words) + ")", re.I)


def _when(t) -> datetime | None:
    return datetime.fromisoformat(t) if isinstance(t, str) else t


def facts_of(engine) -> list[Fact]:
    """What is true in the town, as far as it is worth asking who knows."""
    ag, out = engine.agents, []
    fathers = set()
    for p in engine.pregnancies:
        mother, father = ag.get(p["mother_id"]), ag.get(p["father_id"])
        if mother is None or father is None:
            continue
        both = {mother.id, father.id}
        began = datetime.fromisoformat(p["start_day"])
        out.append(Fact("pregnancy", f"{mother.name} is with child" if not p["resolved"]
                        else f"{mother.name} was with child (the child is born)", [mother.id],
                        _words("with child", "pregnan", "belly", "expecting", "carrying", "swell"), both, [mother.id],
                        began))
        if (mother.id, father.id) in fathers:
            continue  # one father for all the pair's children
        fathers.add((mother.id, father.id))
        out.append(Fact("father", f"{father.name} is the father of {mother.name}'s child", [mother.id, father.id],
                        _words("father", "child", "baby", "pregnan", "carrying", "bastard", "heir", "born"), both,
                        [father.id, mother.id], began))
    births = {i: ev["time"] for ev in engine.store.events_of("birth", datetime.min) for i in ev["agents"]}
    for child in ag.values():
        if len(child.persona.parents) == 2 and all(i in ag for i in child.persona.parents):
            mother = ag[child.persona.parents[0]]
            out.append(Fact("birth", f"{mother.name} gave birth to {child.name}", [child.id],
                            _words(r"born", "birth", "newborn", "gave birth", "baby", "son", "daughter"),
                            {*child.persona.parents, child.id}, [mother.id], _when(births.get(child.id))))
    seen = set()
    for a in ag.values():
        for bid, bond in a.bonds.items():
            pair = tuple(sorted((a.id, bid)))
            if bond.get("intimate") and bid in ag and ag[bid].bonds.get(a.id, {}).get("intimate") and pair not in seen:
                seen.add(pair)
                out.append(Fact("lovers", f"{a.name} and {ag[bid].name} are lovers", list(pair),
                                _words("lover", "made love", "in love", "intimate", "passion", "kiss", "bed"),
                                set(pair), list(pair)))
    for kind, label, words in (("returned", "tried to leave Hollowmere and was turned back",
                                _words("leave", "left", "fled", "slip", "turned back", "run away", "escape")),
                               ("arrest", "was locked in the watchhouse cell",
                                _words("locked", "cell", "arrest", "seiz", "watchhouse")),
                               ("move", "moved in together", _words("moved in", "moves in", "live with", "den"))):
        for ev in engine.store.events_of(kind, datetime.min):
            ids = [i for i in ev["agents"] if i in ag and not ag[i].persona.dependent]
            if ids:
                out.append(Fact(kind, ev["text"], ids[:2], words, set(ids), ids[:2], _when(ev["time"])))
    goblins = [a for a in ag.values() if a.persona.is_goblin]
    for g in goblins:
        out.append(Fact("goblin", f"{g.name}, a goblin, lives near Hollowmere", [g.id], _words(".*"), {g.id}, [g.id]))
    return out


def _how(m: Memory, names: dict[str, str]) -> str:
    s = m.source
    if s in ("saw", "lived"):
        return "saw"
    if s.startswith("told:"):
        who = s.split(":", 1)[1]
        return f"told by {names.get(who, who).split()[0]}"
    if s == "word":
        return "heard"
    if s in ("own", "divine"):
        return "thought"
    return {"observation": "saw", "chat": "told", "thought": "thought", "reflection": "thought"}.get(m.kind, "memory")


def build(engine) -> list[dict]:
    return scan(engine, facts_of(engine))


def scan(engine, facts: list[Fact]) -> list[dict]:
    """Read every memory for each fact. Touches only the agents' memories, so it can run off the main thread."""
    names = {i: a.name for i, a in engine.agents.items()}
    grown = [a for a in engine.agents.values() if not a.persona.dependent]
    named_in: dict[int, list[str]] = {}  # a memory names the same people for every fact: look once
    out = []
    for f in facts:
        knowers, since = [], None
        for a in grown:
            hit = None
            for m in list(a.memory.items):
                if m.kind == "plan" or (f.after and m.created < f.after):
                    continue
                if m.id not in named_in:
                    named_in[m.id] = mentions(m.text, names)
                named = [n for n in named_in[m.id] if n != a.id]
                if a.id in f.names:
                    named.append(a.id)  # a memory of one's own is about oneself
                if all(n in named for n in f.names) and f.pattern.search(m.text):
                    hit = m
                    break
            involved = a.id in f.involved
            if hit is None and not involved:
                continue
            knowers.append({"agent": a.id, "name": a.name, "color": a.persona.color, "involved": involved,
                            "how": "involved" if (involved and hit is None) else _how(hit, names),
                            "since": hit.created.isoformat() if hit else None,
                            "snippet": hit.text[:140] if hit else ""})
            if hit and (since is None or hit.created.isoformat() < since):
                since = hit.created.isoformat()
        out.append({"kind": f.kind, "text": f.text, "about": f.about, "since": since or "", "count": len(knowers),
                    "knowers": sorted(knowers, key=lambda k: (not k["involved"], k["since"] or "~")),
                    "unaware": [a.name for a in grown if a.id not in {k["agent"] for k in knowers}]})
    return sorted(out, key=lambda f: f["since"] or "", reverse=True)
