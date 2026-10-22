"""An offline stand-in for the LLM.

Produces schema-valid, roughly plausible answers without a model, so the whole
simulation and UI can run (and be tested) before a server is available.
"""

from __future__ import annotations

import asyncio
import random
import re
from typing import TypeVar

from pydantic import BaseModel

from hollowmere.llm.client import LLMStats

T = TypeVar("T", bound=BaseModel)

_LINES = [
    "Another cart of bodies went past the gate this morning.",
    "Keep your voice down. The watch has ears in every wall.",
    "My mother says the well water tastes of iron now.",
    "Pray if you like. I'd sooner bar the door.",
    "Have you heard? The Lord's collector comes in two days.",
    "The candles in the chapel burned blue last night, I swear it.",
    "I've no coin to spare, and no patience either.",
    "Mind yourself after curfew. Roderic is in a hanging mood.",
]


def _enum(schema: dict, *path: str) -> list[str]:
    node = schema
    for key in path:
        node = node[key]
    if "$ref" in node:
        node = schema["$defs"][node["$ref"].split("/")[-1]]
    return node.get("enum") or [node.get("const")]


class FakeLLM:
    def __init__(self, seed: int = 7, latency: float = 0.05):
        self.stats = LLMStats()
        self.rng = random.Random(seed)
        self.latency = latency

    async def aclose(self) -> None:
        pass

    async def structured(self, system: str, user: str, schema: type[T], *, label: str,
                         temperature: float | None = None) -> T:
        await asyncio.sleep(self.latency)
        self.stats.calls += 1
        self.stats.by_label[label] = self.stats.by_label.get(label, 0) + 1
        js = schema.model_json_schema()
        data = getattr(self, f"_{schema.__name__.lower()}")(js, user)
        return schema.model_validate(data)

    # One method per schema name ------------------------------------------------

    def _dayplan(self, js: dict, user: str) -> dict:
        places = _enum(js, "$defs", "PlanBlock", "properties", "place")
        bed = re.search(r"Your bed is (\S+)\.", user).group(1)
        work = re.search(r"Your work is at (\S+)\.", user).group(1)
        public = [p for p in places if "/" not in p and p not in (work, bed.split("/")[0])]
        if "Humans who have sworn to protect you" in user:  # only the goblin's identity says this (Wenna knows of goblins)
            return {
                "morning_thought": "Hungry again. The town smells of bread and fear.",
                "wake_up": "12:00",
                "blocks": [
                    {"start": "12:00", "end": "16:00", "activity": "tends the shrine and the fire", "place": work},
                    {"start": "16:00", "end": "19:00", "activity": "skulks at the edge of the woods", "place": "witchwood"},
                    {"start": "19:00", "end": "23:00", "activity": "creeps about the graves", "place": "graveyard"},
                    {"start": "23:00", "end": "02:00", "activity": "steals scraps behind the bakery", "place": "square"},
                    {"start": "02:00", "end": "03:30", "activity": "goes to sleep", "place": bed},
                ],
            }
        wake = self.rng.choice(["05:30", "06:00", "06:30"])
        out = self.rng.choice(public)
        return {
            "morning_thought": "Another grey day. I must keep my head down and my belly full.",
            "wake_up": wake,
            "blocks": [
                {"start": wake, "end": "07:00", "activity": "rises, washes and eats a heel of bread", "place": bed},
                {"start": "07:00", "end": "12:00", "activity": "works", "place": work},
                {"start": "12:00", "end": "13:00", "activity": "eats a bowl of pottage", "place": "tavern"},
                {"start": "13:00", "end": "16:00", "activity": "works", "place": work},
                {"start": "16:00", "end": "18:00", "activity": "runs an errand", "place": out},
                {"start": "18:00", "end": "21:00", "activity": "drinks and listens to gossip", "place": "tavern"},
                {"start": "21:00", "end": "05:00", "activity": "goes to sleep", "place": bed},
            ],
        }

    def _decomposition(self, js: dict, user: str) -> dict:
        places = _enum(js, "$defs", "Step", "properties", "place")
        minutes = int(re.search(r"\((\d+) minutes\)", user).group(1))
        n = max(1, min(4, minutes // 30))
        emojis = ["🔨", "🍺", "🕯️", "🌿", "⚰️", "🗡️", "🥖", "👀"]
        out: dict = {"steps": [{"activity": self.rng.choice(["tends to things", "looks about warily", "keeps busy",
                                                              "mutters a prayer"]),
                                "minutes": minutes // n, "place": self.rng.choice(places),
                                "emoji": self.rng.choice(emojis)} for _ in range(n)]}
        if "seek" in js.get("properties", {}):
            out["seek"] = self.rng.choice(_enum(js, "properties", "seek"))
        return out

    def _goblinchoice(self, js: dict, user: str) -> dict:
        people = _enum(js, "properties", "whom")
        sight = user.split("IN SIGHT", 1)[1]
        friendly = any(w in sight for w in ("(fond", "(trust", "sworn to protect"))
        action = "approach" if friendly and len(people) == 2 else self.rng.choice(["hide", "watch", "flee"])
        return {"thought": "Careful now. Who is it?", "action": action, "whom": people[0],
                "reason": "to beg a crust and a kind word" if action == "approach" else ""}

    def _reaction(self, js: dict, user: str) -> dict:
        talk = self.rng.random() < 0.35
        return {"thought": "I wonder what they are about at this hour.", "importance": self.rng.randint(2, 5),
                "talk": talk, "reason": "to hear the latest news" if talk else ""}

    def _utterance(self, js: dict, user: str) -> dict:
        turns = user.count('": "') + user.count(': "')
        return {"says": self.rng.choice(_LINES), "end": turns >= 4 and self.rng.random() < 0.5}

    def _chatdigest(self, js: dict, user: str) -> dict:
        out = {"summary": "We spoke of the plague and the collector's coming. Everyone is afraid.",
               "importance": 4, "commitments": [],
               "feeling": self.rng.choice(["wary", "neutral", "fond"]), "view_of_them": "As frightened as the rest."}
        if "toward_goblin" in js.get("properties", {}):
            out["toward_goblin"] = self.rng.choice(["drive off", "undecided", "tolerate", "feed", "protect"])
        return out

    def _focalquestions(self, js: dict, user: str) -> dict:
        return {"questions": ["Who can I trust in this town?", "Will the plague reach my door?"]}

    def _insights(self, js: dict, user: str) -> dict:
        ids = [int(i) for i in re.findall(r"^\[(\d+)\]", user, re.MULTILINE)][:3]
        return {"insights": [{"text": "Fear is making everyone in Hollowmere a stranger.",
                              "evidence": ids, "importance": 5}]}

    def _dayreview(self, js: dict, user: str) -> dict:
        return {"summary": "A long, cold day. Nothing changed, and that frightens me.", "importance": 3,
                "intentions": ["keep my head down"]}
