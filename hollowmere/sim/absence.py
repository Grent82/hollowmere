"""Leaving Hollowmere and turning back: the engine's side of Agent.absence.

The model decides that someone means to go (agreed_departure, one reading of the talk); the engine only carries it
out. A flight from the town always fails, after a day or three, for a reason drawn from data/world.json, so
nobody is lost for good. Those who know the missing hear that they are gone, and later why they came back;
the rest of the town learns it as it learns everything else, in talk.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, time, timedelta

from hollowmere.agents import prompts
from hollowmere.agents.agent import CELL, Agent, day_key, due_at
from hollowmere.agents.persona import DATA, RANK
from hollowmere.agents.schemas import journey_schema
from hollowmere.agents import romance
from hollowmere.llm.client import LLMError

TURNED_BACK = json.loads((DATA / "world.json").read_text(encoding="utf-8"))["turned_back"]
DEPARTURE = time(22, 0)
RETURN = time(17, 0)
NEWS_AT = time(6, 0)
RELEASE = time(8, 0)
MOVE_IN = time(20, 0)
WHY_LINES = 8  # lines of the parting talk the storyteller is shown
MISSED_AFTER_DAYS = 3  # the town says out loud that someone has not been seen, then again every week
MISSED_EVERY_DAYS = 7
VISIT_GAP_DAYS = 2  # a short visit home lasts this long before they are drawn out again
TRIED = "I tried to leave Hollowmere"


log = logging.getLogger(__name__)


def _names(agents: list[Agent]) -> str:
    return " and ".join(a.first for a in agents)


class Absences:
    """Mixed into Engine. Needs: agents, world, cfg, rng, now, _queue, _event()."""

    def schedule_departure(self, group: list[Agent], now: datetime, why: str = "") -> None:
        """One person or two settled on leaving: they set out at the next 22:00. A child never goes alone: it
        travels with its mother (see `_dependents`)."""
        group = [x for x in group if not x.persona.dependent]
        if not group:
            return
        ids = [x.id for x in group]
        if any(x.absence for x in group) or any(q["do"] == "depart" and set(ids) & set(q["ids"]) for q in self._queue):
            return
        at = datetime.combine(day_key(now), DEPARTURE)
        if at <= now:
            at += timedelta(days=1)
        self._queue.append({"do": "depart", "ids": ids, "at": at.isoformat(), "why": why})

    def schedule_move_in(self, mover: Agent, host: Agent, now: datetime) -> None:
        """Settled on living under the host's roof: the mover carries their things over at the next 20:00."""
        # A human may move into the den, but a goblin cannot go indoors, so he cannot live under a roof.
        if (mover.persona.is_goblin or mover.persona.dependent or host.persona.dependent
                or mover.absence or host.absence or mover.persona.home == host.persona.home
                or any(q["do"] == "move_in" and mover.id in q["ids"] for q in self._queue)):
            return
        at = datetime.combine(day_key(now), MOVE_IN)
        if at <= now:
            at += timedelta(days=1)
        self._queue.append({"do": "move_in", "ids": [mover.id], "to": host.id, "at": at.isoformat()})

    def _dependents(self, ids: list[str]) -> list[Agent]:
        """Infants stay with their mother, wherever she goes."""
        return [x for x in self.agents.values() if x.persona.dependent and x.persona.parents[:1] and x.persona.parents[0] in ids]

    def circle(self, group: list[Agent]) -> list[Agent]:
        """Who would notice they were gone: family, whoever shares their roof, and anyone fond of one of them
        or used to talking with them. (Everyone in Hollowmere has a bond with everyone, so bonds alone say nothing.)"""
        ids = {x.id for x in group}
        homes = {x.persona.home for x in group}

        def close(o: Agent) -> bool:
            return o.persona.home in homes or any(
                self.ctx.kin.related(o.id, i) or o.bonds.get(i, {}).get("talks", 0) >= 3
                or RANK[o.bonds.get(i, {}).get("stance", "neutral")] >= RANK["fond"] for i in ids)
        return [o for o in self.agents.values()
                if o.id not in ids and not o.persona.dependent and not o.is_away and close(o)]

    async def _absences(self, now: datetime) -> None:
        """Run what has come due: departures, the morning's news, and everyone whose absence has run out."""
        due, self._queue = [q for q in self._queue if datetime.fromisoformat(q["at"]) <= now], \
            [q for q in self._queue if datetime.fromisoformat(q["at"]) > now]
        for q in due:
            try:
                await self._run_entry(q, now)
            except Exception:  # noqa: BLE001 - one bad entry must not cost the others their news
                log.exception("absence queue entry failed: %s", q)
        done: set[str] = set()
        for a in list(self.agents.values()):
            if a.absence is None or a.id in done or datetime.fromisoformat(a.absence["until"]) > now:
                continue
            group = [self.agents[i] for i in a.absence.get("with", [a.id]) if i in self.agents]  # old saves lack it
            done.update(x.id for x in group)
            if a.is_away:
                if a.absence.get("stays") and await self._tell_journey([x for x in group if not x.persona.dependent],
                                                                       now, later=True):
                    continue  # still out there; the model has said when they are looked at next
                await self.return_home(group, now)
            else:
                await self.release(group, now)

    async def _run_entry(self, q: dict, now: datetime) -> None:
        group = [self.agents[i] for i in q["ids"] if i in self.agents]
        if q["do"] == "depart":
            if group and not any(x.absence for x in group):
                self.depart(group, now)
                await self._tell_journey(group, now, q.get("why", ""))
        elif q["do"] == "move_in":
            host = self.agents.get(q["to"])
            if group and host is not None and not group[0].absence and not host.absence:
                await self.move_in(group[0], host, now)
        elif q["do"] == "arrested":
            who, by = group[0], self.agents[q["by"]]
            await who._remember("observation", f"{by.name} had me seized and locked in the cell of the "
                                               "watchhouse.", now, 9, source="lived", about=[by.id])
            await by._remember("observation", f"I had {who.name} locked in the watchhouse cell.", now, 7,
                               source="lived", about=[who.id])
            for o in self.circle([who]):
                if o is not by:
                    await o._remember("observation", f"Word is that {who.name} was locked in the watchhouse "
                                                     f"cell by {by.name}.", now, 7, source="word",
                                      about=[who.id, by.id])
        elif q["do"] == "missed":
            gone = [x for x in group if x.is_away and not x.persona.dependent]
            if gone:
                days = (day_key(now) - day_key(datetime.fromisoformat(gone[0].absence["since"]))).days
                self._event("missed", [x.id for x in gone], f"{_names(gone)} {'has' if len(gone) == 1 else 'have'} "
                                                           f"not been seen in Hollowmere for {days} days.")
                for o in self.circle(gone):
                    await o._remember("observation", f"{_names(gone)} {'has' if len(gone) == 1 else 'have'} been gone "
                                      f"from Hollowmere for {days} days; no one has heard from them.", now, 6,
                                      source="word", about=[x.id for x in gone])
                self._queue_missed([x.id for x in gone], day_key(now) + timedelta(days=MISSED_EVERY_DAYS))
        elif q["do"] == "rumor":
            gone = [x for x in group if x.is_away and not x.persona.dependent]
            for o in self.circle(gone) if gone else []:
                await o._remember("observation", f"A rumor goes about the town: {q['text']}", now, 5, source="word",
                                  about=[x.id for x in gone])
            if gone:
                self._event("missed", [x.id for x in gone], f"A rumor goes about the town: {q['text']}")
        elif q["do"] == "vanish":
            for o in self.circle(group):
                await o._remember("observation", f"{_names(group)} {'has' if len(group) == 1 else 'have'} not "
                                  f"been seen since {q['since']}; no one knows where.", now, 6, source="word",
                                  about=[x.id for x in group])

    async def move_in(self, mover: Agent, host: Agent, now: datetime) -> None:
        """The mover now lives and sleeps where the host does, babies with them; both and those close are told."""
        for x in [mover, *self._dependents([mover.id])]:
            x.set_home(host.persona.home, host.persona.home_object, self.world)
        await mover._remember("observation", f"I moved into {host.name}'s home to live there.", now, 8,
                              source="lived", about=[host.id])
        await host._remember("observation", f"{mover.name} moved into my home to live with me.", now, 8,
                             source="lived", about=[mover.id])
        for o in self.circle([mover, host]):
            await o._remember("observation", f"Word is that {mover.name} moved in with {host.name}.", now, 6,
                              source="word", about=[mover.id, host.id])
        self._event("move", [mover.id, host.id], f"{mover.first} moves in with {host.first}.")

    def depart(self, group: list[Agent], now: datetime) -> None:
        """Out of town, with their babies. They are not seen, not planned for, and not sought."""
        everyone = group + self._dependents([x.id for x in group])
        days = self.rng.randint(self.cfg.absence_min_days, self.cfg.absence_max_days)
        until = datetime.combine(day_key(now) + timedelta(days=days), RETURN)
        reason = self.rng.choice(TURNED_BACK)
        for x in everyone:
            convo = self.conversations.get(x.convo_id) if x.convo_id else None
            if convo is not None:
                self._end_conversation(convo, "goodbye")
            x.absence = {"kind": "away", "since": now.isoformat(), "until": until.isoformat(),
                         "reason": reason, "with": [y.id for y in everyone]}
            x.mode, x.path, x.task, x.seek, x.hiding_until, x.bubble = "away", [], None, None, None, None
            x.commitments = [c for c in x.commitments if due_at(c) > until]
            self._seeking.pop(x.id, None)
        tomorrow = datetime.combine(day_key(now) + timedelta(days=1), NEWS_AT)
        self._queue.append({"do": "vanish", "ids": [x.id for x in group], "at": tomorrow.isoformat(),
                            "since": f"{now:%A} evening"})
        self._event("depart", [x.id for x in group], f"{_names(group)} slip out of Hollowmere.")

    async def recall(self, agent_id: str) -> None:
        """The player calls someone who is away back home, with the little ones they carry: it wins over whatever
        the storyteller had in mind, and the model is not asked. They are home by the next evening."""
        a = self.agents.get(agent_id)
        if a is None or not a.is_away or a.persona.dependent:
            raise ValueError("recall someone who is away from Hollowmere")
        kids = [c for c in self._dependents([a.id]) if c.is_away]
        members = [a, *kids]
        ids = {x.id for x in members}
        for x in self.agents.values():  # those left on the road are no longer with them
            if x.absence and x.id not in ids and ids & set(x.absence.get("with", [])):
                x.absence = {**x.absence, "with": [i for i in x.absence["with"] if i not in ids]}
        at = datetime.combine(day_key(self.now), RETURN)
        if at <= self.now:
            at += timedelta(days=1)
        for x in members:
            x.absence = {**x.absence, "until": at.isoformat(), "with": [y.id for y in members], "stays": False,
                         "visit": False, "recalled": True, "reason": "called home",
                         "memory": "Something drew me home, and I could not stay away." if x is a else "", "state": ""}
        self._event("recall", [a.id], f"You call {_names([a])} home.")

    def _born_away(self, child: Agent, mother: Agent, now: datetime) -> None:
        """The child comes into the world where its mother is, with her and whoever is with her, unseen."""
        ids = [*mother.absence["with"], child.id]
        for x in (self.agents[i] for i in mother.absence["with"] if i in self.agents):
            x.absence = {**x.absence, "with": ids}
        child.absence = {**mother.absence, "with": ids, "memory": "", "state": "", "visit": False, "born_away": True}
        child.x, child.y, child.mode = mother.x, mother.y, "away"
        self._queue.append({"do": "rumor", "ids": [mother.id], "at": datetime.combine(
            day_key(now) + timedelta(days=1), NEWS_AT).isoformat(),
            "text": f"{mother.name} is said to have been brought to bed of a child out on the road."})

    async def _announce_birth(self, child: Agent, now: datetime) -> None:
        """A child born away is home: her family, the father and whoever knew her belly learn of it at last."""
        mother, father = (self.agents.get(i) for i in child.persona.parents)
        if mother is None or father is None:
            return
        await self._tell_mothers_family(child, mother, now)
        await self._tell_witnesses(child, mother, now)
        if not any(child.name in m.text for m in father.memory.items):
            await father._remember("observation", romance.birth_memory(False, child.persona, mother.persona), now, 9,
                                   source="lived", about=[child.id, mother.id])
        self._event("birth", [mother.id, father.id, child.id],
                    f"{mother.first} and {father.first} welcome a child, {child.first}.")

    def _queue_missed(self, ids: list[str], day) -> None:
        """Someone still gone is said to be missed on this day, at the morning news; once for any one of them."""
        if not any(q["do"] == "missed" and set(ids) & set(q["ids"]) for q in self._queue):
            self._queue.append({"do": "missed", "ids": ids, "at": datetime.combine(day, NEWS_AT).isoformat()})

    def _expecting(self) -> dict[str, str]:
        return {p["mother_id"]: p["due_day"] for p in self.pregnancies if not p["resolved"]}

    async def _tell_journey(self, group: list[Agent], now: datetime, why: str = "", later: bool = False) -> bool:
        """The model tells what befalls those who set out, or what has become of those who stayed away: the days
        until they are home or looked at again, and for each of them how it goes on. The code takes the answer
        only within its limits; whoever the model leaves out, or a failed call, keeps what they had. True if told."""
        away = [x for x in group if x.is_away and not x.persona.dependent]
        if not away:
            return False
        kids = [c for c in self._dependents([x.id for x in away]) if c.is_away]
        since = min(datetime.fromisoformat(x.absence["since"]) for x in away)
        tried = {x.id: [m.text for m in x.memory.items if m.text.startswith(TRIED)] for x in away}
        try:
            out = await self.ctx.llm.structured(
                prompts.journey_system_prompt(),
                prompts.journey(away, kids, now, why, self._expecting(), tried,
                                story=away[0].absence["reason"] if later else "",
                                days_away=max(1, (day_key(now) - day_key(since)).days)),
                journey_schema(tuple(x.first for x in away), self.cfg.away_max_days), label="journey", temperature=0.8)
        except LLMError as e:
            log.warning("%s: the journey could not be told (%s)", _names(away), e)
            return False
        told = {}
        for f in out.fates:
            told.setdefault(f.who, f)
        clusters: dict[tuple, list[Agent]] = {}
        for x in away:
            f = told.get(x.first)
            if f is not None:
                clusters.setdefault((f.fate, f.after_days), []).append(x)
        for (how, days), members in clusters.items():
            until = datetime.combine(day_key(now) + timedelta(days=days), RETURN)
            everyone = members + [c for c in kids if c.persona.parents[:1] and c.persona.parents[0] in {m.id for m in members}]
            for x in everyone:
                f = told.get(x.first)
                x.absence = {**x.absence, "until": until.isoformat(), "reason": out.story.strip() or x.absence["reason"],
                             "told": bool(out.story.strip()),
                             "with": [y.id for y in everyone], "stays": how == "stays_away",
                             "visit": how == "returns_briefly",
                             "memory": f.memory.strip() if f else "", "state": f.state.strip() if f else "",
                             "kids": out.children.strip()}
                x.commitments = [c for c in x.commitments if due_at(c) > until]
        if not later:
            self._queue_missed([x.id for x in away], day_key(since) + timedelta(days=MISSED_AFTER_DAYS))
        if out.rumor.strip():
            self._queue.append({"do": "rumor", "ids": [x.id for x in away], "text": out.rumor.strip(),
                                "at": datetime.combine(day_key(now) + timedelta(days=1), NEWS_AT).isoformat()})
        return True

    async def return_home(self, everyone: list[Agent], now: datetime) -> None:
        """Back home, with a memory of the road, and the people who know them are told."""
        group = [x for x in everyone if not x.persona.dependent]
        reason = everyone[0].absence["reason"]
        recalled = bool(everyone[0].absence.get("recalled"))
        told = bool(everyone[0].absence.get("told"))  # the reason is the storyteller's sentence, not a bare cause
        carried = {x.id: (x.absence.get("memory", ""), x.absence.get("state", "")) for x in group}
        visiting = [x.id for x in group if x.absence.get("visit")]
        born = [x for x in everyone if x.absence.get("born_away")]
        for x in everyone:
            x.absence = None
            x.x, x.y = self.world.target_tile(x.persona.bed) or (x.x, x.y)
            x.mode, x.plan, x.plan_date, x.tasks, x.tasks_block, x.task = "acting", [], None, [], -1, None
        for x in group:
            others = [y for y in group if y is not x]
            company = f" with {_names(others)}" if others else ""
            memory, state = carried[x.id]
            plain = (f"{TRIED}{company}, and came home: {reason}" if told
                     else f"{TRIED}{company}, but {reason} stopped us, and we turned back.")
            await x._remember("observation", memory or plain, now, 8, source="lived", about=[y.id for y in others])
            if state:
                x.set_states([state] + [s["text"] for s in x.states], now)
        are = "is" if len(group) == 1 else "are"
        if recalled:
            news = f"Word is that {_names(group)} {are} back in Hollowmere."
            line = f"{_names(group)} {are} called home to Hollowmere."
        elif told:
            news = f"Word is that {_names(group)} tried to leave Hollowmere and came home: {reason}"
            line = f"{_names(group)} come back to Hollowmere. {reason}"
        else:
            news = (f"Word is that {_names(group)} tried to leave Hollowmere and "
                    f"{'was' if len(group) == 1 else 'were'} turned back by {reason}.")
            line = f"{_names(group)} come back to Hollowmere, turned back by {reason}."
        for o in self.circle(group):
            await o._remember("observation", news, now, 7, source="word", about=[x.id for x in group])
        self._event("recall" if recalled else "returned", [x.id for x in group], line)
        for child in born:
            await self._announce_birth(child, now)
        if visiting:
            at = datetime.combine(day_key(now) + timedelta(days=VISIT_GAP_DAYS), DEPARTURE)
            self._queue.append({"do": "depart", "ids": visiting, "at": at.isoformat(),
                                "why": "home for a short while, and drawn out again"})

    def arrest(self, by: Agent, who: Agent, now: datetime) -> bool:
        """`by` has had `who` locked in the watchhouse cell for a day or three. Anyone may do it to anyone."""
        if (by is who or any(x.persona.dependent or x.persona.is_goblin or x.absence for x in (by, who))
                or sum(1 for x in self.agents.values() if x.is_jailed and not x.persona.dependent) >= self.cfg.max_jailed):
            return False
        everyone = [who] + self._dependents([who.id])
        days = self.rng.randint(self.cfg.detention_min_days, self.cfg.detention_max_days)
        until = datetime.combine((now + timedelta(days=days)).date(), RELEASE)
        if until < now + timedelta(days=days):  # a full day or more, whatever the hour
            until += timedelta(days=1)
        for x in everyone:
            convo = self.conversations.get(x.convo_id) if x.convo_id else None
            if convo is not None:
                self._end_conversation(convo, "goodbye")
            x.absence = {"kind": "jailed", "since": now.isoformat(), "until": until.isoformat(),
                         "reason": f"locked up by {by.name}", "with": [y.id for y in everyone]}
            x.x, x.y = self.world.target_tile(CELL)
            x.mode, x.path, x.task, x.seek, x.hiding_until, x.bubble, x.plan_date = \
                "acting", [], None, None, None, None, None
            self._seeking.pop(x.id, None)
        by.learn_area("guard_cell", self.world)
        self._queue.append({"do": "arrested", "ids": [who.id], "by": by.id, "at": now.isoformat()})  # told next tick
        self._event("arrest", [by.id, who.id], f"{by.first} has {who.first} locked in the watchhouse cell.")
        return True

    async def release(self, everyone: list[Agent], now: datetime) -> None:
        """Let out: home, and a fresh day to plan."""
        since = datetime.fromisoformat(everyone[0].absence["since"])
        days = max(1, (day_key(now) - day_key(since)).days)
        for x in everyone:
            x.absence = None
            x.x, x.y = self.world.target_tile(x.persona.bed) or (x.x, x.y)
            x.mode, x.plan, x.plan_date, x.tasks, x.tasks_block, x.task = "acting", [], None, [], -1, None
        for x in everyone:
            if not x.persona.dependent:
                await x._remember("observation", f"I was let out of the cell after {days} day{'s' * (days > 1)}.", now, 7)
        self._event("released", [x.id for x in everyone if not x.persona.dependent],
                    f"{everyone[0].first} is let out of the watchhouse cell.")
