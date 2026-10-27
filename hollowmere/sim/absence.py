"""Leaving Hollowmere and turning back: the engine's side of Agent.absence.

The model decides that two people mean to go (Digest.leaving, from both of them); the engine only carries it
out. A flight from the town always fails, after a day or three, for a reason drawn from data/world.json, so
nobody is lost for good. Those who know the missing hear that they are gone, and later why they came back;
the rest of the town learns it as it learns everything else, in talk.
"""

from __future__ import annotations

import json
from datetime import datetime, time, timedelta

from hollowmere.agents.agent import Agent, day_key, due_at
from hollowmere.agents.persona import DATA

TURNED_BACK = json.loads((DATA / "world.json").read_text(encoding="utf-8"))["turned_back"]
DEPARTURE = time(22, 0)
RETURN = time(17, 0)
NEWS_AT = time(6, 0)


def _names(agents: list[Agent]) -> str:
    return " and ".join(a.first for a in agents)


class Absences:
    """Mixed into Engine. Needs: agents, world, cfg, rng, now, _queue, _event()."""

    def schedule_departure(self, a: Agent, b: Agent, now: datetime) -> None:
        """Two people agreed to leave: they set out at the next 22:00."""
        ids = [a.id, b.id]
        if any(x.absence for x in (a, b)) or any(q["do"] == "depart" and set(ids) & set(q["ids"]) for q in self._queue):
            return
        at = datetime.combine(day_key(now), DEPARTURE)
        if at <= now:
            at += timedelta(days=1)
        self._queue.append({"do": "depart", "ids": ids, "at": at.isoformat()})

    def _dependents(self, ids: list[str]) -> list[Agent]:
        """Infants stay with their mother, wherever she goes."""
        return [x for x in self.agents.values() if x.persona.infant and x.persona.parents[:1] and x.persona.parents[0] in ids]

    def circle(self, group: list[Agent]) -> list[Agent]:
        """Who would notice they were gone: everyone who knows one of them, and whoever shares their roof."""
        ids = {x.id for x in group}
        homes = {x.persona.home for x in group}
        return [o for o in self.agents.values()
                if o.id not in ids and not o.persona.infant and not o.is_away
                and (ids & o.bonds.keys() or o.persona.home in homes)]

    async def _absences(self, now: datetime) -> None:
        """Run what has come due: departures, the morning's news, and everyone whose absence has run out."""
        due, self._queue = [q for q in self._queue if datetime.fromisoformat(q["at"]) <= now], \
            [q for q in self._queue if datetime.fromisoformat(q["at"]) > now]
        for q in due:
            group = [self.agents[i] for i in q["ids"] if i in self.agents]
            if q["do"] == "depart":
                if group and not any(x.absence for x in group):
                    self.depart(group, now)
            elif q["do"] == "vanish":
                for o in self.circle(group):
                    await o._remember("observation", f"{_names(group)} {'has' if len(group) == 1 else 'have'} not "
                                      f"been seen since {q['since']}; no one knows where.", now, 6)
        done: set[str] = set()
        for a in list(self.agents.values()):
            if a.absence is None or a.id in done or datetime.fromisoformat(a.absence["until"]) > now:
                continue
            group = [self.agents[i] for i in a.absence["with"] if i in self.agents]
            done.update(x.id for x in group)
            if a.is_away:
                await self.return_home(group, now)

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

    async def return_home(self, everyone: list[Agent], now: datetime) -> None:
        """The way was closed to them: back home, with a memory of why, and the people who know them are told."""
        group = [x for x in everyone if not x.persona.infant]
        reason = everyone[0].absence["reason"]
        for x in everyone:
            x.absence = None
            x.x, x.y = self.world.target_tile(x.persona.bed) or (x.x, x.y)
            x.mode, x.plan, x.plan_date, x.tasks, x.tasks_block, x.task = "acting", [], None, [], -1, None
        for x in group:
            others = [y for y in group if y is not x]
            company = f" with {_names(others)}" if others else ""
            await x._remember("observation", f"I tried to leave Hollowmere{company}, but {reason} stopped us, "
                                             "and we turned back.", now, 8)
        for o in self.circle(group):
            await o._remember("observation", f"Word is that {_names(group)} tried to leave Hollowmere and "
                                             f"{'was' if len(group) == 1 else 'were'} turned back by {reason}.",
                              now, 7)
        self._event("returned", [x.id for x in group], f"{_names(group)} come back to Hollowmere, turned back by {reason}.")
