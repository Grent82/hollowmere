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

from hollowmere.agents.agent import CELL, Agent, day_key, due_at
from hollowmere.agents.persona import DATA, RANK

TURNED_BACK = json.loads((DATA / "world.json").read_text(encoding="utf-8"))["turned_back"]
DEPARTURE = time(22, 0)
RETURN = time(17, 0)
NEWS_AT = time(6, 0)
RELEASE = time(8, 0)
MOVE_IN = time(20, 0)


log = logging.getLogger(__name__)


def _names(agents: list[Agent]) -> str:
    return " and ".join(a.first for a in agents)


class Absences:
    """Mixed into Engine. Needs: agents, world, cfg, rng, now, _queue, _event()."""

    def schedule_departure(self, group: list[Agent], now: datetime) -> None:
        """One person or two settled on leaving: they set out at the next 22:00."""
        ids = [x.id for x in group]
        if any(x.absence for x in group) or any(q["do"] == "depart" and set(ids) & set(q["ids"]) for q in self._queue):
            return
        at = datetime.combine(day_key(now), DEPARTURE)
        if at <= now:
            at += timedelta(days=1)
        self._queue.append({"do": "depart", "ids": ids, "at": at.isoformat()})

    def schedule_move_in(self, mover: Agent, host: Agent, now: datetime) -> None:
        """Settled on living under the host's roof: the mover carries their things over at the next 20:00."""
        # A human may move into the den, but a goblin cannot go indoors, so he cannot live under a roof.
        if (mover.persona.is_goblin or mover.persona.infant or host.persona.infant
                or mover.absence or host.absence or mover.persona.home == host.persona.home
                or any(q["do"] == "move_in" and mover.id in q["ids"] for q in self._queue)):
            return
        at = datetime.combine(day_key(now), MOVE_IN)
        if at <= now:
            at += timedelta(days=1)
        self._queue.append({"do": "move_in", "ids": [mover.id], "to": host.id, "at": at.isoformat()})

    def _dependents(self, ids: list[str]) -> list[Agent]:
        """Infants stay with their mother, wherever she goes."""
        return [x for x in self.agents.values() if x.persona.infant and x.persona.parents[:1] and x.persona.parents[0] in ids]

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
                if o.id not in ids and not o.persona.infant and not o.is_away and close(o)]

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
                await self.return_home(group, now)
            else:
                await self.release(group, now)

    async def _run_entry(self, q: dict, now: datetime) -> None:
        group = [self.agents[i] for i in q["ids"] if i in self.agents]
        if q["do"] == "depart":
            if group and not any(x.absence for x in group):
                self.depart(group, now)
        elif q["do"] == "move_in":
            host = self.agents.get(q["to"])
            if group and host is not None and not group[0].absence and not host.absence:
                await self.move_in(group[0], host, now)
        elif q["do"] == "arrested":
            who, by = group[0], self.agents[q["by"]]
            await who._remember("observation", f"{by.name} had me seized and locked in the cell of the "
                                               "watchhouse.", now, 9)
            await by._remember("observation", f"I had {who.name} locked in the watchhouse cell.", now, 7)
            for o in self.circle([who]):
                if o is not by:
                    await o._remember("observation", f"Word is that {who.name} was locked in the watchhouse "
                                                     f"cell by {by.name}.", now, 7)
        elif q["do"] == "vanish":
            for o in self.circle(group):
                await o._remember("observation", f"{_names(group)} {'has' if len(group) == 1 else 'have'} not "
                                  f"been seen since {q['since']}; no one knows where.", now, 6)

    async def move_in(self, mover: Agent, host: Agent, now: datetime) -> None:
        """The mover now lives and sleeps where the host does, babies with them; both and those close are told."""
        for x in [mover, *self._dependents([mover.id])]:
            x.set_home(host.persona.home, host.persona.home_object, self.world)
        await mover._remember("observation", f"I moved into {host.name}'s home to live there.", now, 8)
        await host._remember("observation", f"{mover.name} moved into my home to live with me.", now, 8)
        for o in self.circle([mover, host]):
            await o._remember("observation", f"Word is that {mover.name} moved in with {host.name}.", now, 6)
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

    def arrest(self, by: Agent, who: Agent, now: datetime) -> bool:
        """`by` has had `who` locked in the watchhouse cell for a day or three. Anyone may do it to anyone."""
        if (by is who or any(x.persona.infant or x.persona.is_goblin or x.absence for x in (by, who))
                or sum(1 for x in self.agents.values() if x.is_jailed and not x.persona.infant) >= self.cfg.max_jailed):
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
            if not x.persona.infant:
                await x._remember("observation", f"I was let out of the cell after {days} day{'s' * (days > 1)}.", now, 7)
        self._event("released", [x.id for x in everyone if not x.persona.infant],
                    f"{everyone[0].first} is let out of the watchhouse cell.")
