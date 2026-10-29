"""One small event each morning, seen only by those who are there; a god may be behind it.

A narrator call decides it (or that the day is quiet); the engine only delivers it, as an observation, to whoever is
awake and in the place (or anywhere in town, for weather). It never touches a person's bonds, states or objects."""

from __future__ import annotations

import logging
from datetime import datetime, time, timedelta

from hollowmere.agents import prompts
from hollowmere.agents.agent import day_key, sign_seen
from hollowmere.agents.gods import GODS, GOD_BY_ID
from hollowmere.agents.schemas import daily_event_schema
from hollowmere.sim import reports

log = logging.getLogger(__name__)

PRAYED_WITHIN_DAYS = 5  # a god may stir only if mortals prayed to it this recently
GOD_EVERY_DAYS = 3  # and no more than one omen of a god in this many days
REMEMBERED = 6  # earlier events the call is shown, so none repeats


class DailyEvents:
    """Mixed into Engine. Needs: agents, world, cfg, rng, store, ctx, now, _event()."""

    def _daily_due(self, now: datetime) -> bool:
        return self._daily_for != day_key(now)

    def _wishes(self, god) -> str:
        out = reports.folder(self)
        path = out / "god_wishes.md" if out else None
        if path is None or not path.exists():
            return ""
        lines = [x for x in path.read_text(encoding="utf-8").splitlines() if x.startswith(f"- **{god.name}**")]
        return "\n".join(f"  wish: {x.split(': ', 1)[-1][:160]}" for x in lines[-2:])

    def _stirring_gods(self, now: datetime) -> list:
        """Gods mortals prayed to lately, unless one has already stirred in the last days."""
        recent = self.store.events_of("daily", now - timedelta(days=GOD_EVERY_DAYS - 1))
        if any(e.get("god") for e in recent):
            return []
        cutoff = (now - timedelta(days=PRAYED_WITHIN_DAYS)).isoformat()
        return [g for g in GODS if any(p["time"] > cutoff for p in self.store.prayers_to(g.id, 40))]

    async def _run_daily_event_task(self, now: datetime) -> None:
        """Once a day, even if it fails: a failed call costs the day its event and nothing else."""
        if not self._daily_due(now):
            return
        self._daily_for = day_key(now)
        try:
            await self._run_daily_event(now)
        except Exception:  # noqa: BLE001 - a nicety for the town must never hurt it
            log.exception("daily event failed")

    async def _run_daily_event(self, now: datetime) -> None:
        if self.rng.random() < self.cfg.daily_event_quiet_chance:
            return
        gods = self._stirring_gods(now)
        areas = tuple(a.id for a in self.world.areas.values() if a.public)
        schema = daily_event_schema(areas, tuple(g.id for g in gods))
        recent = [e["text"] for e in self.store.events_of("daily", now - timedelta(days=REMEMBERED))][-REMEMBERED:]
        town_events = self._recent_town_events(now - timedelta(days=2))
        state = next(iter(self.agents.values())).town_state or prompts.DEFAULT_TOWN_STATE
        god_lines = [f"- {g.name}, {g.domain}; {g.temperament}\n{self._wishes(g)}".rstrip() for g in gods]
        out = await self.ctx.llm.structured(
            prompts.daily_event_system_prompt(),
            prompts.daily_event(now, state, recent, town_events, god_lines), schema, label="daily_event",
            temperature=0.9)
        if out.quiet or not out.text.strip():
            return
        at = max(now, datetime.combine(day_key(now), time(out.hour)))
        self._pending_events.append({
            "at": at.isoformat(), "kind": out.kind, "text": out.text.strip(), "where": out.where,
            "god": out.god if out.god in GOD_BY_ID else "none", "clarity": out.clarity if out.god != "none" else "none",
            "importance": out.importance})

    async def _deliver_events(self, now: datetime) -> None:
        due = [e for e in self._pending_events if datetime.fromisoformat(e["at"]) <= now]
        if not due:
            return
        self._pending_events = [e for e in self._pending_events if e not in due]
        for e in due:
            god = GOD_BY_ID.get(e["god"])
            text = sign_seen(e["text"], god.name, e["clarity"]) if god else e["text"]
            for a in list(self.agents.values()):
                here = e["where"] == "town" or self.world.area_at(a.x, a.y) == e["where"]
                if a.awake and not a.is_away and not a.persona.dependent and here:
                    await a.observe(text, now, e["importance"], source="saw")
            self._event("daily", [], e["text"], sort=e["kind"], god=e["god"] if god else None, where=e["where"],
                        clarity=e["clarity"])
