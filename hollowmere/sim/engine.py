"""The simulation loop.

Each tick advances the world by `tick_seconds` of game time:

1. schedule   - every agent advances its plan; planning/decomposition LLM calls run
                concurrently and the tick waits for them (the original ran every
                agent's calls one after another)
2. move       - agents walk up to WALK_TILES along their path
3. converse   - each active conversation produces one line
4. perceive   - agents notice each other; observations go to memory without an LLM
                call, and a throttled subset triggers a reaction decision
5. background - reflection, conversation digests and day reviews run without
                holding up the clock

State is pushed to listeners (the web UI) after every tick instead of being
exchanged through JSON files on disk.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import math
import os
import random
import re
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Awaitable, Callable

from hollowmere.agents import prompts, romance
from hollowmere.agents.agent import latin, Agent, Ctx, Digest, Task, agreed_departure, agreed_meetings, agreed_move, day_key, due_at
from hollowmere.agents.gods import GOD_BY_ID
from hollowmere.agents.kinship import Kinship
from hollowmere.agents.persona import CHILD_AGE, RANK, starting_affinity
from hollowmere.agents.schemas import TownState, child_growth_schema
from hollowmere.config import SimConfig
from hollowmere.llm.client import LLMError
from hollowmere.memory.embed import Embedder
from hollowmere.memory.store import Store
from hollowmere.memory.stream import MemoryStream, repeats
from hollowmere.sim import knowledge, reports
from hollowmere.sim.absence import Absences
from hollowmere.sim.daily import DailyEvents
from hollowmere.world.world import World

log = logging.getLogger(__name__)

WALK_TILES = 4
MAX_TURNS = 12  # lines; past this a small model mostly restates itself
CHAT_COOLDOWN = timedelta(hours=3)
REACT_COOLDOWN = timedelta(minutes=50)
ARRANGED_WINDOW = timedelta(minutes=30)  # a meeting two people arranged is "on" this long either side of its hour
ATTENTION = 3  # new observations an agent stores per tick
OBSERVE_NEAR = 8  # tiles: a step that names someone farther off than this cannot have involved them
SAVE_EVERY_TICKS = 30
SAMPLE_EVERY_TICKS = 10  # where everyone is, for `hollowmere report`
FLEE_TILES = 6
HIDE_FOR = timedelta(minutes=10)
COVER_CLEAR = 8  # tiles: cover this far from every human he avoids (past sight range) is still good cover
HIDING_PLACES = ["goblin_den/nib_nest", "witchwood/bramble_thicket", "riverbank/reeds",
                 "standing_stones/brambles", "graveyard/dead_yew",
                 "tavern_back/rain_barrels", "tavern_front/ale_barrels", "tavern_side/firewood",
                 "mill_race/race_stones", "south_gap/handcart"]
# Event kinds that bear on the town as a whole; raw chat lines and routine plans are left out so the
# town review's prompt stays small and focused on what actually shifted. "prayer" belongs here too by
# that logic, but deliberately stays out: a god's answer is private to the one who prayed, and feeding
# it to the shared narrator would leak it into every agent's town_state - the same boundary enforced by
# never writing a prayer's effect into anyone's bonds (see Agent.pray()).
TOWN_EVENT_KINDS = {"showing", "birth", "reflection", "agreement", "ally", "spurn", "warm", "sighting", "intent", "chat_end",
                    "daily"}

Listener = Callable[[dict], Awaitable[None]]


def _add_counts(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    return {k: a.get(k, 0) + b.get(k, 0) for k in a.keys() | b.keys()}


@dataclass
class Conversation:
    id: str
    a: Agent  # initiator
    b: Agent
    purpose: str
    started: datetime
    transcript: list[tuple[str, str]] = field(default_factory=list)
    approach_ticks: int = 0
    talking: bool = False
    done: bool = False
    busy: bool = False
    intimate: set[str] = field(default_factory=set)  # who has said, in their own line, that this is the moment
    love_done: bool = False  # making love is confirmed once per talk, by whichever path gets there first


class Town(dict):
    """The people by id. A birth finishes in the background and adds one while a tick is still walking through the
    others, so walking goes over a snapshot."""

    def values(self):
        return list(super().values())

    def items(self):
        return list(super().items())

    def keys(self):
        return list(super().keys())

    def __iter__(self):
        return iter(list(super().keys()))


class Engine(Absences, DailyEvents):
    def __init__(self, cfg: SimConfig, world: World, agents: list[Agent], ctx: Ctx, store: Store,
                embedder: Embedder):
        self.cfg = cfg
        self.world = world
        self.agents = Town({a.id: a for a in agents})
        self.ctx = ctx
        self.store = store
        self.embedder = embedder
        self.ctx.kin = Kinship({i: x.persona for i, x in self.agents.items()})
        self._glimpsed_hour = -1
        self.pregnancies: list[dict] = []  # {"mother_id", "father_id", "start_day", "due_day", "resolved"}
        self.now: datetime = cfg.start
        self.tick_count = 0
        self.paused = False
        self.speed = 1.0
        self.conversations: dict[str, Conversation] = {}
        self.events: deque[dict] = deque(maxlen=400)
        self.listeners: set[Listener] = set()
        self._bg: dict[str, asyncio.Task] = {}
        self._convo_ids = itertools.count(1)
        self._session = os.urandom(2).hex()  # conversation ids must not repeat across restarts of one save
        self._stop = asyncio.Event()
        self.rng = random.Random(1347)
        self._judging: set[tuple[str, str]] = set()  # (goblin, human) after a talk, until the human decides
        self._situations: dict[str, tuple[frozenset[str], datetime]] = {}  # goblin -> who was in sight at his last choice
        self._noticed: dict[tuple[str, str], datetime] = {}  # (goblin, human) -> when the human last caught sight of him
        self._sought: dict[str, datetime] = {}  # target -> when they last came face to face with their seeker
        self._seeking: dict[str, str] = {}  # agent -> whoever they are currently on their way to find
        self._llm_before: dict = {}  # LLM counters from earlier sessions of this save
        self._agreements: dict[str, dict[str, Digest]] = {}  # convo id -> agent id -> their digest of it
        self._town_reviewed: date | None = None  # day_key() of the last town-wide review, if any
        self._town_task: asyncio.Task | None = None
        self._grow_tried: dict[str, date] = {}  # child -> the day growing up was last tried
        self._reported: date | None = None  # the day the evening reports last started
        self._report_task: asyncio.Task | None = None
        self._daily_for: date | None = None  # the day the morning's event was last made
        self._known_base = dict(ctx.known_as)  # what the town knew of each person from the start
        self._whispers_for: date | None = None  # the day what the town whispers was last worked out
        self._daily_task: asyncio.Task | None = None
        self._pending_events: list[dict] = []  # made this morning, still to happen at their hour
        self._queue: list[dict] = []  # departures and the news of them still to come, each with an "at" time

    # ---------------------------------------------------------------- running

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        while not self._stop.is_set():
            if self.paused:
                await asyncio.sleep(0.2)
                continue
            started = loop.time()
            try:
                await self.tick()
            except Exception as e:  # noqa: BLE001 - one bad tick must not stop the town for good
                # Before this, the loop task simply ended and the viewer froze on its last frame.
                log.exception("tick at %s failed; trying again", self.now)
                self._event("error", [], f"The simulation hit an error and is retrying: {e!r}"[:300])
                await asyncio.sleep(2)
                continue
            delay = self.cfg.tick_interval / max(self.speed, 0.01) - (loop.time() - started)
            if delay > 0:
                await asyncio.sleep(delay)

    def stop(self) -> None:
        self._stop.set()

    async def tick(self) -> None:
        now = self.now
        await self._absences(now)
        await self._schedule(now)
        self._seek(now)
        self._move(now)
        self._carry_babies()
        await self._stealth(now)
        await self._converse(now)
        await self._perceive(now)
        self._background(now)
        if now.hour != self._glimpsed_hour:
            self._glimpsed_hour = now.hour
            self._write_glimpses(now)
        if self._daily_due(now) and (self._daily_task is None or self._daily_task.done()):
            self._daily_task = asyncio.create_task(self._run_daily_event_task(now))
        await self._deliver_events(now)
        if self._whispers_for != day_key(now) and self.tick_count > 0:
            self._whispers_for = day_key(now)
            asyncio.create_task(self._refresh_whispers_task(now))
        if self.tick_count % SAMPLE_EVERY_TICKS == 0:
            self._sample(now)
        self.tick_count += 1
        self.now = now + timedelta(seconds=self.cfg.tick_seconds)
        if self.tick_count % SAVE_EVERY_TICKS == 0:
            self.save()
        await self._broadcast({"type": "tick", **self.state_view()})

    # ------------------------------------------------------------- phases

    async def _schedule(self, now: datetime) -> None:
        # Planning a day can be followed by decomposing its first block, so repeat
        # until every agent can act (each round runs all agents' calls concurrently).
        for _ in range(3):
            jobs = []
            for a in self.agents.values():
                if a.is_away:
                    continue
                if a.favor is not None and now >= datetime.fromisoformat(a.favor["expires"]):
                    a.favor = None
                if a.hiding_until is not None:
                    if now < a.hiding_until:
                        continue
                    a.hiding_until, a.task = None, None  # coast is clear: back to the plan
                if a.convo_id is None:
                    job = a.schedule_job(self.ctx, now)
                    if job is not None:
                        jobs.append(self._guard(job, a, "schedule"))
            if not jobs:
                break
            await asyncio.gather(*jobs)
        for a in self.agents.values():
            if a.plan and a.plan_date == day_key(now) and a.announced != a.plan_date:
                a.announced = a.plan_date
                self._event("plan", [a.id], f"{a.first} plans the day: " +
                            "; ".join(f"{b.start:%H:%M} {b.activity}" for b in a.plan))

    def _move(self, now: datetime) -> None:
        for a in self.agents.values():
            a.steps = []
            if a.persona.dependent:
                a.path = []  # carried or led along, never walking by itself
            if a.convo_id is not None or not a.path:
                continue
            fleeing = a.hiding_until is not None and now < a.hiding_until
            for _ in range(FLEE_TILES if fleeing else WALK_TILES):
                if not a.path:
                    break
                a.x, a.y = a.path.pop(0)
                a.steps.append((a.x, a.y))
            if not a.path:
                if fleeing:
                    a.mode = "hiding"
                else:
                    a.arrive()
        self._update_objects()

    def _grow_children(self, now: datetime) -> None:
        """A newborn that is old enough (4 days, a goblin 3) is written on as a child, once a day until it works."""
        for baby in list(self.agents.values()):
            born = romance.born_on(baby.persona)
            if not baby.persona.infant or born is None or len(baby.persona.parents) != 2:
                continue
            if (day_key(now) - born).days < (3 if baby.persona.species == "goblin" else 4):
                continue
            if self._grow_tried.get(baby.id) == day_key(now):
                continue
            self._grow_tried[baby.id] = day_key(now)
            self._spawn(baby, self._grow_up(baby, now), f"grow:{baby.id}")

    async def _grow_up(self, child: Agent, now: datetime) -> None:
        mother, father = (self.agents.get(i) for i in child.persona.parents)
        if mother is None or father is None:
            return
        p = child.persona
        goblin = p.species == "goblin"
        pool = romance.growth_pool(mother.persona, father.persona)
        if len(pool) < 2:
            return
        low, high = (2, 3) if goblin else (3, 4)
        schema = child_growth_schema(pool, min(low, len(pool)), min(high, len(pool)))
        try:
            out = await self.ctx.llm.structured(
                prompts.child_draft_system_prompt(), prompts.child_growth(child, mother.persona, father.persona, p.species,
                                     self.world.place_name(p.home)),
                schema, label="child_growth", temperature=0.7)
        except LLMError as e:
            log.warning("%s: growing up failed (%s)", child.id, e)
            return
        traits = [*out.traits, *(t for t in romance.GOBLIN_TRAITS if goblin)]
        look = latin(out.look_now)
        # the new look goes after the old one, unless the model already carried the old one over
        old_look = re.sub(r"\b(?:infant|newborn)\b", "small child", p.appearance, flags=re.I)
        appearance = look if p.appearance[:40].lower() in look.lower() else f"{old_look} {look}".strip()
        grown = p.model_copy(update={
            "age": CHILD_AGE, "traits": ", ".join(dict.fromkeys(traits)), "appearance": appearance,
            "currently": latin(out.currently), "lifestyle": latin(out.lifestyle)})
        child.persona = grown
        child.plan, child.plan_date, child.tasks, child.tasks_block, child.task = [], None, [], -1, None
        day = day_key(now).isoformat()
        child.history = [*child.history, {"text": f"Grew from a newborn into a child on {day}.", "since": day}]
        generated = [d for d in self.store.get("generated_personas", []) if d["id"] != child.id]
        self.store.put("generated_personas", [*generated, grown.model_dump(mode="json")])
        word = "son" if grown.sex == "male" else "daughter"
        for r in self.agents.values():
            bond = r.bonds.get(child.id)
            if bond and str(bond.get("kin", "")).startswith("your newborn"):
                bond["kin"] = f"your young {word}"
        for parent in (mother, father):
            await parent._remember("observation", f"{child.name} has grown from a newborn into a small child, who now "
                                                  f"follows me about and says a few words.", now, 7)
        self._event("grew", [child.id, mother.id, father.id], f"{child.first} has grown into a small child.")

    WHISPER_MIN = 3  # unconcerned people who heard of it, before the town can be said to whisper it
    WHISPERS_PER_PERSON = 2

    def _whispered(self, facts: list) -> dict[str, str]:
        """What the town knows of each person: what it knew from the start, then what is whispered of them."""
        lines: dict[str, list[str]] = {}
        for f in sorted(knowledge.scan(self, facts), key=lambda f: f["since"], reverse=True):
            spread = [k for k in f["knowers"] if not k["involved"] and k["how"].split()[0] in ("told", "heard")]
            if f["kind"] != "goblin" and len(spread) >= self.WHISPER_MIN:
                for pid in f["about"]:
                    lines.setdefault(pid, []).append(f"It is whispered that {f['text']}.")
        known = dict(self._known_base)
        for pid, ls in lines.items():
            known[pid] = f"{self._known_base.get(pid, '')} {' '.join(ls[:self.WHISPERS_PER_PERSON])}".strip()
        return known

    def _apply_known(self, known: dict[str, str]) -> None:
        self.ctx.known_as.clear()
        self.ctx.known_as.update(known)

    def _refresh_whispers(self) -> None:
        self._apply_known(self._whispered(knowledge.facts_of(self)))

    async def _refresh_whispers_task(self, now: datetime) -> None:
        """Once a day, off the main thread (reading every memory takes seconds); the swap itself is not."""
        self._whispers_for = day_key(now)
        try:
            facts = knowledge.facts_of(self)  # touches the save, so it is made here
            self._apply_known(await asyncio.to_thread(self._whispered, facts))
        except Exception:  # noqa: BLE001 - a nicety must never hurt the town
            log.exception("whispers failed")

    def _carry_babies(self) -> None:
        """A baby is wherever the parent who shares its home is. A goblin child is carried only by a goblin: a
        human leaves it hidden in its bed rather than show it to the town."""
        for baby in self.agents.values():
            if not baby.persona.dependent or baby.absence:
                continue
            for pid in baby.persona.parents:
                carer = self.agents.get(pid)
                if carer is None or carer.persona.home != baby.persona.home or carer.absence:
                    continue
                if baby.persona.species == "goblin" and carer.persona.species != "goblin":
                    continue
                baby.x, baby.y = carer.x, carer.y
                break

    # ---------------------------------------------------------------- stealth

    def _humans(self) -> list[Agent]:
        return [a for a in self.agents.values() if a.persona.is_human]

    def _in_sight(self, g: Agent, ignore: Agent | None = None) -> list[Agent]:
        return [h for h in self._humans() if h is not ignore and h.awake and self._visible(g, h)]

    async def _stealth(self, now: datetime) -> None:
        """The goblin chooses what to do whenever someone who could give him away comes into sight.

        Code only notices that the scene changed; approaching, watching, hiding or fleeing is the model's call.
        The same people in sight are not asked about again until REACT_COOLDOWN has passed."""
        choices = []
        for g in self.agents.values():
            if not g.persona.is_goblin or not g.awake or g.thinking:
                continue
            c = self.conversations.get(g.convo_id) if g.convo_id else None
            partner = (c.b if c.a is g else c.a) if c else None
            seen = self._in_sight(g, ignore=partner)
            if all(h.id in g.allies for h in seen):
                self._situations.pop(g.id, None)
                continue
            ids = frozenset(h.id for h in seen)
            last = self._situations.get(g.id)
            if last is not None and ids <= last[0] and now - last[1] < REACT_COOLDOWN:
                if g.mode == "hiding":
                    g.hiding_until = now + HIDE_FOR  # he chose cover from these people; they are still about
                continue
            self._situations[g.id] = (ids, now)
            choices.append(self._goblin_chooses(g, seen, c, partner, now))
        if choices:
            await asyncio.gather(*choices)

    async def _goblin_chooses(self, g: Agent, seen: list[Agent], c: Conversation | None,
                              partner: Agent | None, now: datetime) -> None:
        seen = sorted(seen, key=lambda h: math.hypot(h.x - g.x, h.y - g.y))
        entries = [(h, math.hypot(h.x - g.x, h.y - g.y),
                    now - self._noticed.get((g.id, h.id), datetime.min) < timedelta(minutes=10)) for h in seen]
        cover = (self.world.place_name(g.task.place)
                 if g.hiding_until is not None and g.task is not None and g.task.place in HIDING_PLACES else None)
        try:
            out, by_name = await g.goblin_choice(self.ctx, now, entries, partner, g.mode == "hiding", cover)
            action, whom, thought = out.action, by_name.get(out.whom), out.thought.strip()
        except LLMError as e:
            log.warning("%s: choice failed (%s); hiding", g.id, e)
            out, action, whom, thought = None, "hide", None, ""  # the cautious default when the model fails
        for h in seen:
            self._sought[h.id] = now  # face to face: whoever he was creeping after is found
        self._seeking.pop(g.id, None)
        names = ", ".join(h.first for h in seen)
        said = f': "{thought}"' if thought else "."
        self._event("choice", [g.id] + [h.id for h in seen], f"{g.first} sees {names} and decides to {action}.",
                    feed=False, action=action, whom=whom.id if whom else None, thought=thought,
                    stances={h.id: g.bonds.get(h.id, {}).get("stance") for h in seen})
        if (action == "approach" and whom is not None and c is None and whom.awake and whom.convo_id is None
                and (now - g.last_chat.get(whom.id, datetime.min) >= CHAT_COOLDOWN
                     or self._arranged(g, whom, now) is not None)):
            g.hiding_until, g.task, g.mode = None, None, "acting"
            self._start_conversation(g, whom, out.reason or f"to see whether {whom.first} will help him", now)
            return
        if action in ("hide", "flee"):
            if c is not None:
                self._end_conversation(c, "fled")
            strangers = [h for h in seen if h.id not in g.allies] or seen
            self._flee(g, strangers, now, far=action == "flee", said=said)
            return
        if g.mode == "hiding":
            g.hiding_until = now + HIDE_FOR
        self._event("watch", [g.id], f"{g.first} keeps still and watches {names}{said}")

    def _flee(self, g: Agent, threats: list[Agent], now: datetime, far: bool = False, said: str = ".") -> None:
        hostile = self._hostile(g, threats)
        if self._cover_holds(g, hostile):
            g.hiding_until = now + HIDE_FOR  # already on his way to cover, or in it, and it is still clear of them all
            return
        spot = self._hiding_place(g, hostile, far)
        if spot is not None:
            place, path = spot
            g.hiding_until = now + HIDE_FOR
            g.task = Task(now, now + HIDE_FOR, "bolts for cover" if far else "slinks off to hide", place, "\U0001fae3")
            g.path = path
            g.mode = "walking" if path else "hiding"
            names = ", ".join(t.first for t in threats)
            verb = f"bolts from {names} to" if far else f"spots {names} and slinks off to hide at"
            self._event("flee" if far else "hide", [g.id],
                        f"{g.first} {verb} {self.world.place_name(place)}{said}")

    def _hostile(self, g: Agent, threats: list[Agent]) -> list[Agent]:
        """Whoever he must keep clear of when picking cover: those who set him off, and every other outdoor
        human who is not his ally. Cover chosen against the trigger alone runs him into the next person."""
        out = list(threats)
        for h in self._humans():
            area = self.world.area_at(h.x, h.y)
            if h in out or not h.awake or h.id in g.allies or (area is not None and self.world.areas[area].indoor):
                continue
            out.append(h)
        return out

    def _cover_holds(self, g: Agent, hostile: list[Agent]) -> bool:
        """His current cover (reached or still under way) is clear of everyone he is avoiding."""
        if g.hiding_until is None or g.task is None or g.task.place not in HIDING_PLACES:
            return False
        tile = self.world.target_tile(g.task.place)
        return tile is not None and min(math.hypot(tile[0] - h.x, tile[1] - h.y) for h in hostile) > COVER_CLEAR

    def _hiding_place(self, g: Agent, threats: list[Agent], far: bool = False) -> tuple[str, list[tuple[int, int]]] | None:
        """Nearby cover to hide, or the cover farthest from them to flee to."""
        best, best_score = None, -math.inf
        for place in HIDING_PLACES:
            tile = self.world.target_tile(place)
            if tile is None:
                continue
            path = self.world.find_path((g.x, g.y), tile) if tile != (g.x, g.y) else []
            if tile != (g.x, g.y) and not path:
                continue
            safety = min(math.hypot(tile[0] - t.x, tile[1] - t.y) for t in threats)
            score = min(safety, 40) - 0.1 * len(path) if far else min(safety, 15) - 0.35 * len(path)
            if score > best_score:
                best, best_score = (place, path), score
        return best

    def _seek(self, now: datetime) -> None:
        """Whoever an agent chose, when planning this stretch of the day, to go and find."""
        for g in self.agents.values():
            target = self.agents.get(g.seek) if g.seek else None
            if (target is None or g.persona.infant or not g.awake or g.convo_id is not None or g.hiding_until is not None
                    or g.seek_block != g.current_block(now) or not target.awake or target.absence
                    or now - self._sought.get(target.id, datetime.min) < CHAT_COOLDOWN):
                self._seeking.pop(g.id, None)
                continue
            area = self.world.area_at(target.x, target.y)
            if g.persona.is_goblin and area is not None and self.world.areas[area].indoor:
                continue  # he waits for them to come out; a human visitor can just walk in
            if g.path and self.tick_count % 3:
                continue  # re-aim every few ticks
            path = self.world.find_path((g.x, g.y), (target.x, target.y))[:-2]
            if not path:
                continue
            g.path, g.mode = path, "walking"
            if self._seeking.get(g.id) != target.id:
                self._seeking[g.id] = target.id
                if g.persona.is_goblin:
                    self._event("seek", [g.id, target.id], f"{g.first} creeps after {target.first}.")
                else:
                    self._event("visit", [g.id, target.id], f"{g.first} goes to find {target.first}.")

    def _spot_goblin(self, h: Agent, g: Agent, now: datetime) -> bool:
        """A human who does not know the goblin may glimpse him. Returns whether they did."""
        d = math.hypot(h.x - g.x, h.y - g.y)
        p = 0.6 if d <= 2 else 0.25 if d <= 4 else 0.1
        if g.mode == "hiding":
            p *= 0.15
        if now.hour < 7 or now.hour >= 18:
            p *= 0.5
        if self.rng.random() >= p:
            return False
        self._noticed[(g.id, h.id)] = now
        area = self.world.area_at(g.x, g.y)
        where = f"at {self.world.areas[area].name}" if area else "in the shadows by the road"
        looks = f", {g.persona.appearance}," if g.persona.appearance else ""
        text = f"I glimpsed a goblin{looks} {where}!"
        if h.memory.seen_recently(text, now, minutes=60):
            return True
        self._spawn(h, h.observe(text, now, 8), f"sight:{now.isoformat()}")
        self._event("sighting", [h.id, g.id], f"{h.first} glimpses a goblin {where}.")
        return True

    def _update_objects(self) -> None:
        for o in self.world.objects.values():
            o.state = "idle"
        for a in self.agents.values():
            if a.task and a.mode in ("acting", "asleep"):
                _, _, obj_id = a.task.place.partition("/")
                if obj_id in self.world.objects:
                    self.world.objects[obj_id].state = f"{a.first}: {a.task.activity}"

    async def _converse(self, now: datetime) -> None:
        turns = []
        for c in list(self.conversations.values()):
            if c.done or c.busy:
                continue
            if not c.talking:
                self._approach(c)
                continue
            turns.append(self._turn(c, now))
        if turns:
            await asyncio.gather(*turns)

    def _approach(self, c: Conversation) -> None:
        a, b = c.a, c.b
        c.approach_ticks += 1
        if abs(a.x - b.x) + abs(a.y - b.y) <= 2:
            c.talking = True
            a.mode = b.mode = "talking"
            self._event("chat_start", [a.id, b.id], f"{a.first} approaches {b.first}.")
            return
        if c.approach_ticks > 12:
            self._end_conversation(c, "never_reached")
            return
        path = self.world.find_path((a.x, a.y), (b.x, b.y))
        a.mode = "walking"
        a.steps = []
        for step in path[: max(0, min(WALK_TILES, len(path) - 1))]:
            a.x, a.y = step
            a.steps.append(step)

    async def _turn(self, c: Conversation, now: datetime) -> None:
        c.busy = True
        speaker, listener = (c.a, c.b) if len(c.transcript) % 2 == 0 else (c.b, c.a)
        try:
            purpose = c.purpose if speaker is c.a else ""
            said = [t for _, t in c.transcript]
            invited = listener.id in c.intimate and speaker.id not in c.intimate  # the other has proposed: answer
            out = await speaker.say(self.ctx, now, listener, purpose, c.transcript, invited=invited)
            if c.done:  # someone was taken away while the model was answering
                return
            line = out.says.strip().strip('"')
            if repeats(line, said):  # own lines, or parroting the other: small models loop
                out = await speaker.say(self.ctx, now, listener, purpose, c.transcript, avoid=line, invited=invited)
                line = out.says.strip().strip('"')
                if repeats(line, said):  # twice: the talk has run dry
                    self._end_conversation(c, "repeat")
                    return
            c.transcript.append((speaker.name, line))
            speaker.bubble, listener.bubble = line, None
            self._event("chat", [speaker.id, listener.id], line, speaker=speaker.id, convo=c.id)
            if getattr(out, "intimate", False):
                c.intimate.add(speaker.id)
                if len(c.intimate) == 2:  # both said yes themselves: they go off alone; _settle confirms it
                    self._end_conversation(c, "goodbye")
                    return
            proposed = getattr(out, "intimate", False) and listener.id not in c.intimate  # this very line asks, unanswered
            if out.end or len(c.transcript) >= MAX_TURNS + (1 if proposed and len(c.transcript) == MAX_TURNS else 0):
                self._end_conversation(c, "goodbye" if out.end else "limit")
        except LLMError as e:
            log.warning("conversation %s failed: %s", c.id, e)
            self._end_conversation(c, "error")
        finally:
            c.busy = False

    def _end_conversation(self, c: Conversation, reason: str) -> None:
        """reason: goodbye, limit, repeat, error, or (cut short before it began) fled, never_reached."""
        c.done = True
        self.conversations.pop(c.id, None)
        for x, y in ((c.a, c.b), (c.b, c.a)):
            x.convo_id = None
            x.bubble = None
            x.last_chat[y.id] = self.now
            x.task = None  # resume the schedule (re-pathing from where they stand)
            x.mode = "acting"
            if c.transcript:
                x.note_topic(y.first, c.purpose or c.transcript[0][1])
                if y.persona.is_goblin and x.persona.is_human:
                    self._judging.add((y.id, x.id))
                self._spawn(x, x.digest_chat(self.ctx, self.now, y, list(c.transcript)), f"digest:{c.id}",
                            on_done=self._digested(c, x, y))
        if c.transcript:  # what they agreed to do together is read once, for both of them
            self._spawn(c.a, self._agree(c), f"agree:{c.id}")
            self._spawn(c.a, self._leave(c), f"leave:{c.id}")
            self._spawn(c.a, self._settle_move(c), f"move:{c.id}")
        if reason not in ("fled", "never_reached"):  # they actually came face to face, not just tried to
            self._sought[c.a.id] = self._sought[c.b.id] = self.now  # either side's seek should back off now, too
        self._event("chat_end", [c.a.id, c.b.id], f"{c.a.first} and {c.b.first} part ways.",
                    feed=reason not in ("fled", "never_reached"), transcript=c.transcript, convo=c.id,
                    reason=reason, purpose=c.purpose)

    def _visible(self, a: Agent, b: Agent) -> bool:
        if a.is_away or b.is_away:
            return False
        if math.hypot(a.x - b.x, a.y - b.y) > self.cfg.vision_radius:
            return False
        area_a, area_b = self.world.area_at(a.x, a.y), self.world.area_at(b.x, b.y)
        if area_a == area_b:
            return True
        indoor = lambda aid: aid is not None and self.world.areas[aid].indoor  # noqa: E731
        return not indoor(area_a) and not indoor(area_b)

    def _seen_doing(self, b: Agent) -> str:
        """What a bystander may store about `b`. A task's activity is the planned step. So a step naming someone beyond OBSERVE_NEAR is reduced to where `b` is."""
        if b.task is None or b.mode in ("asleep", "talking", "hiding", "sex"):
            return b.activity_phrase(self.world)
        for other in self.agents.values():
            if other is b:
                continue
            tokens = [w for w in other.persona.name.replace("’", "").split() if len(w) > 2]
            named = any(re.search(rf"\b{re.escape(w)}(?:['’]s)?\b", b.task.activity) for w in tokens)
            if named and abs(b.x - other.x) + abs(b.y - other.y) > OBSERVE_NEAR:
                return b.activity_phrase(self.world, detail=False)
        return b.activity_phrase(self.world)

    def _write_glimpses(self, now: datetime) -> None:
        """Once an hour each watcher's sightings are put in their own words, in the background."""
        for a in self.agents.values():
            if a.glimpses:
                self._spawn(a, a.write_glimpses(self.ctx, now), "glimpses")

    async def _perceive(self, now: datetime) -> None:
        reactions = []
        for a in self.agents.values():
            if not a.awake or a.convo_id is not None or a.thinking:
                continue
            seen = sorted((b for b in self.agents.values() if b is not a and self._visible(a, b)),
                          key=lambda b: abs(a.x - b.x) + abs(a.y - b.y))
            stored = 0
            candidate = None
            for b in seen:
                if b.persona.is_goblin and a.persona.is_human and a.id not in b.allies:
                    if a.knows(b) and b.mode != "hiding":
                        self._noticed[(b.id, a.id)] = now  # someone who has met him sees him plainly
                        if candidate is None and self._may_react(a, b, now):
                            candidate = b  # she may also be the one who approaches, not only he
                    else:
                        if (b.id, a.id) not in self._judging and self._spot_goblin(a, b, now) \
                                and candidate is None and self._may_react(a, b, now):
                            candidate = b  # whether to call out, follow or raise the alarm is theirs to decide
                        continue
                if stored >= ATTENTION:
                    break
                text = f"{b.name} {self._seen_doing(b)}."
                known = b.id in a.persona.relationships
                if not a.glimpse(text, now, 3 if known else 2, [b.id],
                                 "he" if b.persona.sex == "male" else "she"):
                    continue
                stored += 1
                if self.is_showing(b.id):
                    belly = self._belly_text(b)
                    if not a.memory.seen_recently(belly, now, minutes=720):
                        await a.observe(belly, now, 5, source="saw", about=[b.id])
                if candidate is None and self._may_react(a, b, now):
                    candidate = b
            if candidate is not None:
                reactions.append(self._react(a, candidate, now))
        if reactions:
            await asyncio.gather(*reactions)

    def _arranged(self, a: Agent, b: Agent, now: datetime) -> dict | None:
        """A meeting `a` and `b` arranged that is due about now and not yet kept, from either one's calendar."""
        for x, y in ((a, b), (b, a)):
            for c in x.commitments:
                if c.get("with") == y.id and not c.get("met") and abs(due_at(c) - now) <= ARRANGED_WINDOW:
                    return c
        return None

    def _may_react(self, a: Agent, b: Agent, now: datetime) -> bool:
        if not b.awake or a.absence or b.absence or b.convo_id is not None or a.convo_id is not None:
            return False
        if now - a.last_chat.get(b.id, datetime.min) < CHAT_COOLDOWN and self._arranged(a, b, now) is None:
            return False
        if a.persona.is_goblin and any(h.id not in a.allies for h in self._in_sight(a)):
            return False  # _stealth asks him what to do about anyone who could give him away
        if a.hiding_until is not None or b.hiding_until is not None:
            return False
        return now - a.last_react.get(b.id, datetime.min) >= REACT_COOLDOWN

    async def _react(self, a: Agent, b: Agent, now: datetime) -> None:
        arranged = self._arranged(a, b, now)
        talk, purpose = await a.react(self.ctx, now, b, arranged["activity"] if arranged else "")
        if talk:
            self._start_conversation(a, b, purpose, now)

    def _start_conversation(self, a: Agent, b: Agent, purpose: str, now: datetime) -> None:
        # Backstop for every caller, not just the ones that already check this themselves (_may_react,
        # _goblin_chooses' approach branch): nobody re-opens a talk within CHAT_COOLDOWN of the last one.
        arranged = self._arranged(a, b, now)
        if now - a.last_chat.get(b.id, datetime.min) < CHAT_COOLDOWN and arranged is None:
            return
        if a.convo_id is not None or b.convo_id is not None or not b.awake or a.absence or b.absence:
            return
        for x, y in ((a, b), (b, a)):  # the arrangement is kept by this very talk: it cannot be cashed in twice
            for c in x.commitments:
                if c.get("with") == y.id and abs(due_at(c) - now) <= ARRANGED_WINDOW:
                    c["met"] = True
        cid = f"c{next(self._convo_ids)}-{self._session}"
        a.convo_id = b.convo_id = cid
        b.mode = "waiting"
        b.path = []
        self.conversations[cid] = Conversation(cid, a, b, purpose, now)
        self._event("intent", [a.id, b.id], f"{a.first} wants to speak with {b.first}: {purpose}", convo=cid)

    def _background(self, now: datetime) -> None:
        for a in self.agents.values():
            for text in a.new_reflections:
                self._event("reflection", [a.id], text)
            a.new_reflections.clear()
            for p in a.new_prayers:
                god = GOD_BY_ID.get(p["god"])
                self._event("prayer", [a.id], f"{a.first} prays to {god.name if god else p['god']}: {p['text']}",
                            form=p["form"], god=p["god"], clarity=p.get("clarity", "none"))
            a.new_prayers.clear()
            if a.is_away:
                continue  # out of town: no prayer, reflection or day review until they are back
            idx = a.current_block(now)
            if idx is not None and a.plan[idx].god and a.prayed_block != idx:
                a.prayed_block = idx  # set before spawning: the next tick must not re-fire the same block
                self._spawn(a, a.pray(self.ctx, now, a.plan[idx].god), "pray")
            if a.importance_acc >= self.cfg.reflection_threshold:
                self._spawn(a, a.reflect(self.ctx, now), "reflect")
            elif a.mode == "asleep" and a.plan_date == day_key(now) and a.reviewed_date != a.plan_date \
                    and a.wake is not None and now > a.wake:
                self._spawn(a, a.review_day(self.ctx, now), "review")
        self._pregnancy_phases(now)
        self._grow_children(now)
        for preg in self.pregnancies:
            mother = self.agents.get(preg["mother_id"])
            if not preg["resolved"] and date.fromisoformat(preg["due_day"]) <= day_key(now) \
                    and not (mother is not None and mother.absence):  # away or in the cell: it waits for her
                preg["resolved"] = True
                if mother is not None:
                    self._spawn(mother, self._give_birth(preg, now),
                               f"birth:{preg['mother_id']}:{preg['due_day']}")
        if self._town_review_due(now) and (self._town_task is None or self._town_task.done()):
            self._town_task = asyncio.create_task(self._run_town_review(now))
        if self._report_due(now) and (self._report_task is None or self._report_task.done()):
            self._reported = day_key(now)  # once a day, even if it fails
            self._report_task = asyncio.create_task(self._run_reports(now))

    def _report_due(self, now: datetime) -> bool:
        return now.time() >= reports.REPORT_AT and self._reported != day_key(now)

    async def _run_reports(self, now: datetime) -> None:
        try:
            await reports.run(self, now)
        except Exception:  # noqa: BLE001 - a collection for the player's eyes must never hurt the town
            log.exception("evening reports failed")

    def _town_review_due(self, now: datetime) -> bool:
        if self._town_reviewed is None:
            return True
        return (day_key(now) - self._town_reviewed).days >= self.cfg.town_review_days

    def _recent_town_events(self, since: datetime) -> list[str]:
        # self.events is only the bounded, unpersisted UI feed (deque(maxlen=400)); a busy town can
        # wrap that well inside a day, and it is always empty right after a restart. The durable
        # store has the real history.
        return self.store.events_since(since, TOWN_EVENT_KINDS)

    async def _run_town_review(self, now: datetime) -> None:
        try:
            await self._review_town(now)
        except Exception:  # noqa: BLE001 - a failed review must not kill the loop or retry forever
            log.exception("town review failed")
            self._town_reviewed = day_key(now)

    async def _review_town(self, now: datetime) -> None:
        current = next(iter(self.agents.values())).town_state or prompts.DEFAULT_TOWN_STATE
        since = now - timedelta(days=self.cfg.town_review_days)
        events = self._recent_town_events(since)
        out = await self.ctx.llm.structured(
            prompts.narrator_system_prompt(), prompts.town_review(current, now, events),
            TownState, label="town_review")
        text = out.text.strip()
        if text:
            for a in self.agents.values():
                a.town_state = text
        self._town_reviewed = day_key(now)

    def _digested(self, c: Conversation, x: Agent, y: Agent):
        judged = self._judged(x, y)

        def done(result: Digest | None) -> None:
            if judged is not None:
                judged(result.regard if result else None)
            notes = self._agreements.setdefault(c.id, {})
            notes[x.id] = result if result else Digest()
            if len(notes) == 2:
                del self._agreements[c.id]
                self._settle(c, notes)
        return done

    async def _agree(self, c: Conversation) -> None:
        """The meetings the two agreed on, read once from the whole talk and put on both calendars alike.

        A meeting needs both of them there, so it is kept only if it fits both calendars; a clash drops it
        for both rather than leaving one person waiting alone."""
        a, b = c.a, c.b
        for entry in await agreed_meetings(self.ctx, a, b, self.now, list(c.transcript)):
            mine, theirs = {**entry, "with": b.id}, {**entry, "with": a.id}
            if not (a._acceptable(mine, self.world, self.now) and b._acceptable(theirs, self.world, self.now)):
                continue
            await a._commit(mine, self.world, self.now)
            await b._commit(theirs, self.world, self.now)
            self._agreement_event(a, b, entry)

    async def _leave(self, c: Conversation) -> None:
        """Whoever settled in the talk on leaving Hollowmere, read once for both, sets out that night."""
        group = await agreed_departure(self.ctx, c.a, c.b, self.now, list(c.transcript))
        if group:
            self.schedule_departure(group, self.now)

    async def _settle_move(self, c: Conversation) -> None:
        """Whoever settled in the talk on moving into the other's home, read once for both, moves that evening."""
        moved = await agreed_move(self.ctx, c.a, c.b, self.now, list(c.transcript))
        if moved:
            self.schedule_move_in(*moved, self.now)

    def _settle(self, c: Conversation, notes: dict[str, Digest]) -> None:
        """Both sides wrote down the talk separately; act on what both of them say happened in it.

        Meetings are not decided here: `_agree` reads them once for both."""
        a, b = c.a, c.b
        for me, other in ((a, b), (b, a)):  # each side says which way it went; the first told stands
            said = notes.get(me.id, Digest()).arrested
            if said != "no":
                self.arrest(*((me, other) if said == "i_locked_them_up" else (other, me)), self.now)
        said_so = len(c.intimate) == 2 or (notes.get(a.id, Digest()).made_love and notes.get(b.id, Digest()).made_love)
        if said_so and not c.love_done:
            c.love_done = True
            self._spawn(a, self._confirm_love(a, b, self.now), f"love:{c.id}")

    async def _confirm_love(self, a: Agent, b: Agent, now: datetime) -> None:
        """Both sides' own digest agreed this conversation included making love: close the gap in
        affinity, mark the pair as having already happened (so future occasions skip the affinity
        gate entirely), and - for a male+female pair with no pregnancy already in flight - conceive."""
        if self.ctx.kin.related(a.id, b.id) or a.persona.dependent or b.persona.dependent:
            return  # family, or a child: whatever the two digests said, this was not that
        for x, y in ((a, b), (b, a)):
            bond = x.bonds[y.id]
            if RANK[bond["stance"]] < RANK["close"]:
                bond["stance"], bond["affinity"] = "close", starting_affinity("close")
            bond["intimate"] = True
            await x._remember("chat", f"{y.first} and I talked, and it became more - we made love.", now, 7,
                              source="lived", about=[y.id])
        if {a.persona.sex, b.persona.sex} == {"male", "female"}:
            mother, father = (a, b) if a.persona.sex == "female" else (b, a)
            already = any(p["mother_id"] == mother.id and not p["resolved"] for p in self.pregnancies)
            if not already and self.rng.random() < self.cfg.conception_chance:
                due = day_key(now) + timedelta(days=self.cfg.gestation_days)
                self.pregnancies.append({"mother_id": mother.id, "father_id": father.id,
                                         "start_day": day_key(now).isoformat(), "due_day": due.isoformat(),
                                         "resolved": False})
                self._event("pregnancy", [mother.id, father.id],
                           f"{mother.first} conceives a child with {father.first}.")

    async def _tell_mothers_family(self, child: Agent, mother: Agent, when: datetime) -> None:
        """Her own family sees her give birth: parents, grown children, siblings learn whose child this is and
        what it is to them. Not her own parents, who hear of it by eye like the town does (a secret lover stays
        secret), and not the father's side: who the father is stays hers to say."""
        for r in self.agents.values():
            if r.id in (child.id, mother.id) or r.persona.dependent or not self.ctx.kin.related(r.id, mother.id) \
                    or r.id in self.ctx.kin.parents.get(mother.id, set()):
                continue
            word = self.ctx.kin.what_is(child.id, r.id)
            if word is None:
                continue
            bond = r.bonds.setdefault(child.id, {"stance": "fond", "affinity": starting_affinity("fond"),
                                                 "note": "", "talks": 0, "last": None})
            bond.setdefault("kin", f"your newborn {word}")
            text = f"{mother.name} gave birth to {child.name}, who is my {word}."
            if not any(m.text == text for m in r.memory.items):
                await r._remember("observation", text, when, 7, source="saw", about=[mother.id, child.id])

    @staticmethod
    def _belly_text(mother: Agent) -> str:
        return f"{mother.name} is visibly with child; her belly is round under her clothes."

    async def _tell_witnesses(self, child: Agent, mother: Agent, when: datetime) -> None:
        """Whoever saw her belly hears the child has come (the town review carries it to the rest). The parents
        and her family have their own memory already; only mother and child are named, never the father."""
        belly = self._belly_text(mother)
        text = f"Word is that {mother.name} has given birth to {child.name}."
        for r in self.agents.values():
            if r.id == child.id or r.id in child.persona.parents or r.persona.dependent or any(child.name in m.text for m in r.memory.items):
                continue
            if any(m.text == belly for m in r.memory.items):
                await r._remember("observation", text, when, 6, source="word", about=[mother.id, child.id])

    async def backfill_family(self) -> None:
        """Children born before parents carried a family tie and a memory of the birth: give them both, once."""
        for child in list(self.agents.values()):
            if len(child.persona.parents) != 2:
                continue
            born = re.search(r"born on (\d{4}-\d{2}-\d{2})", child.persona.background)
            when = datetime.fromisoformat(born.group(1)) + timedelta(hours=4, minutes=2) if born else self.now
            mother_id, father_id = child.persona.parents
            mother, father = self.agents.get(mother_id), self.agents.get(father_id)
            if mother is None or father is None:
                continue
            await self._tell_mothers_family(child, mother, when)
            await self._tell_witnesses(child, mother, when)
            for parent, other, is_mother in ((mother, father, True), (father, mother, False)):
                bond = parent.bonds.setdefault(child.id, {"stance": "fond", "affinity": starting_affinity("fond"),
                                                          "note": "", "talks": 0, "last": None})
                bond.setdefault("kin", f"your newborn {'son' if child.persona.sex == 'male' else 'daughter'}")
                text = romance.birth_memory(is_mother, child.persona, other.persona)
                if not any(m.text == text for m in parent.memory.items):
                    await parent._remember("observation", text, when, 9, source="lived", about=[child.id, other.id])

    def _pregnancy_phases(self, now: datetime) -> None:
        """Pregnancy shows in three steps: she notices, the town can see it, the child comes."""
        for a in self.agents.values():
            a.condition = ""
        for preg in self.pregnancies:
            mother, father = self.agents.get(preg["mother_id"]), self.agents.get(preg["father_id"])
            if preg["resolved"] or mother is None or father is None:
                continue
            gone = (day_key(now) - date.fromisoformat(preg["start_day"])).days
            left = max(0, (date.fromisoformat(preg["due_day"]) - day_key(now)).days)
            if gone >= self.cfg.pregnancy_known_days:
                mother.condition = (f"You are with child, {father.name}'s; it will come in about {left} days. "
                                    "You carry it with you through all you do.")
                if not preg.get("known"):
                    preg["known"] = True
                    self._spawn(mother, mother._remember(
                        "thought", f"I am with child, {father.name}'s; I felt it change me, and it will come in "
                        f"about {left} days.", now, 8), "pregnancy-known")
            if gone >= self.cfg.pregnancy_shows_days and not preg.get("showing"):
                preg["showing"] = True
                self._event("showing", [mother.id], f"{mother.first} is visibly with child.")

    def is_showing(self, agent_id: str) -> bool:
        return any(p["mother_id"] == agent_id and p.get("showing") and not p["resolved"] for p in self.pregnancies)

    async def _give_birth(self, preg: dict, now: datetime) -> None:
        mother, father = self.agents.get(preg["mother_id"]), self.agents.get(preg["father_id"])
        if mother is None or father is None:
            return
        child_persona = romance.generate_child_persona(mother.persona, father.persona, day_key(now), self.rng)
        child_persona = await romance.write_child_persona(self.ctx, child_persona, mother.persona, father.persona,
                                                          day_key(now))
        child = Agent(child_persona, MemoryStream(child_persona.id, self.store, self.embedder), self.world)
        child.history = [{"text": f"Born on {day_key(now).isoformat()} to {mother.name} and {father.name}.",
                          "since": day_key(now).isoformat()}]
        self.agents[child.id] = child
        self.ctx.names[child.id] = child_persona.name
        self.ctx.kin = Kinship({i: x.persona for i, x in self.agents.items()})
        romance.recognize_child(mother, father, child)
        await self._tell_mothers_family(child, mother, now)
        await self._tell_witnesses(child, mother, now)
        await mother._remember("observation", romance.birth_memory(True, child.persona, father.persona), now, 9,
                               source="lived", about=[child.id, father.id])
        await father._remember("observation", romance.birth_memory(False, child.persona, mother.persona), now, 9,
                               source="lived", about=[child.id, mother.id])
        generated = self.store.get("generated_personas", [])
        generated.append(child_persona.model_dump(mode="json"))
        self.store.put("generated_personas", generated)
        from hollowmere.sim.setup import seed_memories  # deferred: setup.py imports this module
        await seed_memories(child, child_persona, self.ctx.names, now)
        if child_persona.species == "goblin" and not mother.persona.is_goblin:
            await self._place_goblin_child(child, mother, father, now)
        self._event("birth", [mother.id, father.id, child.id],
                   f"{mother.first} and {father.first} welcome a child, {child.first}.")

    async def _place_goblin_child(self, child: Agent, mother: Agent, father: Agent, now: datetime) -> None:
        """A human mother chooses whether her goblin child grows up in her home or in its father's den."""
        if not await mother.decide_baby_home(self.ctx, child, father, now):
            return
        child.set_home(father.persona.home, father.persona.home_object, self.world)
        child.x, child.y = self.world.target_tile(child.persona.bed) or (child.x, child.y)
        place = self.world.place_name(father.persona.home)
        await mother._remember("observation", f"I gave {child.name} to {father.name}, to grow up in the den "
                                              f"at {place}.", now, 8)
        await father._remember("observation", f"{mother.name} gave {child.name} into my keeping, in the den at "
                                              f"{place}.", now, 8)
        self._event("move", [mother.id, father.id, child.id], f"{child.first} goes to live in {father.first}'s den.")

    def _agreement_event(self, a: Agent, b: Agent, m: dict) -> None:
        self._event("agreement", [a.id, b.id],
                    f"{a.first} and {b.first} agree to meet at {self.world.place_name(m['place'])} "
                    f"at {m['time']}{'' if m['date'] == day_key(self.now).isoformat() else ' tomorrow'}.",
                    date=m["date"], at=m["time"], place=m["place"])

    def _judged(self, human: Agent, goblin: Agent):
        if not goblin.persona.is_goblin or not human.persona.is_human:
            return None

        def done(regard: str | None) -> None:
            self._judging.discard((goblin.id, human.id))
            if regard is None:
                return
            prev = goblin.regard.get(human.id, {})
            goblin.regard[human.id] = {"step": regard, "at": self.now.isoformat(),
                                       "rebuffs": prev.get("rebuffs", 0) + (regard == "drive off")}
            if regard == "protect":
                goblin.allies.add(human.id)
                human.learn_area(goblin.persona.home, self.world)
                self._event("ally", [human.id, goblin.id], f"{human.first} has chosen to protect {goblin.first}.")
            elif regard == "drive off":
                self._event("spurn", [human.id, goblin.id], f"{human.first} drives {goblin.first} off.")
            elif regard in ("tolerate", "feed") and regard != prev.get("step"):
                self._event("warm", [human.id, goblin.id],
                            f"{human.first} will {'put up with' if regard == 'tolerate' else 'feed'} {goblin.first}.")
        return done

    def _spawn(self, a: Agent, coro, label: str, on_done=None) -> None:
        key = f"{a.id}:{label}"
        if key in self._bg and not self._bg[key].done():
            coro.close()
            return

        async def runner():
            try:
                result = await coro
            except Exception:  # noqa: BLE001 - background work must never kill the loop
                log.exception("%s failed", key)
                if on_done is not None:
                    on_done(None)  # let the callback clear any state waiting on this result
                return
            if on_done is not None:
                on_done(result)

        self._bg[key] = asyncio.create_task(runner())

    async def _guard(self, coro, a: Agent, label: str) -> None:
        try:
            await coro
        except Exception:  # noqa: BLE001
            log.exception("%s: %s failed", a.id, label)
            a.thinking = None

    def _sample(self, now: datetime) -> None:
        self.store.add_samples(now, [
            (a.id, a.x, a.y, a.mode, a.task.place if a.task else None, self.world.area_at(a.x, a.y))
            for a in self.agents.values() if not a.is_away])

    # --------------------------------------------------------------- output

    def _event(self, kind: str, agent_ids: list[str], text: str, feed: bool = True, **extra) -> None:
        """Record an event in the save and, if feed, show it in the viewer."""
        self.store.add_event(self.now, kind, agent_ids, text, {k: v for k, v in extra.items() if k != "transcript"})
        if not feed:
            return
        ev = {"kind": kind, "agents": agent_ids, "text": text, "time": self.now.isoformat(), **extra}
        self.events.append(ev)
        asyncio.get_running_loop().create_task(self._broadcast({"type": "event", "event": ev}))

    async def _broadcast(self, msg: dict) -> None:
        for listener in list(self.listeners):
            try:
                await listener(msg)
            except Exception:  # noqa: BLE001 - a dead socket must not stop the simulation
                self.listeners.discard(listener)

    def state_view(self) -> dict:
        return {
            "time": self.now.isoformat(),
            "paused": self.paused,
            "speed": self.speed,
            "agents": [a.view() | {"bubble": a.bubble} for a in self.agents.values()],
            "objects": {o.id: o.state for o in self.world.objects.values() if o.state != "idle"},
            "llm": self.ctx.llm.stats.as_dict(),
        }

    def agent_detail(self, agent_id: str) -> dict | None:
        a = self.agents.get(agent_id)
        if a is None:
            return None
        p = a.persona
        kin = Kinship({i: x.persona for i, x in self.agents.items()})
        return {
            **a.view(),
            "pregnancy": self._pregnancy_of(agent_id),
            "age": p.age, "traits": p.traits, "states": a.states, "history": a.history, "background": p.background, "currently": a.situation or p.currently,
            "plan": [b.to_dict() | {"place_name": self.world.place_name(b.place)} for b in a.plan],
            "tasks": [t.to_dict() for t in a.tasks],
            "memories": [m.as_dict() for m in a.memory.items[-60:]][::-1],
            "reflections": [m.as_dict() for m in a.memory.items if m.kind == "reflection"][-20:][::-1],
            "commitments": a.commitments,
            "importance_acc": a.importance_acc,
            "species": p.species,
            "drive": a.drive,
            "favor": a.favor | {"god_name": GOD_BY_ID[a.favor["god"]].name} if a.favor and a.favor["god"] in GOD_BY_ID else None,
            "prayers": self.store.prayers_for(agent_id),
            "allies": [self.agents[i].name for i in sorted(a.allies) if i in self.agents],
            "bonds": sorted(
                ({"id": k, "name": self.agents[k].name, "color": self.agents[k].persona.color, **b,
                  "base": p.relationships.get(k, ""),
                  "kin": kin.what_is(k, agent_id),  # what they are to this agent: "mother", "niece", ...
                  "protects": a.persona.is_goblin and k in a.allies,
                  "protected": self.agents[k].persona.is_goblin and a.id in self.agents[k].allies}
                 for k, b in a.bonds.items() if k in self.agents),
                key=lambda b: (-b["talks"], b["last"] or "", b["name"])),
            "seeking": self.agents[self._seeking[a.id]].name if a.id in self._seeking else None,
            "chat_log": [
                {**e, "speaker_name": self.agents[e["speaker"]].name if e["speaker"] in self.agents else None,
                 "with_name": self.agents[e["with"]].name if e["with"] in self.agents else None,
                 "with_color": self.agents[e["with"]].persona.color if e["with"] in self.agents else None}
                for e in self.store.chat_log_for(agent_id)],
        }

    def _pregnancy_of(self, agent_id: str) -> dict | None:
        for preg in self.pregnancies:
            if preg["mother_id"] == agent_id and not preg["resolved"]:
                father = self.agents.get(preg["father_id"])
                due, start = date.fromisoformat(preg["due_day"]), date.fromisoformat(preg["start_day"])
                return {"father": father.name if father else None, "due": preg["due_day"],
                        "days_left": max(0, (due - day_key(self.now)).days), "days_total": (due - start).days}
        return None

    # ---------------------------------------------------------- persistence

    def save(self) -> None:
        self.store.put("clock", self.now.isoformat())
        self.store.put("agents", {a.id: a.snapshot() for a in self.agents.values()})
        self.store.put("town_reviewed", self._town_reviewed.isoformat() if self._town_reviewed else None)
        self.store.put("reported", self._reported.isoformat() if self._reported else None)
        self.store.put("daily_for", self._daily_for.isoformat() if self._daily_for else None)
        self.store.put("daily_pending", self._pending_events)
        self.store.put("pregnancies", self.pregnancies)
        self.store.put("absence_queue", self._queue)
        s = self.ctx.llm.stats
        self.store.put("llm_stats", {
            name: _add_counts(self._llm_before.get(name, {}), getattr(s, attr))
            for name, attr in (("calls", "by_label"), ("repairs", "repairs_by_label"),
                               ("failures", "failures_by_label"))})

    def load(self) -> bool:
        clock = self.store.get("clock")
        snaps = self.store.get("agents")
        if not clock or not snaps:
            return False
        self.now = datetime.fromisoformat(clock)
        self._llm_before = self.store.get("llm_stats", {})
        reviewed = self.store.get("town_reviewed")
        self._town_reviewed = date.fromisoformat(reviewed) if reviewed else None
        reported = self.store.get("reported")
        self._reported = date.fromisoformat(reported) if reported else None
        daily = self.store.get("daily_for")
        self._daily_for = date.fromisoformat(daily) if daily else None
        self._pending_events = self.store.get("daily_pending", [])
        self.pregnancies = self.store.get("pregnancies", [])
        self._queue = self.store.get("absence_queue", [])
        self.store.trim_record(self.now)
        for aid, snap in snaps.items():
            if aid in self.agents:
                self.agents[aid].restore(snap)
                if snap.get("home"):  # moved since the persona file was written
                    self.agents[aid].set_home(snap["home"], snap["home_object"], self.world)
        for goblin in self.agents.values():
            for ally_id in goblin.allies:
                if ally_id in self.agents:
                    self.agents[ally_id].learn_area(goblin.persona.home, self.world)
        return True
