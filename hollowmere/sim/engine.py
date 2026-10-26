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
from hollowmere.agents.agent import Agent, Ctx, Digest, Task, day_key, due_at
from hollowmere.agents.gods import GOD_BY_ID
from hollowmere.agents.persona import RANK, starting_affinity
from hollowmere.agents.schemas import TownState
from hollowmere.config import SimConfig
from hollowmere.llm.client import LLMError
from hollowmere.memory.embed import Embedder
from hollowmere.memory.store import Store
from hollowmere.memory.stream import MemoryStream, repeats
from hollowmere.world.world import World

log = logging.getLogger(__name__)

WALK_TILES = 4
MAX_TURNS = 6  # lines; past this a small model mostly restates itself
CHAT_COOLDOWN = timedelta(hours=3)
REACT_COOLDOWN = timedelta(minutes=50)
ARRANGED_WINDOW = timedelta(minutes=30)  # a meeting two people arranged is "on" this long either side of its hour
ATTENTION = 3  # new observations an agent stores per tick
OBSERVE_NEAR = 8  # tiles: a step that names someone farther off than this cannot have involved them
SAVE_EVERY_TICKS = 30
SAMPLE_EVERY_TICKS = 10  # where everyone is, for `hollowmere report`
FLEE_TILES = 6
HIDE_FOR = timedelta(minutes=10)
HIDING_PLACES = ["goblin_den/nib_nest", "witchwood/bramble_thicket", "riverbank/reeds",
                 "standing_stones/brambles", "graveyard/dead_yew"]
# Event kinds that bear on the town as a whole; raw chat lines and routine plans are left out so the
# town review's prompt stays small and focused on what actually shifted. "prayer" belongs here too by
# that logic, but deliberately stays out: a god's answer is private to the one who prayed, and feeding
# it to the shared narrator would leak it into every agent's town_state - the same boundary enforced by
# never writing a prayer's effect into anyone's bonds (see Agent.pray()).
TOWN_EVENT_KINDS = {"reflection", "agreement", "ally", "spurn", "warm", "sighting", "intent", "chat_end"}

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


class Engine:
    def __init__(self, cfg: SimConfig, world: World, agents: list[Agent], ctx: Ctx, store: Store,
                embedder: Embedder):
        self.cfg = cfg
        self.world = world
        self.agents = {a.id: a for a in agents}
        self.ctx = ctx
        self.store = store
        self.embedder = embedder
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
        await self._schedule(now)
        self._seek(now)
        self._move(now)
        await self._stealth(now)
        await self._converse(now)
        await self._perceive(now)
        self._background(now)
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

    # ---------------------------------------------------------------- stealth

    def _humans(self) -> list[Agent]:
        return [a for a in self.agents.values() if not a.persona.is_goblin]

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
        try:
            out, by_name = await g.goblin_choice(self.ctx, now, entries, partner, g.mode == "hiding")
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
                and now - g.last_chat.get(whom.id, datetime.min) >= CHAT_COOLDOWN):
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
        spot = self._hiding_place(g, threats, far)
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
            if (target is None or not g.awake or g.convo_id is not None or g.hiding_until is not None
                    or g.seek_block != g.current_block(now) or not target.awake
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
            out = await speaker.say(self.ctx, now, listener, purpose, c.transcript)
            line = out.says.strip().strip('"')
            if repeats(line, said):  # own lines, or parroting the other: small models loop
                out = await speaker.say(self.ctx, now, listener, purpose, c.transcript, avoid=line)
                line = out.says.strip().strip('"')
                if repeats(line, said):  # twice: the talk has run dry
                    self._end_conversation(c, "repeat")
                    return
            c.transcript.append((speaker.name, line))
            speaker.bubble, listener.bubble = line, None
            self._event("chat", [speaker.id, listener.id], line, speaker=speaker.id, convo=c.id)
            if out.end or len(c.transcript) >= MAX_TURNS:
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
                if y.persona.is_goblin and not x.persona.is_goblin:
                    self._judging.add((y.id, x.id))
                self._spawn(x, x.digest_chat(self.ctx, self.now, y, list(c.transcript)), f"digest:{c.id}",
                            on_done=self._digested(c, x, y))
        if reason not in ("fled", "never_reached"):  # they actually came face to face, not just tried to
            self._sought[c.a.id] = self._sought[c.b.id] = self.now  # either side's seek should back off now, too
        self._event("chat_end", [c.a.id, c.b.id], f"{c.a.first} and {c.b.first} part ways.",
                    feed=reason not in ("fled", "never_reached"), transcript=c.transcript, convo=c.id,
                    reason=reason, purpose=c.purpose)

    def _visible(self, a: Agent, b: Agent) -> bool:
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
                if b.persona.is_goblin and not a.persona.is_goblin and a.id not in b.allies:
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
                if a.memory.seen_recently(text, now, minutes=90):
                    continue
                stored += 1
                known = b.id in a.persona.relationships
                await a.observe(text, now, 3 if known else 2)
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
        if not b.awake or b.convo_id is not None or a.convo_id is not None:
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
        if a.convo_id is not None or b.convo_id is not None or not b.awake:
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
            idx = a.current_block(now)
            if idx is not None and a.plan[idx].god and a.prayed_block != idx:
                a.prayed_block = idx  # set before spawning: the next tick must not re-fire the same block
                self._spawn(a, a.pray(self.ctx, now, a.plan[idx].god), "pray")
            if a.importance_acc >= self.cfg.reflection_threshold:
                self._spawn(a, a.reflect(self.ctx, now), "reflect")
            elif a.mode == "asleep" and a.plan_date == day_key(now) and a.reviewed_date != a.plan_date \
                    and a.wake is not None and now > a.wake:
                self._spawn(a, a.review_day(self.ctx, now), "review")
        for preg in self.pregnancies:
            if not preg["resolved"] and date.fromisoformat(preg["due_day"]) <= day_key(now):
                preg["resolved"] = True
                mother = self.agents.get(preg["mother_id"])
                if mother is not None:
                    self._spawn(mother, self._give_birth(preg, now),
                               f"birth:{preg['mother_id']}:{preg['due_day']}")
        if self._town_review_due(now) and (self._town_task is None or self._town_task.done()):
            self._town_task = asyncio.create_task(self._run_town_review(now))

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

    def _settle(self, c: Conversation, notes: dict[str, Digest]) -> None:
        """Both sides wrote down the meeting separately; make them agree on one time and place.

        Both notes come from digesting this same conversation `c`, so any entry on one side and any
        entry on the other describe the same real meeting - a large gap between them is one side
        mis-hearing the hour (an "at six" read as dawn by one and dusk by the other), never evidence
        of two different meetings. So every pairing is reconciled, however far apart the guesses land;
        there is nothing to gain by leaving two people with contradictory ideas of their own appointment.
        The one who sought the talk proposed it, so their version wins.

        A meeting only one side noted means the other side's own digest didn't recognise it as theirs
        to keep (small models misjudge "who" easily) - that side still needs the meeting on their
        calendar, not just a bystander's private intention, so it is mirrored onto them mechanically."""
        a, b = c.a, c.b
        mine, theirs = list(notes.get(a.id, Digest()).together), list(notes.get(b.id, Digest()).together)
        unmatched: list[dict] = []
        for m in mine:
            if not theirs:
                unmatched.append(m)
                continue
            match = min(theirs, key=lambda o: abs(due_at(o) - due_at(m)))
            theirs.remove(match)
            if (match["time"], match["place"]) != (m["time"], m["place"]):
                b.move_commitment(match, m["time"], m["place"], self.now)
            self._agreement_event(a, b, m)
        for m in unmatched:
            self._mirror_commitment(b, m, a.id)
        for m in theirs:  # b noted entries a's own digest never matched
            self._mirror_commitment(a, m, b.id)
        if notes.get(a.id, Digest()).made_love and notes.get(b.id, Digest()).made_love:
            self._spawn(a, self._confirm_love(a, b, self.now), f"love:{c.id}")

    async def _confirm_love(self, a: Agent, b: Agent, now: datetime) -> None:
        """Both sides' own digest agreed this conversation included making love: close the gap in
        affinity, mark the pair as having already happened (so future occasions skip the affinity
        gate entirely), and - for a male+female pair with no pregnancy already in flight - conceive."""
        for x, y in ((a, b), (b, a)):
            bond = x.bonds[y.id]
            if RANK[bond["stance"]] < RANK["close"]:
                bond["stance"], bond["affinity"] = "close", starting_affinity("close")
            bond["intimate"] = True
            await x._remember("chat", f"{y.first} and I talked, and it became more - we made love.", now, 7)
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

    async def _give_birth(self, preg: dict, now: datetime) -> None:
        mother, father = self.agents.get(preg["mother_id"]), self.agents.get(preg["father_id"])
        if mother is None or father is None:
            return
        child_persona = romance.generate_child_persona(mother.persona, father.persona, day_key(now), self.rng)
        child = Agent(child_persona, MemoryStream(child_persona.id, self.store, self.embedder), self.world)
        self.agents[child.id] = child
        self.ctx.names[child.id] = child_persona.name
        romance.recognize_child(mother, father, child)
        generated = self.store.get("generated_personas", [])
        generated.append(child_persona.model_dump(mode="json"))
        self.store.put("generated_personas", generated)
        from hollowmere.sim.setup import seed_memories  # deferred: setup.py imports this module
        await seed_memories(child, child_persona, self.ctx.names, now)
        self._event("birth", [mother.id, father.id, child.id],
                   f"{mother.first} and {father.first} welcome a child, {child.first}.")

    def _agreement_event(self, a: Agent, b: Agent, m: dict) -> None:
        self._event("agreement", [a.id, b.id],
                    f"{a.first} and {b.first} agree to meet at {self.world.place_name(m['place'])} "
                    f"at {m['time']}{'' if m['date'] == day_key(self.now).isoformat() else ' tomorrow'}.",
                    date=m["date"], at=m["time"], place=m["place"])

    def _mirror_commitment(self, missing: Agent, entry: dict, proposer_id: str) -> None:
        """Put the other half of a mutual commitment on the person whose own digest missed it."""
        mirrored = {**entry, "with": proposer_id}
        if missing._acceptable(mirrored, self.world, self.now):
            self._agreement_event(self.agents[proposer_id], missing, entry)
            self._spawn(missing, missing._commit(mirrored, self.world, self.now),
                       f"mirror:{proposer_id}:{entry['time']}")

    def _judged(self, human: Agent, goblin: Agent):
        if not goblin.persona.is_goblin or human.persona.is_goblin:
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
            for a in self.agents.values()])

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
        return {
            **a.view(),
            "age": p.age, "traits": p.traits, "background": p.background, "currently": p.currently,
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

    # ---------------------------------------------------------- persistence

    def save(self) -> None:
        self.store.put("clock", self.now.isoformat())
        self.store.put("agents", {a.id: a.snapshot() for a in self.agents.values()})
        self.store.put("town_reviewed", self._town_reviewed.isoformat() if self._town_reviewed else None)
        self.store.put("pregnancies", self.pregnancies)
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
        self.pregnancies = self.store.get("pregnancies", [])
        self.store.trim_record(self.now)
        for aid, snap in snaps.items():
            if aid in self.agents:
                self.agents[aid].restore(snap)
        for goblin in self.agents.values():
            for ally_id in goblin.allies:
                if ally_id in self.agents:
                    self.agents[ally_id].learn_area(goblin.persona.home, self.world)
        return True
