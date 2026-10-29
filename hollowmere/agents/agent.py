"""A generative agent: persona + memory stream + plan + the cognitive steps that maintain them.

Cognitive steps are coroutines. The engine decides when to run them and whether
the world clock waits for them (planning, reacting, speaking) or lets them run
in the background (reflection, digesting a conversation, reviewing the day).
"""

from __future__ import annotations

import logging
import re
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Literal

from hollowmere.agents import prompts, romance
from hollowmere.agents.gods import GOD_BY_ID
from hollowmere.agents.kinship import Kinship
from hollowmere.agents.persona import Persona, affinity_label, starting_affinity, update_affinity
from hollowmere.agents.schemas import (
    BabyHome,
    LifeFacts,
    TIME_OF_DAY, DayReview, DivineAnswer, FocalQuestions, Insights, Reaction, Utterance, utterance_schema,
    agreement_schema, chat_digest_schema, day_plan_schema, decomposition_schema, goblin_choice_schema,
    leaving_schema, moving_schema,
)
from hollowmere.llm.client import LLM, LLMError
from hollowmere.memory.stream import MemoryStream, repeats
from hollowmere.world.world import World

log = logging.getLogger(__name__)

DAY_START = time(4, 0)  # plans are made at 04:00; times before that belong to the previous day
SLEEP_RE = re.compile(r"\b(sleep|sleeps|asleep|goes to bed|to bed|retires)\b", re.IGNORECASE)
WORD_RE = re.compile(r"[a-z0-9]+")
DECOMPOSE_MIN_MINUTES = 25
TOPICS_KEPT = 6
PROMISE_GAP = timedelta(minutes=60)  # one promise per hour: a second one there is a duplicate or a clash
PROMISE_LEAD = timedelta(minutes=10)  # a promise must be at least this far ahead


def due_at(c: dict) -> datetime:
    return parse_hhmm(date.fromisoformat(c["date"]), c["time"])


IMMEDIATE_RE = re.compile(r"(right )?now\b|at once\b|right away\b|straightaway\b|immediately\b|instantly\b")


def settle_near_now(c: dict, now: datetime) -> dict:
    """A promise to go "now" or "tonight" often lands on the very minute of the talk, which _acceptable
    would call already past: set off just ahead instead. Anything further behind is left to be refused."""
    if now - PROMISE_LEAD <= due_at(c) < now + PROMISE_LEAD:
        soon = now + PROMISE_LEAD
        c["date"], c["time"] = day_key(soon).isoformat(), soon.strftime("%H:%M")
    return c


def roll_past_hour(c: dict, now: datetime) -> dict:
    """The schema says "today if that hour has not yet passed, otherwise tomorrow", but a small model still
    writes today for "before the sun crests" at 08:47. An hour gone by more than the near-now window means
    tomorrow's."""
    if due_at(c) < now - PROMISE_LEAD:
        c["date"] = (date.fromisoformat(c["date"]) + timedelta(days=1)).isoformat()
    return c


def spoken(when: str, transcript: list[tuple[str, str]]) -> bool:
    """Whether the time words were really said: models fill in "12:00" for "I will bless the well".
    "Now" is the exception: "come, let us go" says it without the word."""
    if IMMEDIATE_RE.match(when.strip().lower()):
        return True
    words = set(WORD_RE.findall(when.lower())) - {"at", "the", "by", "in", "on", "a", "an", "of"}
    said = set(WORD_RE.findall(" ".join(text for _, text in transcript).lower()))
    return bool(words) and words <= said


@dataclass
class Digest:
    """What digesting a conversation produced that the engine acts on."""
    regard: str | None = None  # a human who met the goblin: what they now mean to do about it (GoblinRegard)
    made_love: bool = False  # this agent's own judgment of whether this conversation included making love
    arrested: str = "no"  # this agent's own judgment: "i_locked_them_up", "they_locked_me_up", or "no"

Mode = Literal["asleep", "walking", "acting", "talking", "waiting", "hiding", "sex", "away"]


@dataclass
class Block:
    start: datetime
    end: datetime
    activity: str
    place: str
    god: str | None = None  # which god this block is a prayer to, if any

    def to_dict(self) -> dict:
        return {"start": self.start.isoformat(), "end": self.end.isoformat(),
                "activity": self.activity, "place": self.place, "god": self.god}

    @classmethod
    def from_dict(cls, d: dict) -> Block:
        return cls(datetime.fromisoformat(d["start"]), datetime.fromisoformat(d["end"]), d["activity"],
                   d["place"], d.get("god"))


@dataclass
class Task:
    start: datetime
    end: datetime
    activity: str
    place: str
    emoji: str

    def to_dict(self) -> dict:
        return {"start": self.start.isoformat(), "end": self.end.isoformat(), "activity": self.activity,
                "place": self.place, "emoji": self.emoji}

    @classmethod
    def from_dict(cls, d: dict) -> Task:
        return cls(datetime.fromisoformat(d["start"]), datetime.fromisoformat(d["end"]), d["activity"],
                   d["place"], d["emoji"])


@dataclass
class Ctx:
    """What a cognitive step needs from the outside world."""
    llm: LLM
    world: World
    names: dict[str, str]  # agent id -> full name
    known_as: dict[str, str] = field(default_factory=dict)  # agent id -> what the whole town knows of them
    kin: Kinship | None = None  # who is family to whom; the engine keeps it current as children are born


def day_key(now: datetime) -> date:
    return (now - timedelta(hours=DAY_START.hour, minutes=DAY_START.minute)).date()


def recognised(text: str, god_name: str, clarity: str) -> str:
    """A god's answer as the mortal remembers it: only as sure of the sender as the god let them be."""
    if clarity == "doubt":
        return f"{text} Was it only chance? Or did {god_name} hear me?"
    if clarity == "suspected":
        return f"{text} I think {god_name} answered me, though I cannot be sure."
    if clarity == "certain":
        return f"{god_name} answered my prayer: {text}"
    return text


def parse_hhmm(day: date, hhmm: str) -> datetime:
    h, m = (int(x) for x in hhmm.split(":"))
    t = datetime.combine(day, time(h, m))
    return t + timedelta(days=1) if time(h, m) < DAY_START else t


def is_sleep(activity: str) -> bool:
    return bool(SLEEP_RE.search(activity))


def normalise_plan(day: date, wake: datetime, raw: list[tuple[str, str, str, str, str | None]],
                   bed: str) -> list[Block]:
    """Turn LLM blocks into a gap-free, non-overlapping schedule from wake-up to bedtime."""
    blocks: list[Block] = []
    for start_s, end_s, activity, place, god in raw:
        start, end = parse_hhmm(day, start_s), parse_hhmm(day, end_s)
        if end <= start:
            end = start + timedelta(hours=1)
        blocks.append(Block(start, end, activity.strip(), place, god))
    blocks.sort(key=lambda b: b.start)
    out: list[Block] = []
    cursor = wake
    for b in blocks:
        b.start = max(b.start, cursor)
        if b.end <= b.start:
            continue
        if out and b.start > out[-1].end:  # close gaps by stretching the previous block
            out[-1].end = b.start
        elif not out and b.start > wake:
            b.start = wake
        out.append(b)
        cursor = b.end
    limit = datetime.combine(day + timedelta(days=1), DAY_START) - timedelta(minutes=30)
    out = [b for b in out if b.start < limit]
    for b in out:
        b.end = min(b.end, limit)
    if not out or not is_sleep(out[-1].activity):
        last_end = out[-1].end if out else wake
        out.append(Block(last_end, limit, "goes to sleep", bed))
    return out


async def agreed_meetings(ctx: Ctx, a: "Agent", b: "Agent", now: datetime,
                          transcript: list[tuple[str, str]]) -> list[dict]:
    """What `a` and `b` agreed to do together in this one talk, decided once for both of them.

    Each side digesting the talk on its own used to read a vague hour differently and then had to be reconciled;
    one shared reading cannot disagree with itself. Entries are {date, time, activity, place} for the engine to
    put on both calendars. Only places both of them go to are offered, so a goblin is never sent into the tavern
    and a visit to either one's home is possible."""
    places = tuple(p for p in dict.fromkeys(a.place_ids + b.place_ids)
                   if a.can_go(p, ctx.world) and b.can_go(p, ctx.world))
    try:
        out = await ctx.llm.structured(
            prompts.agreement_system_prompt(),
            prompts.agreement(a.first, b.first, now, transcript, {p: ctx.world.place_name(p) for p in places}),
            agreement_schema(places), label="agree", temperature=0.2)
    except LLMError as e:
        log.warning("%s and %s: agreement failed (%s)", a.id, b.id, e)
        return []
    today = day_key(now)
    found = []
    for m in out.meetings:
        if not m.both_agreed or not spoken(m.when_said, transcript):
            continue  # one-sided, or a time nobody named
        when = today if m.day == "today" else today + timedelta(days=1)
        entry = {"date": when.isoformat(), "time": TIME_OF_DAY[m.time_of_day] or now.strftime("%H:%M"),
                 "activity": m.activity.strip(), "place": m.place}
        found.append(settle_near_now(roll_past_hour(entry, now) if m.day == "today" else entry, now))
    return found


async def agreed_departure(ctx: Ctx, a: "Agent", b: "Agent", now: datetime,
                           transcript: list[tuple[str, str]]) -> list["Agent"]:
    """Who, if anyone, settled in this one talk on leaving Hollowmere, decided once for both of them.

    Two digests each had to say so and almost never both did: Isolde and Silas agreed to flee in seven talks
    and set out in none. One shared reading cannot disagree with itself. The engine carries it out."""
    try:
        out = await ctx.llm.structured(
            prompts.leaving_system_prompt(), prompts.leaving(a.first, b.first, now, transcript),
            leaving_schema(a.first, b.first), label="leave", temperature=0.2)
    except LLMError as e:
        log.warning("%s and %s: leaving failed (%s)", a.id, b.id, e)
        return []
    for d in out.departures:
        if d.settled and spoken(d.words, transcript):  # the words are really in the talk
            return [a, b] if d.who == "both" else [a if d.who == a.first else b]
    return []


async def agreed_move(ctx: Ctx, a: "Agent", b: "Agent", now: datetime,
                      transcript: list[tuple[str, str]]) -> tuple["Agent", "Agent"] | None:
    """(mover, host) if one of the two settled in this talk on moving into the other's home, else None."""
    try:
        out = await ctx.llm.structured(
            prompts.leaving_system_prompt(), prompts.moving(a.first, b.first, now, transcript),
            moving_schema(a.first, b.first), label="move", temperature=0.2)
    except LLMError as e:
        log.warning("%s and %s: moving failed (%s)", a.id, b.id, e)
        return None
    for m in out.moves:
        if m.settled and spoken(m.words, transcript):  # the words are really in the talk
            return (a, b) if m.who == a.first else (b, a)
    return None


MAX_STATES = 3
MAX_HISTORY = 6  # lasting life facts per person
LIFE_FACT_IMPORTANCE = 8  # a day holds a memory at least this weighty before lasting facts are even asked for

def restates_background(fact: str, persona: Persona) -> bool:
    """Whether a fact only says again what the persona's own description does (a trade, a rank, a family tie)."""
    own = set(WORD_RE.findall(persona.name.lower()))
    words = {w for w in WORD_RE.findall(fact.lower()) if len(w) > 3} - own
    if len(words) < 2:
        return False
    said = set(WORD_RE.findall(persona.background.lower()))
    return len(words & said) / len(words) >= 0.8


CELL = "guard_cell/cell_pallet"  # where anyone locked up sits


class Agent:
    def __init__(self, persona: Persona, memory: MemoryStream, world: World):
        self.persona = persona
        self.memory = memory
        self.id = persona.id
        self.name = persona.name
        own = {persona.home, persona.work}
        # The goblin cannot walk into the tavern or the bakery; plans that send him indoors only end in a bolt.
        # A baby stays where it is carried: its mother's home, never the streets.
        self.known_areas = [a.id for a in world.areas.values()
                            if a.id in own or (a.public and not persona.infant and not (persona.is_goblin and a.indoor))]
        self.place_ids = self._places_in(world)
        self.x, self.y = world.target_tile(persona.start) or (0, 0)
        self.path: list[tuple[int, int]] = []
        self.steps: list[tuple[int, int]] = []  # tiles walked during the last tick, for animation
        self.plan: list[Block] = []
        self.plan_date: date | None = None
        self.wake: datetime | None = None
        self.tasks: list[Task] = []
        self.tasks_block: int = -1
        self.task: Task | None = None
        self.mode: Mode = "asleep"
        self.thinking: str | None = None
        self.convo_id: str | None = None
        self.last_chat: dict[str, datetime] = {}
        self.last_react: dict[str, datetime] = {}
        self.importance_acc = 0
        self.reviewed_date: date | None = None
        self.commitments: list[dict] = []  # {"date": iso, "time": "HH:MM", "activity", "place"}
        self.bubble: str | None = None
        self.announced: date | None = None  # last plan_date shown in the event feed
        self.allies: set[str] = set()  # goblin only: humans who have chosen to protect it
        # goblin only: human -> {"step": GoblinRegard, "at": iso time, "rebuffs": times they drove him off}
        self.regard: dict[str, dict] = {}
        self.hiding_until: datetime | None = None
        self.topics: deque[str] = deque(maxlen=TOPICS_KEPT)  # what recent conversations were about
        self.new_reflections: list[str] = []  # not yet shown in the event feed
        # In the agent's own words, what it wants for itself beyond today's tasks; written and
        # rewritten freely by review_day(), never interpreted by any code.
        self.drive: str = ""
        # Passing states of mind: {"text", "since": iso date or None until the first review dates it}.
        # Only the day review ends or adds one; there is no fixed span.
        self.states: list[dict] = [{"text": s, "since": None} for s in persona.states]
        self.history: list[dict] = []  # lasting facts, {"text", "since"}: what happened to them that stays true
        self.condition: str = ""  # what the engine says is true of them now (with child, due in N days); not persisted
        self.situation: str = ""  # carried forward by day reviews; empty until then (the persona's `currently` stands)
        # The town's current situation (plague, harvest, the Lord...), rewritten periodically by a
        # town-wide review and kept identical across every agent; empty until the first review runs,
        # in which case prompts.system_prompt() falls back to a seed sentence.
        self.town_state: str = ""
        # How this agent sees each person it knows: seeded from the persona, then nudged after every
        # talk by update_affinity() rather than overwritten raw. "affinity" is the continuous score
        # that carries the history; "stance" is just that score's label, for prompts and the viewer -
        # no code branches on the label itself.
        self.bonds: dict[str, dict] = {
            k: {"stance": persona.stances.get(k, "neutral"), "affinity": starting_affinity(persona.stances.get(k, "neutral")),
                "note": "", "talks": 0, "last": None}
            for k in persona.relationships.keys() | persona.stances.keys()}
        self.seek: str | None = None  # whom the current plan block is spent going to find, if anyone
        self.seek_block = -1
        self.favor: dict | None = None  # {"god", "text", "expires"}: a buff from the last god to answer -
                                        # one at a time, always replaced by whichever god answers next
        self.pending_dream: dict | None = None  # {"text", "importance"}, delivered at the next plan_day()
        self.prayed_block: int = -1  # plan index already resolved this day, mirrors tasks_block/seek_block
        self.new_prayers: list[dict] = []  # not yet shown in the event feed
        # {"kind": "away" | "jailed", "since": iso, "until": iso, "reason": str}: out of town, or in the cell.
        # An infant's absence is its mother's, set and cleared with hers.
        self.absence: dict | None = None

    def _places_in(self, world: World) -> tuple[str, ...]:
        return tuple(pid for area in self.known_areas
                     for pid in [area] + [o.place_id for o in world.objects_in(area)])

    def set_home(self, home: str, home_object: str, world: World) -> None:
        """Live and sleep somewhere else from now on: the old roof is no longer theirs to name, the new one is."""
        old_bed = self.persona.bed
        old_home = self.persona.home
        self.persona.home, self.persona.home_object = home, home_object
        if old_home != home:
            area = world.areas.get(old_home)
            if area is not None and not area.public and old_home != self.persona.work:
                self.known_areas = [x for x in self.known_areas if x != old_home]
        if home in world.areas and home not in self.known_areas:
            self.known_areas.append(home)
        self.place_ids = self._places_in(world)
        bed = self.persona.bed
        for block in self.plan:
            if block.place == old_bed:
                block.place = bed
        for task in [*self.tasks, *([self.task] if self.task else [])]:
            if task.place == old_bed:
                task.place = bed

    def looks_now(self) -> str:
        """What another person sees of them: their usual looks, and a changed bearing while a favor lingers.
        The god is never named - the other only sees the change."""
        looks = self.persona.appearance
        if not self.favor:
            return looks
        return f"{looks} Today they carry themselves differently: {self.favor['text'].rstrip('.')}.".strip()

    def learn_area(self, area_id: str, world: World) -> None:
        """A place this person was not free to name before, e.g. the goblin's burrow once they swore to protect him."""
        if area_id in world.areas and area_id not in self.known_areas:
            self.known_areas.append(area_id)
            self.place_ids = self._places_in(world)

    def knows(self, other: "Agent") -> bool:
        return other.id in self.bonds or self.id in other.allies

    def note_topic(self, other_first: str, topic: str) -> None:
        self.topics.append(f"with {other_first}: {topic[:140]}")

    # ------------------------------------------------------------------- views

    @property
    def first(self) -> str:
        return self.persona.first_name

    @property
    def is_away(self) -> bool:
        return self.absence is not None and self.absence["kind"] == "away"

    @property
    def is_jailed(self) -> bool:
        return self.absence is not None and self.absence["kind"] == "jailed"

    @property
    def awake(self) -> bool:
        return self.mode not in ("asleep", "away")

    def activity_phrase(self, world: World, detail: bool = True) -> str:
        """What someone looks to be doing; detail=False keeps only where, not the planned step."""
        if self.mode == "asleep":
            return "is asleep"
        if self.mode == "away":
            return "is away from town"
        if self.mode == "talking":
            return "is deep in conversation"
        if self.mode == "hiding":
            return "is hiding"
        if self.mode == "sex":
            return "is having sex"
        if self.task is None:
            return "is standing about"
        where = world.place_name(self.task.place)
        head = f"{'is on the way to' if self.mode == 'walking' else 'is at'} {where}"
        return f"{head}, busy: {self.task.activity}" if detail else head

    def current_block(self, now: datetime) -> int | None:
        for i, b in enumerate(self.plan):
            if b.start <= now < b.end:
                return i
        return None

    def view(self) -> dict:
        return {
            "id": self.id, "name": self.name, "first": self.first, "color": self.persona.color,
            "x": self.x, "y": self.y, "steps": self.steps, "mode": self.mode, "absence": self.absence["kind"] if self.absence else None,
            "thinking": self.thinking,
            "activity": self.task.activity if self.task else None,
            "emoji": "💤" if self.mode == "asleep" else (self.task.emoji if self.task else ""),
            "place": self.task.place if self.task else None, "convo": self.convo_id,
            "species": self.persona.species, "allies": sorted(self.allies),
        }

    # ---------------------------------------------------------------- schedule

    def schedule_job(self, ctx: Ctx, now: datetime):
        """Advance the agent's schedule. Returns a coroutine if the LLM is needed before it can act."""
        today = day_key(now)
        if self.is_jailed:  # nothing to plan: the day is the cell
            if self.task is None or self.task.place != CELL or now >= self.task.end:
                self._set_task(ctx.world, Task(now, now + timedelta(hours=1), "sits locked in the cell", CELL, ""))
            return None
        if self.plan_date != today:
            return self.plan_day(ctx, now)
        idx = self.current_block(now)
        if idx is None:
            self._set_task(ctx.world, Task(now, now + timedelta(hours=1), "sleeps", self.persona.bed, "💤"))
            return None
        if idx != self.tasks_block:
            block = self.plan[idx]
            minutes = (block.end - max(block.start, now)).total_seconds() / 60
            if minutes < DECOMPOSE_MIN_MINUTES or is_sleep(block.activity):
                self.tasks = [Task(block.start, block.end, block.activity, block.place,
                                   "💤" if is_sleep(block.activity) else "")]
                self.tasks_block = idx
            else:
                return self.decompose(ctx, now, idx)
        current = next((t for t in self.tasks if t.start <= now < t.end), None) or self.tasks[-1]
        if current is not self.task:
            self._set_task(ctx.world, current)
        return None

    def _set_task(self, world: World, task: Task) -> None:
        if self.task is not None and self.task.activity == task.activity and self.task.place == task.place:
            self.task = task
            return
        self.task = task
        salt = sum(map(ord, self.id))
        target = world.target_tile(task.place, salt) or (self.x, self.y)
        self.path = world.find_path((self.x, self.y), target)
        bed = task.place == self.persona.bed and is_sleep(task.activity)
        self.mode = "walking" if self.path else ("asleep" if bed else "acting")

    def arrive(self) -> None:
        if self.task and self.task.place == self.persona.bed and is_sleep(self.task.activity):
            self.mode = "asleep"
        else:
            self.mode = "acting"

    # ------------------------------------------------------------- cognition

    async def plan_day(self, ctx: Ctx, now: datetime) -> None:
        today = day_key(now)
        self.thinking = "planning the day"
        self._drop_old_commitments(now)
        if self.pending_dream:
            await self.observe(self.pending_dream["text"], now, self.pending_dream["importance"])
            self.pending_dream = None
        try:
            who_matters = [ctx.names.get(rid, rid) for rid in self.persona.relationships]
            kin = [f"{ctx.names.get(k, k)}, {b['kin']}" for k, b in self.bonds.items() if b.get("kin")]
            mems = await self.memory.retrieve(
                [f"{self.first}'s plans and intentions", f"what {self.first} must do today",
                 f"{self.situation or self.persona.currently}", *kin,
                 *(f"where and when {self.first} has seen {name}, their habits and comings and goings"
                   for name in who_matters)], now, k=12)
            due = [c for c in self.commitments if c["date"] == today.isoformat()]
            promises = [f"{c['time']} — {c['activity']} at {c['place']}" for c in due]
            schema = day_plan_schema(self.place_ids)
            out = await ctx.llm.structured(
                prompts.system_prompt(self),
                prompts.plan_day(self, ctx.world, ctx.names, now, mems, promises, known_as=ctx.known_as),
                schema, label="plan_day")
            wake = parse_hhmm(today, out.wake_up)
            raw = [(b.start, b.end, b.activity, b.place, None if b.pray_to == "none" else b.pray_to)
                   for b in out.blocks]
            self.plan = normalise_plan(today, wake, raw, self.persona.bed)
            for c in due:
                self._insert_commitment(c, today)
            self.wake = self.plan[0].start
            await self.memory.add("thought", out.morning_thought, now, 3)
            summary = "; ".join(f"{b.start:%H:%M} {b.activity}" for b in self.plan)
            await self.memory.add("plan", f"My plan for {today:%A}: {summary}", now, 4)
        except LLMError as e:
            log.warning("%s: planning failed (%s); using habits", self.id, e)
            self.plan = self._fallback_plan(today)
        self.plan_date = today
        self.tasks_block = -1
        self.prayed_block = -1
        self.thinking = None

    def _fallback_plan(self, day: date) -> list[Block]:
        p = self.persona
        at = lambda h, m=0: datetime.combine(day, time(h, m))  # noqa: E731
        if p.is_goblin:
            return [
                Block(at(12), at(16), "tends the shrine and the fire", p.work),
                Block(at(16), at(20), "skulks at the edge of the Witchwood", "witchwood"),
                Block(at(20), at(23, 59), "creeps into town to steal scraps", "graveyard"),
                Block(at(23, 59), at(23, 59) + timedelta(hours=3, minutes=30), "goes to sleep", p.bed),
            ]
        return [
            Block(at(6), at(7), "rises and breaks fast", p.bed),
            Block(at(7), at(12), "works", p.work),
            Block(at(12), at(13), "takes a meal at the tavern", "tavern"),
            Block(at(13), at(18), "works", p.work),
            Block(at(18), at(21), "drinks at the tavern", "tavern"),
            Block(at(21), at(21) + timedelta(hours=6, minutes=30), "goes to sleep", p.bed),
        ]

    def _insert_commitment(self, c: dict, today: date) -> None:
        """Make room for the commitment without shrinking anyone's day: whatever it displaces -
        including sleep - keeps its original length and simply happens later instead of being cut short."""
        start = parse_hhmm(today, c["time"])
        end = start + timedelta(hours=1)
        area = c["place"].split("/")[0]
        if any(b.start <= start + PROMISE_LEAD and b.end > start and b.place.split("/")[0] == area
               for b in self.plan):
            return  # the plan already has them there at that time (an hour later is not "on time")
        shift = end - start
        new: list[Block] = []
        for b in self.plan:
            if b.end <= start:
                new.append(b)
            elif b.start < start:
                new.append(Block(b.start, start, b.activity, b.place, b.god))
                new.append(Block(end, end + (b.end - start), b.activity, b.place, b.god))
            else:
                new.append(Block(b.start + shift, b.end + shift, b.activity, b.place, b.god))
        new.append(Block(start, end, c["activity"], c["place"]))
        self.plan = sorted(new, key=lambda b: b.start)

    async def decompose(self, ctx: Ctx, now: datetime, idx: int) -> None:
        block = self.plan[idx]
        start = max(block.start, now)
        self.thinking = f"thinking about: {block.activity}"
        area = block.place.split("/")[0]
        allowed = tuple(p for p in self.place_ids if p.split("/")[0] == area) or (block.place,)
        seekable = {ctx.names[k]: k for k in sorted(self.bonds.keys() | self.allies) if k in ctx.names}
        self.seek, self.seek_block = None, idx
        try:
            mems = await self.memory.retrieve([block.activity], now, k=6)
            out = await ctx.llm.structured(
                prompts.system_prompt(self),
                prompts.decompose(self, ctx.world, ctx.names, now, block.activity, start, block.end,
                                  ctx.world.place_name(area), mems),
                decomposition_schema(allowed, tuple(seekable)), label="decompose")
            self.seek = seekable.get(getattr(out, "seek", "nobody"))
            total = sum(s.minutes for s in out.steps)
            span = (block.end - start).total_seconds()
            tasks, cursor = [], start
            for s in out.steps:
                length = timedelta(seconds=span * s.minutes / total)
                tasks.append(Task(cursor, cursor + length, s.activity, s.place, s.emoji))
                cursor += length
            tasks[-1].end = block.end
            self.tasks = tasks
        except LLMError as e:
            log.warning("%s: decomposition failed (%s)", self.id, e)
            self.tasks = [Task(start, block.end, block.activity, block.place, "")]
        self.tasks_block = idx
        self.thinking = None
        self.task = None  # force _set_task on the next schedule pass

    async def react(self, ctx: Ctx, now: datetime, other: Agent, arranged: str = "") -> tuple[bool, str]:
        """Notice another agent. Returns (wants_to_talk, purpose). `arranged` is what the two agreed to meet for."""
        self.last_react[other.id] = now
        self.thinking = f"noticing {other.first}"
        try:
            mems = await self.memory.retrieve(
                [other.name, f"{other.first} {other.task.activity if other.task else ''}",
                 f"warmth, friendship or love between {self.first} and {other.first}"], now, k=8)
            place = ctx.world.place_name(self.task.place) if self.task else "the street"
            goblin = other.persona.is_goblin and not self.persona.is_goblin
            aside = prompts.goblin_aside(self, other, now) if goblin else ""
            other_name = other.name if not goblin or self.knows(other) else "a goblin"  # looks follow from the persona
            out = await ctx.llm.structured(
                prompts.system_prompt(self),
                prompts.react(self, ctx.names, now, place, self.task.activity if self.task else "standing about",
                              other_name, other.activity_phrase(ctx.world), mems, aside,
                              topics=list(self.topics), other_looks=other.looks_now(), arranged=arranged),
                Reaction, label="react")
        except LLMError as e:
            log.warning("%s: reaction failed (%s)", self.id, e)
            return False, ""
        finally:
            self.thinking = None
        await self._remember("thought", out.thought, now, out.importance)
        return out.talk, out.reason

    async def goblin_choice(self, ctx: Ctx, now: datetime, seen: list[tuple["Agent", float, bool]],
                            talking_with: "Agent | None", hidden: bool, cover: str | None = None):
        """The goblin sees people who could give him away. seen: (person, distance, they have seen him).

        Returns the GoblinChoice and a map from the names offered back to the agents."""
        by_name = {h.name: h for h, _, _ in seen}
        self.thinking = "watching from the shadows"
        try:
            mems = await self.memory.retrieve([h.name for h in by_name.values()], now, k=8)
            place = ctx.world.place_name(self.task.place) if self.task else "the edge of the road"
            out = await ctx.llm.structured(
                prompts.system_prompt(self),
                prompts.goblin_choice(self, ctx.world, ctx.names, now, place, seen, talking_with, hidden, mems, cover),
                goblin_choice_schema(tuple(by_name)), label="goblin_choice")
        finally:
            self.thinking = None
        await self._remember("thought", out.thought, now, 4)
        return out, by_name

    async def say(self, ctx: Ctx, now: datetime, other: Agent, purpose: str,
                  transcript: list[tuple[str, str]], avoid: str = "", invited: bool = False) -> Utterance:
        self.thinking = None
        last = transcript[-1][1] if transcript else purpose
        mems = await self.memory.retrieve([other.name, last or other.name], now, k=8)
        place = ctx.world.place_name(self.task.place) if self.task else "the street"
        other_doing = other.task.activity if other.task else "standing about"
        aside = prompts.goblin_aside(self, other, now) if other.persona.is_goblin and not self.persona.is_goblin else ""
        romance_eligible = romance.eligible_for_love(self.persona.sex, other.persona.sex, self.bonds.get(other.id),
                                                  kin=ctx.kin is not None and ctx.kin.related(self.id, other.id))
        return await ctx.llm.structured(
            prompts.system_prompt(self),
            prompts.converse(self, ctx.names, now, place, other.name, other_doing, purpose, transcript, mems, aside,
                             town_places=[ctx.world.areas[a].name for a in self.known_areas],
                             topics=list(self.topics), known_as=ctx.known_as, avoid=avoid,
                             other_looks=other.looks_now(), romance_eligible=romance_eligible,
                             invited=invited),
            utterance_schema(romance_eligible), label="converse", temperature=0.85)

    async def digest_chat(self, ctx: Ctx, now: datetime, other: Agent,
                          transcript: list[tuple[str, str]]) -> Digest:
        """Remember a conversation and any agreement made in it."""
        judging_goblin = other.persona.is_goblin and not self.persona.is_goblin and self.id not in other.allies
        aside = prompts.goblin_aside(self, other, now) if judging_goblin else ""
        owed = [f"{c['time']} — {c['activity']} at {c['place']}" for c in self.commitments]
        romance_eligible = romance.eligible_for_love(self.persona.sex, other.persona.sex, self.bonds.get(other.id),
                                                  kin=ctx.kin is not None and ctx.kin.related(self.id, other.id))
        # a meeting is two people's: only offer places the other one will go to as well
        places = tuple(p for p in self.place_ids if other.can_go(p, ctx.world))
        try:
            out = await ctx.llm.structured(
                prompts.system_prompt(self),
                prompts.chat_digest(self, ctx.names, now, other.name, transcript, aside, existing=owed),
                chat_digest_schema(places, judging_goblin, romance_eligible), label="digest")
        except LLMError as e:
            log.warning("%s: digest failed (%s)", self.id, e)
            await self._remember("chat", f"I spoke with {other.name}.", now, 3)
            return Digest()
        await self._remember("chat", out.summary, now, out.importance)
        bond = self.bonds.get(other.id, {"talks": 0, "affinity": starting_affinity("neutral")})
        affinity = update_affinity(bond.get("affinity", starting_affinity(bond.get("stance", "neutral"))),
                                   out.feeling, out.importance)
        self.bonds[other.id] = {"stance": affinity_label(affinity), "affinity": affinity,
                                "note": out.view_of_them.strip(),
                                "talks": bond["talks"] + 1, "last": now.isoformat(),
                                **({"kin": bond["kin"]} if bond.get("kin") else {})}
        today = day_key(now)
        result = Digest(made_love=romance_eligible and getattr(out, "made_love", False),
                        arrested=getattr(out, "arrested", "no"))
        for c in out.commitments:
            if c.who != "me" or not c.accepted or not spoken(c.when_said, transcript):
                continue  # their errand, a shared meeting (agreed_meetings), an unanswered demand, or a time nobody named
            when = today if c.day == "today" else today + timedelta(days=1)
            entry = {"date": when.isoformat(), "time": c.time, "activity": c.activity, "place": c.place,
                     "with": None}
            entry = settle_near_now(roll_past_hour(entry, now) if c.day == "today" else entry, now)
            if not self._acceptable(entry, ctx.world, now):
                continue
            await self._commit(entry, ctx.world, now)
        if judging_goblin:
            result.regard = out.toward_goblin
            if out.toward_goblin == "protect":
                await self.memory.add("thought", f"I have decided to keep the goblin {other.name}'s secret and "
                                                 "protect him.", now, 8)
        return result

    def can_go(self, place: str, world: World) -> bool:
        """Whether this person goes to `place` at all: the goblin never walks indoors, except home and work."""
        area = world.areas.get(place.split("/")[0])
        return not (self.persona.is_goblin and area is not None and area.indoor
                    and area.id not in (self.persona.home, self.persona.work))

    def _acceptable(self, c: dict, world: World, now: datetime) -> bool:
        """Small models list routine, past events and contradictions as promises; keep only real ones."""
        try:
            at = due_at(c)
        except ValueError:
            return False
        if at < now + PROMISE_LEAD:
            return False  # already past: a record of the talk, not a plan
        if self.first.lower() in c["activity"].lower().split():
            return False  # "report to Brannoc" in Brannoc's own list: the other person's side
        if not self.can_go(c["place"], world):
            return False  # the goblin does not walk into the tavern
        return not any(abs(due_at(o) - at) < PROMISE_GAP for o in self.commitments if not o.get("met"))

    async def _commit(self, c: dict, world: World, now: datetime) -> None:
        self.commitments.append(c)
        day = "today" if date.fromisoformat(c["date"]) == day_key(now) else "tomorrow"
        await self.memory.add("plan", f"I mean to {c['activity']} at {world.place_name(c['place'])} "
                                      f"{day} at {c['time']}.", now, 5)
        self._schedule_commitment(c, now)

    def _schedule_commitment(self, c: dict, now: datetime) -> None:
        today = day_key(now)
        if c["date"] == today.isoformat() and self.plan_date == today and due_at(c) > now:
            self._insert_commitment(c, today)
            self.tasks_block = -1

    def _drop_old_commitments(self, now: datetime) -> None:
        self.commitments = [c for c in self.commitments if due_at(c) > now - timedelta(hours=2)]

    async def reflect(self, ctx: Ctx, now: datetime) -> None:
        self.importance_acc = 0
        recent = self.memory.recent(60, kinds={"observation", "chat"})  # own conclusions are not evidence
        if len(recent) < 5:
            return
        try:
            qs = await ctx.llm.structured(
                prompts.system_prompt(self), prompts.focal_questions(self, ctx.names, now, recent[-40:]),
                FocalQuestions, label="reflect_questions")
            mems = await self.memory.retrieve(qs.questions, now, k=8, exclude_kinds={"reflection", "thought", "plan"})
            out = await ctx.llm.structured(
                prompts.system_prompt(self), prompts.insights(self, ctx.names, now, qs.questions, mems),
                Insights, label="reflect_insights")
        except LLMError as e:
            log.warning("%s: reflection failed (%s)", self.id, e)
            return
        valid = {m.id for m in mems}
        for ins in out.insights:
            await self._add_reflection(ins.text, now, ins.importance, [i for i in ins.evidence if i in valid])

    async def _add_reflection(self, text: str, now: datetime, importance: int, evidence: list[int] | None = None) -> None:
        """Store a conclusion unless it restates one already drawn (small models reach the same one nightly)."""
        if repeats(text, (m.text for m in self.memory.items if m.kind == "reflection")):
            return
        await self.memory.add("reflection", text, now, importance, evidence)
        self.new_reflections.append(text)

    async def review_day(self, ctx: Ctx, now: datetime) -> None:
        self.reviewed_date = day_key(now)
        start = datetime.combine(day_key(now), DAY_START)
        today = [m for m in self.memory.items if m.created >= start and m.kind != "plan" and m.importance >= 3]
        if not today:
            return
        try:
            out = await ctx.llm.structured(
                prompts.system_prompt(self), prompts.day_review(self, ctx.names, now, today[-40:]),
                DayReview, label="day_review")
        except LLMError as e:
            log.warning("%s: day review failed (%s)", self.id, e)
            return
        await self._add_reflection(out.summary, now, out.importance)
        for intent in out.intentions:
            await self.memory.add("plan", f"Tomorrow: {intent}", now, 4)
        if out.drive.strip():
            self.drive = out.drive.strip()
        if out.situation.strip():
            self.situation = out.situation.strip()
        self.set_states(out.states, now)
        await self.update_history(ctx, now, today)

    async def update_history(self, ctx: Ctx, now: datetime, today: list) -> None:
        """After a very important day: one new lasting fact, if the day's own memories show it. Never deletes by
        itself: a held fact goes only to make room for a new one."""
        if not any(m.importance >= LIFE_FACT_IMPORTANCE for m in today):
            return
        shown = [m for m in today if m.importance >= 6][-15:]
        try:
            out = await ctx.llm.structured(
                prompts.life_facts_system_prompt(), prompts.life_facts(self, ctx.names, now, shown),
                LifeFacts, label="life_facts", temperature=0.3)
        except LLMError as e:
            log.warning("%s: life facts failed (%s)", self.id, e)
            return
        held = {f["text"].strip().lower() for f in self.history}
        new = [t.strip() for t in out.facts if t.strip() and t.strip().lower() not in held
               and not restates_background(t, self.persona)]
        proof = " ".join(WORD_RE.findall(out.evidence.lower()))
        if not new or not proof or proof not in " ".join(WORD_RE.findall(" ".join(m.text for m in shown).lower())):
            return  # nothing new, or nothing in the day that shows it
        kept = {t.strip().lower() for t in out.facts}
        left_out = [f for f in self.history if f["text"].strip().lower() not in kept]
        history = [f for f in self.history if len(left_out) != 1 or f is not left_out[0]]  # one obsolete fact may go
        history.append({"text": new[0], "since": day_key(now).isoformat()})
        self.history = history[-MAX_HISTORY:]

    def state_ages(self, now: datetime) -> list[tuple[dict, int]]:
        """Each state with its age in days; one that has no date yet starts counting today."""
        today = day_key(now)
        for s in self.states:
            if not s.get("since"):
                s["since"] = today.isoformat()
        return [(s, (today - date.fromisoformat(s["since"])).days) for s in self.states]

    def set_states(self, wanted: list[str], now: datetime) -> None:
        """The day's verdict on how they are now. A state still named keeps its age. At most MAX_STATES: the ones
        already held come first, newcomers only while there is room."""
        old = {s["text"].strip().lower(): s for s in self.states}
        texts = [t.strip() for t in wanted if t.strip()]
        ordered = [t for t in texts if t.lower() in old] + [t for t in texts if t.lower() not in old]
        result: list[dict] = []
        for text in ordered:
            if len(result) >= MAX_STATES:
                break
            if any(r["text"].lower() == text.lower() for r in result):
                continue
            previous = old.get(text.lower())
            result.append({"text": text, "since": previous["since"] if previous else day_key(now).isoformat()})
        self.states = result

    async def decide_baby_home(self, ctx: Ctx, child: "Agent", father: "Agent", now: datetime) -> bool:
        """Where the mother puts her goblin child: True to give it to its father's home, False to keep it. Her
        reason becomes a memory of hers; the engine moves the child. A failed call keeps it with her."""
        self.thinking = f"deciding where {child.first} will live"
        try:
            mems = await self.memory.retrieve(
                [f"my child {child.first}", f"{father.first} and the child we made", "who must never learn of it"],
                now, k=8)
            out = await ctx.llm.structured(
                prompts.system_prompt(self), prompts.baby_home(self, ctx.names, now, child, father, mems,
                                                          ctx.world.place_name(self.persona.home),
                                                          ctx.world.place_name(father.persona.home)),
                BabyHome, label="baby_home", temperature=0.7)
        except LLMError as e:
            log.warning("%s: choosing a home for %s failed (%s)", self.id, child.id, e)
            return False
        finally:
            self.thinking = None
        await self._remember("thought", out.reason, now, 7)
        return out.where == "father"

    async def pray(self, ctx: Ctx, now: datetime, god_id: str) -> None:
        """Pray to one god. The god may only touch this agent: its own memory, its own favor, its own
        system prompt - never anyone else's bonds, directly or by proxy. See hollowmere/agents/gods.py
        and prompts.god_system_prompt() for the boundary stated to the model itself."""
        god = GOD_BY_ID.get(god_id)
        if god is None:
            return
        idx = self.current_block(now)
        offering = self.plan[idx].activity if idx is not None else "prays"
        self.thinking = f"praying to {god.name}"
        try:
            mems = await self.memory.retrieve(
                [f"{self.first}'s past prayers and dealings with {god.name}", god.domain], now, k=8)
            seen = {m.id for m in mems}
            recent = [m for m in self.memory.items if m.id not in seen and m.kind in ("observation", "chat", "reflection")
                      and m.importance >= 5 and now - m.created <= timedelta(days=1)][-6:]
            out = await ctx.llm.structured(
                prompts.god_system_prompt(god), prompts.divine_prayer(self, god, now, offering, mems, recent),
                DivineAnswer, label="divine_answer")
        except LLMError as e:
            log.warning("%s: prayer to %s failed (%s)", self.id, god.name, e)
            self.thinking = None
            return
        self.thinking = None
        memory = recognised(out.text, god.name, out.clarity)
        if out.form == "dream":
            self.pending_dream = {"text": memory, "importance": out.importance}
        else:
            await self.observe(memory, now, out.importance)
        if out.form == "buff":
            hours = max(1, out.duration_hours)
            self.favor = {"god": god.id, "text": out.effect or out.text, "clarity": out.clarity,
                         "expires": (now + timedelta(hours=hours)).isoformat()}
        eased = out.eases.strip().lower()
        if eased or out.lingers.strip():
            self.set_states([s["text"] for s in self.states if s["text"].lower() != eased] + [out.lingers],
                            now)
        self.new_prayers.append({"god": god.id, "form": out.form, "text": out.text, "clarity": out.clarity})

    async def observe(self, text: str, now: datetime, importance: int) -> None:
        await self._remember("observation", text, now, importance)

    async def _remember(self, kind, text: str, now: datetime, importance: int) -> None:
        await self.memory.add(kind, text, now, importance)
        if kind in ("observation", "chat"):  # only what was experienced counts toward the next reflection:
            self.importance_acc += importance  # thoughts and reflections never trigger one (as in the original)

    # ------------------------------------------------------------ persistence

    def snapshot(self) -> dict:
        return {
            "x": self.x, "y": self.y, "plan": [b.to_dict() for b in self.plan],
            "plan_date": self.plan_date.isoformat() if self.plan_date else None,
            "tasks": [t.to_dict() for t in self.tasks], "tasks_block": self.tasks_block,
            "last_chat": {k: v.isoformat() for k, v in self.last_chat.items()},
            "last_react": {k: v.isoformat() for k, v in self.last_react.items()},
            "importance_acc": self.importance_acc,
            "reviewed_date": self.reviewed_date.isoformat() if self.reviewed_date else None,
            "commitments": self.commitments, "allies": sorted(self.allies), "regard": self.regard,
            "topics": list(self.topics),
            "announced": self.announced.isoformat() if self.announced else None,
            "bonds": self.bonds,
            "hiding_until": self.hiding_until.isoformat() if self.hiding_until else None,
            "seek": self.seek, "seek_block": self.seek_block,
            "drive": self.drive, "situation": self.situation, "town_state": self.town_state,
            "states": self.states, "history": self.history, "favor": self.favor, "pending_dream": self.pending_dream, "prayed_block": self.prayed_block,
            "absence": self.absence, "home": self.persona.home, "home_object": self.persona.home_object,
        }

    def restore(self, d: dict) -> None:
        self.x, self.y = d["x"], d["y"]
        self.plan = [Block.from_dict(b) for b in d["plan"]]
        self.plan_date = date.fromisoformat(d["plan_date"]) if d["plan_date"] else None
        self.tasks = [Task.from_dict(t) for t in d["tasks"]]
        self.tasks_block = d["tasks_block"]
        self.last_chat = {k: datetime.fromisoformat(v) for k, v in d["last_chat"].items()}
        self.last_react = {k: datetime.fromisoformat(v) for k, v in d.get("last_react", {}).items()}
        self.importance_acc = d["importance_acc"]
        self.reviewed_date = date.fromisoformat(d["reviewed_date"]) if d["reviewed_date"] else None
        self.announced = date.fromisoformat(d["announced"]) if d.get("announced") else None
        # Saves from before the checks above hold duplicates and clashes; keep the first of each hour.
        kept: list[dict] = []
        for c in d["commitments"]:
            try:
                if all(abs(due_at(o) - due_at(c)) >= PROMISE_GAP for o in kept):
                    kept.append({**c, "with": c.get("with")})
            except ValueError:
                continue
        self.commitments = kept
        self.allies = set(d.get("allies", []))
        self.regard = d.get("regard", {})
        self.topics = deque(d.get("topics", []), maxlen=TOPICS_KEPT)
        self.hiding_until = datetime.fromisoformat(d["hiding_until"]) if d.get("hiding_until") else None
        self.seek, self.seek_block = d.get("seek"), d.get("seek_block", -1)
        self.bonds.update(d.get("bonds", {}))  # an older save has none: keep the persona's
        for b in self.bonds.values():
            b.setdefault("affinity", starting_affinity(b.get("stance", "neutral")))  # older saves lack this
        self.drive = d.get("drive", "")  # older saves lack this
        self.situation = d.get("situation", "")  # older saves lack this
        self.town_state = d.get("town_state", "")  # older saves lack this
        self.states = d.get("states", self.states)  # older saves lack this: the persona's own stand
        self.history = d.get("history", [])  # older saves lack this
        self.favor = d.get("favor")  # older saves lack this
        self.pending_dream = d.get("pending_dream")  # older saves lack this
        self.prayed_block = d.get("prayed_block", -1)  # older saves lack this
        self.absence = d.get("absence")  # older saves lack this
        if self.is_away:
            self.mode = "away"
