"""`hollowmere report`: judge a run from its save instead of by watching it.

Reads what the engine records as it runs (events, position samples every ten
game minutes, memories, LLM counters) and checks the things that go wrong with
a small model: people standing in the wrong place or stuck on the road, talks
that loop or spill private thoughts, the same topic raised over and over,
reflections drawn twice, promises not kept, and LLM calls that fail.

Lines marked CHECK are worth a look; they are heuristics, not proof of a bug.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta

from hollowmere.agents.persona import Persona
from hollowmere.memory.store import Store
from hollowmere.memory.stream import repeats
from hollowmere.world.world import World

SAMPLE_MINUTES = 10
NIGHT = range(1, 4)  # 01:00 to 03:59
LEAK_WORDS = 7       # a private thought counts as spoken when this many words in a row reappear
STOP = set("""the a an and or but of to in on at for with from by is are was were be been it its this that these those
i me my you your he him his she her we us our they them their not no so as if then than there here what who
which when where how all any some one into out up down about would could should will shall must have has had
do does did just only very more most own same such too can""".split())

_WORD = re.compile(r"[a-z']+")


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _shingles(text: str, n: int = LEAK_WORDS) -> set[tuple[str, ...]]:
    w = _words(text)
    # at least three real words, so "and I need to know what" is not taken for a secret
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1) if sum(x not in STOP for x in w[i:i + n]) >= 3}


def _content(text: str) -> set[str]:
    return {w for w in _words(text) if len(w) > 3 and w not in STOP}


def _similar_topic(a: str, b: str) -> bool:
    x, y = _content(a), _content(b)
    return bool(x and y) and len(x & y) / len(x | y) >= 0.4


def _minutes(c: dict) -> int:
    h, m = (int(x) for x in c["time"].split(":"))
    return h * 60 + m


def _pct(n: int, total: int) -> str:
    return f"{100 * n / total:.0f}%" if total else "-"


def _flag(bad: bool) -> str:
    return "CHECK" if bad else "ok   "


def _cut(text: str, n: int = 110) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


class Report:
    def __init__(self, store: Store, world: World, personas: list[Persona], hours: float | None = None):
        self.db = store.db
        self.world = world
        self.people = {p.id: p for p in personas}
        clock = store.get("clock")
        self.now = datetime.fromisoformat(clock) if clock else None
        self.since = self.now - timedelta(hours=hours) if self.now and hours else None
        self.snaps: dict = store.get("agents", {}) or {}
        self.llm: dict = store.get("llm_stats", {}) or {}
        self.out: list[str] = []

    # --------------------------------------------------------------- reading

    def _rows(self, sql: str, args: tuple = ()) -> list:
        return self.db.execute(sql, args).fetchall()

    def _since(self, col: str = "time") -> tuple[str, tuple]:
        return (f" WHERE {col} >= ?", (self.since.isoformat(),)) if self.since else ("", ())

    def _events(self, *kinds: str) -> list[dict]:
        where, args = self._since()
        rows = self._rows(f"SELECT * FROM events{where} ORDER BY id", args)
        return [{"time": datetime.fromisoformat(r["time"]), "kind": r["kind"], "agents": json.loads(r["agents"]),
                 "text": r["text"], **json.loads(r["extra"])} for r in rows if not kinds or r["kind"] in kinds]

    def _name(self, aid: str) -> str:
        p = self.people.get(aid)
        return p.first_name if p else aid

    # ---------------------------------------------------------------- output

    def line(self, text: str = "") -> None:
        self.out.append(text)

    def head(self, title: str) -> None:
        self.line()
        self.line(title)
        self.line("-" * len(title))

    def build(self) -> str:
        self.line("HOLLOWMERE RUN REPORT")
        if self.now is None:
            self.line("The save has no clock yet: run the game first.")
            return "\n".join(self.out)
        first = self._rows("SELECT MIN(time) AS t FROM samples")[0]["t"]
        start = max(datetime.fromisoformat(first), self.since) if first and self.since else (
            datetime.fromisoformat(first) if first else None)
        span = f"{start:%a %d %b %H:%M} to {self.now:%a %d %b %H:%M}" if start else f"up to {self.now:%a %H:%M}"
        self.line(f"Game time covered: {span}")
        if not first:
            self.line("No position samples or events recorded: this save predates the report. Memory-based")
            self.line("sections below still work; play on (or start --fresh) for the rest.")
        self.movement()
        self.conversations()
        self.reflections()
        self.promises()
        self.goblin()
        self.llm_calls()
        self.line()
        return "\n".join(self.out)

    # -------------------------------------------------------------- sections

    def movement(self) -> None:
        self.head("MOVEMENT")
        where, args = self._since()
        rows = self._rows(f"SELECT * FROM samples{where} ORDER BY agent_id, time", args)
        if not rows:
            self.line("no samples")
            return
        modes = Counter(r["mode"] for r in rows)
        self.line("time spent: " + ", ".join(f"{m} {_pct(n, len(rows))}" for m, n in modes.most_common()))

        acting = [r for r in rows if r["mode"] == "acting" and r["place"]]
        wrong = [r for r in acting if r["area"] != r["place"].split("/")[0]]
        self.line(f"{_flag(len(wrong) > 0.1 * len(acting))} busy at the task's own place: "
                  f"{_pct(len(acting) - len(wrong), len(acting))} of {len(acting)} samples")
        by_agent = Counter(r["agent_id"] for r in wrong)
        for aid, n in by_agent.most_common(4):
            r = next(r for r in wrong if r["agent_id"] == aid)
            here = self.world.areas[r["area"]].name if r["area"] in self.world.areas else "the open road"
            self.line(f"        {self._name(aid)}: {n}x, e.g. {r['time'][11:16]} meant to be at "
                      f"{self.world.place_name(r['place'])}, was at {here}")

        stuck, long_walks = [], []
        per_agent: dict[str, list] = defaultdict(list)
        for r in rows:
            per_agent[r["agent_id"]].append(r)
        for aid, rs in per_agent.items():
            run: list = []
            for r in rs + [None]:
                if r is not None and r["mode"] == "walking":
                    run.append(r)
                    continue
                if len(run) >= 3 and len({(x["x"], x["y"]) for x in run}) == 1:
                    stuck.append((aid, run))
                elif len(run) >= 6:
                    long_walks.append((aid, run))
                run = []
        self.line(f"{_flag(bool(stuck))} stuck (walking without moving for 30+ min): {len(stuck)}")
        for aid, run in stuck[:4]:
            self.line(f"        {self._name(aid)} at ({run[0]['x']},{run[0]['y']}) from {run[0]['time'][11:16]}")
        self.line(f"{_flag(bool(long_walks))} walks of an hour or more: {len(long_walks)}")
        for aid, run in long_walks[:4]:
            self.line(f"        {self._name(aid)} from {run[0]['time'][11:16]}, {len(run) * SAMPLE_MINUTES} min "
                      f"toward {self.world.place_name(run[-1]['place'] or '?')}")

        night = [r for r in rows if datetime.fromisoformat(r["time"]).hour in NIGHT
                 and not (self.people.get(r["agent_id"]) and self.people[r["agent_id"]].is_goblin)]
        awake = Counter(r["agent_id"] for r in night if r["mode"] != "asleep")
        self.line(f"{_flag(len(awake) > 2)} humans awake between 01:00 and 04:00: "
                  + (", ".join(f"{self._name(a)} ({_pct(n, sum(r['agent_id'] == a for r in night))})"
                               for a, n in awake.most_common()) or "none"))

    def conversations(self) -> None:
        self.head("CONVERSATIONS")
        intents = self._events("intent")
        ends = self._events("chat_end")
        lines = self._events("chat")
        if not ends:
            self.line("no conversations recorded")
        reasons = Counter(e.get("reason", "?") for e in ends)
        held = [e for e in ends if e.get("reason") not in ("fled", "never_reached")]
        per_convo = Counter(e.get("convo") for e in lines)
        lengths = [per_convo[e.get("convo")] for e in held]
        self.line(f"wanted to talk: {len(intents)}, actually talked: {len(held)}, "
                  f"fell through before a word: {reasons['fled'] + reasons['never_reached']}")
        if held:
            self.line(f"lines per talk: average {sum(lengths) / len(lengths):.1f}, longest {max(lengths)}")
            self.line("how talks ended: " + ", ".join(f"{k} {_pct(n, len(held))}" for k, n in reasons.most_common()
                                                      if k not in ("fled", "never_reached")))
            self.line(f"{_flag(reasons['limit'] > 0.5 * len(held))} talks cut off at the line limit: "
                      f"{_pct(reasons['limit'], len(held))}")
            self.line(f"{_flag(reasons['repeat'] > 0.3 * len(held))} talks ended because a line repeated: "
                      f"{_pct(reasons['repeat'], len(held))}")
            self.line(f"{_flag(reasons['error'] > 0)} talks broken by an LLM error: {reasons['error']}")

        # the same aim again and again
        by_pair: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for e in intents:  # asking several people the same thing is fair; asking one person twice is a loop
            by_pair[(e["agents"][0], e["agents"][1])].append(e)
        repeated, examples = 0, []
        for (aid, _), es in by_pair.items():
            for i, e in enumerate(es):
                aim = e["text"].split(": ", 1)[-1]
                earlier = [x for x in es[:i] if x["time"].date() == e["time"].date()]
                match = next((x for x in earlier if _similar_topic(aim, x["text"].split(": ", 1)[-1])), None)
                if match:
                    repeated += 1
                    if all(x[0] != aid for x in examples):  # one example per person
                        examples.append((aid, match["text"].split(": ", 1)[-1], aim))
        self.line(f"{_flag(repeated > 0.15 * max(1, len(intents)))} the same aim raised with the same person again that day: "
                  f"{repeated} of {len(intents)}")
        for aid, a, b in examples[:3]:
            self.line(f"        {self._name(aid)}: \"{_cut(a, 60)}\" / \"{_cut(b, 60)}\"")

        # private thoughts spoken aloud
        mems = self._memories()
        leaks = []
        for e in lines:
            sid = e.get("speaker")
            if not sid:
                continue
            said = _shingles(e["text"])
            if not said:
                continue
            public = set().union(*(_shingles(m["text"]) for m in mems[sid]
                                   if m["kind"] in ("observation", "chat") and m["created"] <= e["time"]))
            for m in mems[sid]:
                if m["kind"] in ("thought", "reflection") and m["created"] <= e["time"]:
                    if (said & _shingles(m["text"])) - public:
                        leaks.append((sid, m["text"], e["text"]))
                        break
        self.line(f"{_flag(bool(leaks))} lines that repeat the speaker's private thought: {len(leaks)} of {len(lines)}"
                  " (read the examples: some are fair to say)")
        for sid, thought, said in leaks[:3]:
            self.line(f"        {self._name(sid)} thought \"{_cut(thought, 60)}\"")
            self.line(f"          and said \"{_cut(said, 70)}\"")

    def _memories(self) -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = defaultdict(list)
        for r in self._rows("SELECT agent_id, kind, text, created, evidence FROM memories ORDER BY id"):
            out[r["agent_id"]].append({"kind": r["kind"], "text": r["text"],
                                       "created": datetime.fromisoformat(r["created"]),
                                       "evidence": json.loads(r["evidence"])})
        return out

    def reflections(self) -> None:
        self.head("REFLECTIONS")
        mems = self._memories()
        total, dups, latest = 0, [], []
        counts = []
        for aid, ms in mems.items():
            every = [m for m in ms if m["kind"] == "reflection"]
            refl = [m for m in every if not self.since or m["created"] >= self.since]
            total += len(refl)
            counts.append(f"{self._name(aid)} {len(refl)}")
            for i, m in enumerate(every):
                if (not self.since or m["created"] >= self.since) and repeats(m["text"], [x["text"] for x in every[:i]]):
                    dups.append((aid, m["text"]))
            if refl:
                latest.append((aid, refl[-1]["text"]))
        self.line(f"drawn: {total} ({', '.join(counts)})")
        self.line(f"{_flag(total == 0 and bool(self._rows('SELECT 1 FROM samples LIMIT 1')))} anyone reflecting at all")
        self.line(f"{_flag(bool(dups))} near-duplicates of an earlier reflection: {len(dups)}")
        for aid, text in dups[:3]:
            self.line(f"        {self._name(aid)}: \"{_cut(text, 90)}\"")
        self.line("latest per person (read these: do they follow from what happened?):")
        for aid, text in latest:
            self.line(f"        {self._name(aid)}: {_cut(text, 100)}")

    def promises(self) -> None:
        self.head("PLANS AND PROMISES")
        plans = {(e["agents"][0], e["time"].date()) for e in self._events("plan")}  # a restart re-announces
        self.line(f"day plans made: {len(plans)}")
        failed = self.llm.get("failures", {}).get("plan_day", 0)
        self.line(f"{_flag(failed > 0)} plans that fell back to habit (the model failed): {failed}")
        kept, missed, unseen = 0, [], 0
        for aid, snap in self.snaps.items():
            for c in snap.get("commitments", []):
                h, m = (int(x) for x in c["time"].split(":"))
                at = datetime.fromisoformat(c["date"]).replace(hour=h, minute=m)
                if at > self.now or (self.since and at < self.since):
                    continue
                area = c["place"].split("/")[0]
                rows = self._rows("SELECT area FROM samples WHERE agent_id=? AND time BETWEEN ? AND ?",
                                  (aid, (at - timedelta(minutes=15)).isoformat(),
                                   (at + timedelta(minutes=60)).isoformat()))
                if not rows:
                    unseen += 1
                elif any(r["area"] == area for r in rows):
                    kept += 1
                else:
                    missed.append((aid, c, at))
        due = kept + len(missed)
        self.line(f"{_flag(len(missed) > kept)} own promises kept (being at the place on time): {kept} of {due}"
                  + (f" ({unseen} more before sampling began)" if unseen else ""))
        for aid, c, at in missed[:4]:
            self.line(f"        {self._name(aid)} missed {at:%a %H:%M} {_cut(c['activity'], 50)} at "
                      f"{self.world.place_name(c['place'])}")
        clashes = 0
        for snap in self.snaps.values():
            cs = sorted(snap.get("commitments", []), key=lambda c: (c["date"], c["time"]))
            clashes += sum(1 for x, y in zip(cs, cs[1:]) if x["date"] == y["date"] and _minutes(y) - _minutes(x) < 60)
        self.line(f"{_flag(clashes > 0)} promises within the same hour as another of that person's: {clashes}")

        met, stood_up, upcoming = 0, [], 0
        for e in self._events("agreement"):
            h, m = (int(x) for x in e["at"].split(":"))
            at = datetime.fromisoformat(e["date"]).replace(hour=h, minute=m)
            if at > self.now:
                upcoming += 1
                continue
            area = e["place"].split("/")[0]
            there = [aid for aid in e["agents"] if any(r["area"] == area for r in self._rows(
                "SELECT area FROM samples WHERE agent_id=? AND time BETWEEN ? AND ?",
                (aid, (at - timedelta(minutes=15)).isoformat(), (at + timedelta(minutes=60)).isoformat())))]
            if len(there) == len(e["agents"]):
                met += 1
            else:
                stood_up.append((e, there))
        self.line(f"{_flag(len(stood_up) > met)} agreed meetings where both turned up: {met} of {met + len(stood_up)}"
                  + (f" ({upcoming} still to come)" if upcoming else ""))
        for e, there in stood_up[:4]:
            who = ", ".join(self._name(a) for a in there) or "nobody"
            self.line(f"        {_cut(e['text'], 80)} came: {who}")

    def goblin(self) -> None:
        goblins = [p for p in self.people.values() if p.is_goblin]
        if not goblins:
            return
        self.head("THE GOBLIN")
        kinds = Counter(e["kind"] for e in self._events("sighting", "hide", "flee", "ally", "warm", "spurn", "seek"))
        self.line(f"glimpsed {kinds['sighting']}x, hid {kinds['hide']}x, fled {kinds['flee']}x")
        # What he chose, by how he felt about the people in sight: hiding from friends would be a bad sign.
        by_stance: dict[str, Counter] = {}
        for e in self._events("choice"):
            for stance in set((e.get("stances") or {}).values()):
                by_stance.setdefault(stance or "stranger", Counter())[e.get("action")] += 1
        for stance in sorted(by_stance):
            acts = ", ".join(f"{a} {n}x" for a, n in by_stance[stance].most_common())
            self.line(f"        with someone {stance} in sight: {acts}")
        starts = self._events("chat_start")
        seeks = self._events("seek")
        found = sum(any(set(s["agents"]) == set(c["agents"]) and 0 <= (c["time"] - s["time"]).total_seconds() <= 7200
                        for c in starts) for s in seeks)
        if seeks:
            self.line(f"crept after someone {len(seeks)}x; a talk followed within two hours {found}x")
        for g in goblins:
            met = {e["agents"][0] if e["agents"][0] != g.id else e["agents"][1]
                   for e in self._events("chat_end") if g.id in e["agents"]}
            allies = self.snaps.get(g.id, {}).get("allies", [])
            self.line(f"{g.first_name} talked with: {', '.join(self._name(a) for a in sorted(met)) or 'nobody'}")
            self.line(f"protectors: {', '.join(self._name(a) for a in allies) or 'none'}; "
                      f"won over {kinds['ally']}x, softened {kinds['warm']}x, drove him off {kinds['spurn']}x")
            bonds = self.snaps.get(g.id, {}).get("bonds", {})
            regard = self.snaps.get(g.id, {}).get("regard", {})
            for aid, b in sorted(bonds.items(), key=lambda kv: -kv[1].get("talks", 0)):
                note = f": {b['note']}" if b.get("note") else ""
                mind = f", they: {regard[aid]['step']}" if aid in regard else ""
                self.line(f"        {self._name(aid)} ({b['stance']}, {b.get('talks', 0)} talks{mind}){note}")

    def llm_calls(self) -> None:
        self.head("LLM CALLS (since the save began)")
        calls, repairs, fails = (self.llm.get(k, {}) for k in ("calls", "repairs", "failures"))
        if not calls:
            self.line("no counters saved yet (they are written with each autosave)")
            return
        total, failed = sum(calls.values()), sum(fails.values())
        self.line(f"{_flag(failed > 0.05 * total)} {total} calls, {sum(repairs.values())} needed a retry, "
                  f"{failed} failed outright")
        self.line(f"        {'kind':<18}{'calls':>7}{'retried':>9}{'failed':>8}")
        for label in sorted(calls, key=calls.get, reverse=True):
            self.line(f"        {label:<18}{calls[label]:>7}{repairs.get(label, 0):>9}{fails.get(label, 0):>8}")
