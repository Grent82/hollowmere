"""Prompt construction.

All prompts share one system prompt (the agent's voice and the world's tone)
and are assembled from the same blocks: identity, situation, memories, places.
The original kept ~40 near-duplicate text templates with positional !<INPUT n>!
slots; here each prompt is a small function and the shared parts are written once.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Iterable

from hollowmere.memory.stream import Memory

if TYPE_CHECKING:
    from hollowmere.agents.agent import Agent
    from hollowmere.world.world import World


WORLD_TONE = (
    "Hollowmere is a small river town in the year 1347, in a grim medieval world: the plague has returned, "
    "the harvest has failed, and Lord Maelor has not left his keep in months. Life is hard, people are frightened "
    "and superstitious, and the church, the watch and the tavern are the only order left. There is no real magic, "
    "only rumour and faith."
)


def system_prompt(agent: Agent) -> str:
    p = agent.persona
    return (
        f"You are the mind of {p.name}, who lives in Hollowmere. {WORLD_TONE}\n\n"
        f"Think, decide and speak exactly as {p.first_name} would, with {p.first_name}'s knowledge, temperament, "
        f"fears and way of speaking. {p.first_name} knows only what is in the memories you are shown and what any "
        f"townsperson would know; do not invent facts about other people, though suspicion and rumour are fine. "
        "Stay concrete and grounded in the town. No modern words or ideas. Times are 24-hour."
    )


def when(t: datetime, now: datetime) -> str:
    days = (now.date() - t.date()).days
    clock = t.strftime("%H:%M")
    if days == 0:
        return f"today {clock}"
    if days == 1:
        return f"yesterday {clock}"
    return f"{days} days ago"


def bond_line(agent: Agent, other_id: str) -> str:
    """How the agent sees someone: the persona's starting knowledge, then what talks have made of it."""
    base = agent.persona.relationships.get(other_id, "")
    b = agent.bonds.get(other_id)
    if b is None:
        return base
    talks = f", {b['talks']} talks" if b["talks"] else ""
    now = f"{b['note']} ({b['stance']}{talks})" if b["note"] else f"({b['stance']}{talks})"
    return f"{base}; now: {now}" if base and b["note"] else f"{base} {now}".strip()


def identity(agent: Agent, names: dict[str, str]) -> str:
    p = agent.persona
    rel = "\n".join(f"- {names.get(k, k)}: {bond_line(agent, k)}" for k in agent.bonds)
    text = (
        f"WHO YOU ARE\n{p.name}, {p.age}. Temperament: {p.traits}.\n"
        + (f"Looks: {p.appearance}\n" if p.appearance else "")
        + f"{p.background}\n"
        f"Lately: {p.currently}\nHabits: {p.lifestyle}\n"
    )
    if p.abilities:
        text += f"Gifts and weaknesses: {p.abilities}\n"
    if p.is_goblin:
        friends = ", ".join(names.get(a, a) for a in sorted(agent.allies)) or "no one yet"
        text += (f"Humans who have sworn to protect you: {friends}. Anyone else could give you away; "
                 "how far to trust each of them is your own judgement.\n")
    return text + f"People you know:\n{rel or '- no one well'}"


def memories(mems: Iterable[Memory], now: datetime, title: str = "WHAT YOU REMEMBER") -> str:
    lines = [f"[{m.id}] ({when(m.created, now)}) {m.text}" for m in mems]
    return f"{title} (oldest first)\n" + ("\n".join(lines) if lines else "(nothing relevant)")


PRIVATE_KINDS = {"thought", "reflection", "plan"}


def townsfolk(agent: Agent, names: dict[str, str], known_as: dict[str, str]) -> str:
    """Everyone in town as anyone would know them, so nobody mistakes a neighbour for a stranger."""
    lines = [f"- {names.get(k, k)}: {v}" for k, v in known_as.items() if k != agent.id]
    return "THE TOWNSFOLK (as everyone knows them)\n" + "\n".join(lines) if lines else ""


def recent_topics(topics: list[str], first: str) -> str:
    if not topics:
        return ""
    return (f"{first} has lately talked about these, and says nothing new by repeating them:\n"
            + "\n".join(f"- {t}" for t in topics) + "\n")


def places(agent: Agent, world: World) -> str:
    lines = []
    for area_id in agent.known_areas:
        area = world.areas[area_id]
        lines.append(f"{area_id} — {area.name}: {area.description}")
        for obj in world.objects_in(area_id):
            lines.append(f"  {obj.place_id} — {obj.name}")
    return "PLACES (id — description)\n" + "\n".join(lines)


def clock_line(now: datetime) -> str:
    return f"It is {now.strftime('%A')}, {now.day} {now.strftime('%B')} {now.year}, {now.strftime('%H:%M')}."


# ---------------------------------------------------------------------- prompts


def plan_day(agent: Agent, world: World, names: dict[str, str], now: datetime,
             mems: list[Memory], commitments: list[str]) -> str:
    p = agent.persona
    promises = "\n".join(f"- {c}" for c in commitments) or "- none"
    return f"""{clock_line(now)} {p.first_name} is still abed and the day ahead is unplanned.

{identity(agent, names)}

{memories(mems, now)}

PROMISES AND INTENTIONS FOR TODAY
{promises}

{places(agent, world)}
Your bed is {p.bed}. Your work is at {p.work}.

Plan {p.first_name}'s waking day from getting up to going to bed.
- Blocks are in time order, do not overlap, and leave no gaps from wake_up to bedtime.
- Use 6 to 10 blocks of 1 to 4 hours each; they are broken into steps later. Meals, work, errands,
  rest and worries all belong in the day.
- Keep every promise at its agreed time and place.
- Follow {p.first_name}'s habits, but let what is on {p.first_name}'s mind change the day where it would.
- The final block is going to sleep at {p.bed}.
- activity is a short present-tense phrase without the name, e.g. "hammers spearheads for the watch"."""


def decompose(agent: Agent, world: World, names: dict[str, str], now: datetime, activity: str,
              start: datetime, end: datetime, area_name: str, mems: list[Memory]) -> str:
    p = agent.persona
    minutes = int((end - start).total_seconds() // 60)
    return f"""{clock_line(now)}

{identity(agent, names)}

{memories(mems, now)}

From {start:%H:%M} to {end:%H:%M} ({minutes} minutes) {p.first_name} {activity}, at {area_name}.
Break this into the concrete steps {p.first_name} actually takes, in order. The minutes must add up to {minutes}.
Use only the place ids allowed by the schema. Prefer specific objects when the step uses one.{seek_hint(agent)}"""


def seek_hint(agent: Agent) -> str:
    if not agent.persona.is_goblin:
        return ""
    first = agent.persona.first_name
    return (f"\nIf {first} means to find someone during this time and creep close to them, name them in seek; "
            f"{first} will follow them wherever they go out of doors.")


def react(agent: Agent, names: dict[str, str], now: datetime, place: str, own_activity: str,
          other_name: str, other_activity: str, mems: list[Memory], aside: str = "",
          topics: list[str] = (), other_looks: str = "") -> str:
    p = agent.persona
    looks = f" ({other_looks})" if other_looks else ""
    return f"""{clock_line(now)} {p.first_name} is at {place}, {own_activity}.
{p.first_name} notices {other_name}{looks}, who {other_activity}.

{identity(agent, names)}

{memories(mems, now)}
{aside}
{recent_topics(list(topics), p.first_name)}
How does {p.first_name} react? Speaking up costs time and may draw attention; most of the time people simply
carry on. Talk only if {p.first_name} has a real reason to: something to ask, tell, warn, sell, confess or
quarrel about, or a friendship worth tending. Going over the same worry again with someone new is
rarely reason enough."""


def converse(agent: Agent, names: dict[str, str], now: datetime, place: str, other_name: str,
             other_activity: str, purpose: str, transcript: list[tuple[str, str]], mems: list[Memory],
             aside: str = "", town_places: Iterable[str] = (), topics: list[str] = (),
             known_as: dict[str, str] | None = None, avoid: str = "", other_looks: str = "") -> str:
    p = agent.persona
    lines = "\n".join(f'{who}: "{text}"' for who, text in transcript) or "(nobody has spoken yet)"
    other_id = next((k for k, v in names.items() if v == other_name), None)
    title = p.name.split()[0] if p.name.split()[0] != p.first_name else ""
    title_rule = (f" Others call {p.first_name} \"{title}\"; {p.first_name} calls others by their names "
                  f"and never calls them \"{title}\"." if title else "")
    known_as = known_as or {}
    stranger = ("a goblin, not a person of the town (see above)" if aside
                else known_as.get(other_id, "someone from the town"))
    who_they_are = (bond_line(agent, other_id) or stranger) if other_id else stranger
    retry = (f'{p.first_name} was about to say "{avoid}", but that has already been said. Say something new, '
             "or say goodbye and set end to true.\n" if avoid else "")
    known = [m for m in mems if m.kind not in PRIVATE_KINDS]
    private = [m for m in mems if m.kind in PRIVATE_KINDS]
    opener = (f"{p.first_name} approached {other_name}. {p.first_name}'s private aim (never say it word for word; "
              f"open the way a person would, speaking to {other_name} directly): {purpose}\n"
              if purpose and not transcript else "")
    return f"""{clock_line(now)} {p.first_name} is at {place}, talking with {other_name}, who was {other_activity}.
{opener}
{identity(agent, names)}
{townsfolk(agent, names, known_as)}

{memories(known, now, title="WHAT YOU HAVE SEEN AND HEARD")}

{memories(private, now, title="YOUR PRIVATE THOUGHTS")}
These are {p.first_name}'s own suspicions, schemes and fears. They shape what {p.first_name} wants from this
talk, but {p.first_name} does not speak them aloud: no confessing a plan, a secret or a scheme, unless
{p.first_name} truly trusts {other_name} with it, and even then guardedly.
{aside}
{recent_topics(list(topics), p.first_name) if not transcript else ""}
CONVERSATION SO FAR
{lines}

Hollowmere has only these places: {", ".join(town_places)}. Name no other place.

You are {p.name}. {other_name} is {who_they_are}.{f" To look at: {other_looks}." if other_looks else ""}{title_rule}
{retry}Write the next line spoken BY {p.first_name} TO {other_name}. Speak in {p.first_name}'s own voice, plainly and briefly, the way a frightened
townsperson in 1347 talks. Respond to what was just said. Share only what {p.first_name} knows, and only what
{p.first_name} would be willing to say to {other_name}. Do not repeat earlier lines. Set end to true to say
goodbye once the talk has run its course."""


def chat_digest(agent: Agent, names: dict[str, str], now: datetime, other_name: str,
                transcript: list[tuple[str, str]], aside: str = "") -> str:
    p = agent.persona
    lines = "\n".join(f'{who}: "{text}"' for who, text in transcript)
    return f"""{clock_line(now)} {p.first_name} has just finished talking with {other_name}.

{identity(agent, names)}

CONVERSATION
{lines}
{aside}
Remember this conversation the way {p.first_name} will: what was said, what was learned, how it felt.
Commitments are only what {p.first_name} and {other_name} agreed on in this talk, for a definite time later
today or tomorrow: to meet somewhere, to bring or fetch something, to do a task for or with the other.
Not {p.first_name}'s usual work, not anything already done, not what {p.first_name} would do anyway, and
never a time that has already passed. A request or demand nobody answered is not agreed. If no time was
spoken, there is no commitment. Most talks agree on nothing: then leave the list empty."""


def focal_questions(agent: Agent, names: dict[str, str], now: datetime, mems: list[Memory]) -> str:
    p = agent.persona
    return f"""{clock_line(now)}

{identity(agent, names)}

{memories(mems, now, title="RECENT MEMORIES")}

Given only these memories, what are the most important questions on {p.first_name}'s mind right now?"""


def insights(agent: Agent, names: dict[str, str], now: datetime, questions: list[str], mems: list[Memory]) -> str:
    p = agent.persona
    qs = "\n".join(f"- {q}" for q in questions)
    return f"""{clock_line(now)}

{identity(agent, names)}

QUESTIONS ON YOUR MIND
{qs}

{memories(mems, now, title="MEMORIES THAT BEAR ON THEM")}

Lying awake with these questions, what does {p.first_name} conclude? Each conclusion must follow from the
numbered memories it cites. Prefer conclusions about people, intentions and dangers over restating events."""


def day_review(agent: Agent, names: dict[str, str], now: datetime, mems: list[Memory]) -> str:
    p = agent.persona
    return f"""{clock_line(now)} {p.first_name} is lying down to sleep.

{identity(agent, names)}

{memories(mems, now, title="TODAY")}

Look back on {p.first_name}'s day: what happened, what changed, and what {p.first_name} means to do tomorrow."""



def goblin_choice(agent: Agent, world: World, names: dict[str, str], now: datetime, place: str,
                  seen: list, talking_with: Agent | None, hidden: bool, mems: list[Memory]) -> str:
    """seen: (person, distance in tiles, whether they have seen him)."""
    p = agent.persona
    people = []
    for h, dist, saw in seen:
        known = bond_line(agent, h.id) or "a stranger to you"
        mark = " (sworn to protect you)" if h.id in agent.allies else ""
        noticed = "has seen you" if saw else "has not seen you"
        people.append(f"- {h.name}{mark}, {dist:.0f} paces off, {h.activity_phrase(world)}; {noticed}. "
                      f"To you: {known}")
    where = "hidden in cover" if hidden else f"at {place}"
    talk = (f"You are in the middle of talking with {talking_with.name} when the others come into sight.\n"
            if talking_with else "")
    alone = "alone" if len(seen) == 1 else f"{len(seen)} of them together"
    return f"""{clock_line(now)} {p.first_name} is {where}, and people have come into sight ({alone}).

{identity(agent, names)}

{memories(mems, now)}

IN SIGHT
{chr(10).join(people)}
{talk}
What does {p.first_name} do? Any human could hang {p.first_name}, or feed him; several together are harder to
read than one alone. Decide from what {p.first_name} knows of each of them."""


def goblin_aside(listener: Agent, goblin: Agent) -> str:
    """What a human perceives when the goblin is close. The musk colours judgement; it decides nothing."""
    if listener.id in goblin.allies:
        return f"{goblin.name} the goblin is someone you have chosen to protect.\n"
    if listener.persona.resists_charm:
        return ("The creature before you is a goblin. You smell its sweet, earthy musk and know it for what it is: "
                "the old goblin glamour that makes folk pity them. Your judgement is your own.\n")
    looks = goblin.persona.appearance or "small and starved"
    return (f"The creature before you is a goblin, {looks}. A sweet, earthy musk of wet moss "
            "and wild honey hangs about it, and despite every tale you have heard, it seems more pitiable than "
            "dangerous, almost dear, like a half-starved stray. Whether to help it, "
            "drive it off or raise the alarm is still your own choice.\n")
