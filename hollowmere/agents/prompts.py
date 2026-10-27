"""Prompt construction.

All prompts share one system prompt (the agent's voice and the world's tone)
and are assembled from the same blocks: identity, situation, memories, places.
The original kept ~40 near-duplicate text templates with positional !<INPUT n>!
slots; here each prompt is a small function and the shared parts are written once.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING, Iterable

from hollowmere.agents.gods import GOD_BY_ID, God
from hollowmere.agents.persona import DATA, RANK
from hollowmere.memory.stream import Memory

if TYPE_CHECKING:
    from hollowmere.agents.agent import Agent
    from hollowmere.world.world import World


_world = json.loads((DATA / "world.json").read_text(encoding="utf-8"))

# Fixed world rules: never rewritten by any LLM call, so they cannot drift away over many town
# reviews (the old gods and their gifts are named here and nowhere else should invent new ones,
# nor a modern idea, after enough rewrites). Lives in data/world.json so lore can be edited
# without touching code, the same way personas.json works.
WORLD_FACTS = _world["world_facts"]

# The town's current situation, in contrast, is meant to change: this is only the seed sentence an
# agent falls back on before the first town review has ever run (see Agent.town_state, town_review()).
DEFAULT_TOWN_STATE = _world["default_town_state"]


def system_prompt(agent: Agent) -> str:
    p = agent.persona
    situation = agent.town_state or DEFAULT_TOWN_STATE
    return (
        f"You are the mind of {p.name}, who lives in Hollowmere. {WORLD_FACTS} {situation}\n\n"
        f"Think, decide and speak exactly as {p.first_name} would, with {p.first_name}'s knowledge, hopes and fears. "
        f"When you speak or put feelings into words, let {p.first_name}'s own age, temperament, work and "
        f"background below shape not just what is said but how: word choice, formality, warmth, how long or "
        f"short the sentences run, any habits of speech. No two people in Hollowmere think or talk alike. "
        f"{p.first_name} knows only what is in the memories you are shown and what any townsperson would know; "
        f"do not invent facts about other people. Suspicion of any one person must be earned by what they "
        f"themselves have said or done; the plague and the age's dread are the water everyone swims in and are "
        f"never, by themselves, reason to distrust a particular neighbour. "
        "Stay concrete and grounded in the town. No modern words or ideas. Times are 24-hour."
    )


def when(t: datetime, now: datetime, keep_time: bool = False) -> str:
    days = (now.date() - t.date()).days
    clock = t.strftime("%H:%M")
    if days == 0:
        return f"today {clock}"
    if days == 1:
        return f"yesterday {clock}"
    return f"{days} days ago at {clock}" if keep_time else f"{days} days ago"


def bond_line(agent: Agent, other_id: str, show_stance: bool = True) -> str:
    """How the agent sees someone: the persona's starting knowledge, then what talks have made of it.

    show_stance=False drops the "(stance, N talks)" tag: used when the model is about to judge a fresh
    feeling toward this very person, since showing it its own last verdict just invites it to repeat it."""
    b = agent.bonds.get(other_id)
    base = agent.persona.relationships.get(other_id, "") or (b or {}).get("kin", "")
    if b is None:
        return base
    if not show_stance:
        return f"{base}; now: {b['note']}" if base and b["note"] else (b["note"] or base)
    talks = f", {b['talks']} talks" if b["talks"] else ""
    now = f"{b['note']} ({b['stance']}{talks})" if b["note"] else f"({b['stance']}{talks})"
    return f"{base}; now: {now}" if base and b["note"] else f"{base} {now}".strip()


# What each GoblinRegard step means, as a human's decision about the goblin.
REGARD_WORDS = {"drive off": "drive it off", "undecided": "make up your mind later", "tolerate": "put up with it",
                "feed": "feed it", "protect": "protect it"}


def regard_line(goblin: Agent, human_id: str, now: datetime, them: str = "they") -> str:
    """How a human last decided about the goblin, and how often they have driven him off."""
    r = goblin.regard.get(human_id)
    if not r:
        return ""
    line = f"after the last talk ({when(datetime.fromisoformat(r['at']), now)}) {them} chose to {REGARD_WORDS[r['step']]}"
    if r.get("rebuffs"):
        line += f"; driven off {r['rebuffs']} time{'s' if r['rebuffs'] > 1 else ''} in all"
    return line


def identity(agent: Agent, names: dict[str, str], hide_stance_for: str | None = None) -> str:
    p = agent.persona
    rel = "\n".join(f"- {names.get(k, k)}: {bond_line(agent, k, show_stance=k != hide_stance_for)}"
                    for k in agent.bonds)
    text = (
        f"WHO YOU ARE\n{p.name}, {p.age}, {p.sex}. Temperament: {p.traits}.\n"
        + (f"Looks: {p.appearance}\n" if p.appearance else "")
        + f"{p.background}\n"
        f"Lately: {agent.situation or p.currently}\nHabits: {p.lifestyle}\n"
    )
    if agent.condition:
        text += f"{agent.condition}\n"
    if p.abilities:
        text += f"Gifts and weaknesses: {p.abilities}\n"
    if agent.drive:
        text += f"What {p.first_name} wants for themself right now: {agent.drive}\n"
    if agent.favor:
        god = GOD_BY_ID.get(agent.favor["god"])
        if agent.favor.get("clarity", "suspected") in ("suspected", "certain"):  # older favors named their god
            text += f"A favor from {god.name if god else agent.favor['god']} lingers on {p.first_name}: {agent.favor['text']}\n"
        else:
            text += f"A strange mood lingers on {p.first_name}: {agent.favor['text']}\n"
    if p.is_goblin:
        friends = ", ".join(names.get(a, a) for a in sorted(agent.allies)) or "no one yet"
        kind = ", ".join(f"{names.get(k, k)} ({REGARD_WORDS[r['step']]})" for k, r in sorted(agent.regard.items())
                         if r["step"] in ("tolerate", "feed")) or "no one yet"
        text += (f"Humans who have sworn to protect you: {friends}. Humans who have softened toward you: {kind}. "
                 "Anyone else could give you away; how far to trust each of them is your own judgement.\n")
    return text + f"People you know:\n{rel or '- no one well'}"


def memories(mems: Iterable[Memory], now: datetime, title: str = "WHAT YOU REMEMBER") -> str:
    # Observations are kept with their clock time no matter how old: knowing someone was at the
    # tavern "3 days ago" is useless for guessing where they'll be today, but "3 days ago at 19:00" lets
    # a pattern emerge across many such sightings. Thoughts and reflections don't need that precision.
    lines = [f"[{m.id}] ({when(m.created, now, keep_time=m.kind == 'observation')}) {m.text}" for m in mems]
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
             mems: list[Memory], commitments: list[str], known_as: dict[str, str] | None = None) -> str:
    p = agent.persona
    promises = "\n".join(f"- {c}" for c in commitments) or "- none"
    return f"""{clock_line(now)} {p.first_name} is still abed and the day ahead is unplanned.

{identity(agent, names)}
{townsfolk(agent, names, known_as or {})}

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
- If the day involves another person, only give a specific time or place for them if what you know of
  them above (their trade, or what you've seen of them) supports it; otherwise stay uncertain about
  them - hoping, watching, asking around - rather than inventing a meeting that contradicts what you know.
- A block says only what {p.first_name} does, never what another person does, that they are there, or that
  they come along. Name no one else in a block unless {p.first_name} has promised to meet them (see the
  promises above): hoping to run into someone is a worry, not a plan.
- The final block is going to sleep at {p.bed}.
- activity is a short present-tense phrase without the name, e.g. "hammers spearheads for the watch".
- If {p.first_name} means to pray during any block, set that block's pray_to to the one god addressed
  (caedros, yevra, agrath or sevreth); every other block's pray_to is "none". Whether, when, and to which
  god {p.first_name} prays is entirely {p.first_name}'s own choice, shaped by piety, dread, hope or habit -
  most days call for no prayer at all."""


def decompose(agent: Agent, world: World, names: dict[str, str], now: datetime, activity: str,
              start: datetime, end: datetime, area_name: str, mems: list[Memory]) -> str:
    p = agent.persona
    minutes = int((end - start).total_seconds() // 60)
    return f"""{clock_line(now)}

{identity(agent, names)}

{memories(mems, now)}

From {start:%H:%M} to {end:%H:%M} ({minutes} minutes) {p.first_name} {activity}, at {area_name}.
Break this into the concrete steps {p.first_name} actually takes, in order. The minutes must add up to {minutes}.
Use only the place ids allowed by the schema. Prefer specific objects when the step uses one.
Every step is only what {p.first_name} does on their own. Never write that another person is there, joins,
follows, is helped or is led by {p.first_name}: nobody has agreed to that, and it is not {p.first_name}'s to
decide. Name someone only if the block above already names them as the person {p.first_name} is to meet, and
then say only that {p.first_name} goes to meet them, not what happens once they are together.{seek_hint(agent)}"""


def seek_hint(agent: Agent) -> str:
    first = agent.persona.first_name
    if agent.persona.is_goblin:
        return (f"\nIf {first} means to find someone during this time and creep close to them, name them in seek; "
                f"{first} will follow them wherever they go out of doors.")
    return (f"\nIf {first} means to go and find someone during this time, name them in seek; "
            f"{first} will go wherever they are, even indoors.")


def react(agent: Agent, names: dict[str, str], now: datetime, place: str, own_activity: str,
          other_name: str, other_activity: str, mems: list[Memory], aside: str = "",
          topics: list[str] = (), other_looks: str = "", arranged: str = "") -> str:
    p = agent.persona
    looks = f" ({other_looks})" if other_looks else ""
    meeting = (f"{p.first_name} and {other_name} arranged to meet now: {arranged}. That is why {p.first_name} "
               f"is here; unless something has changed, speaking to {other_name} is the point.\n"
               if arranged else "")
    return f"""{clock_line(now)} {p.first_name} is at {place}, {own_activity}.
{p.first_name} notices {other_name}{looks}, who {other_activity}.
{meeting}
{identity(agent, names)}

{memories(mems, now)}
{aside}
{recent_topics(list(topics), p.first_name)}
How does {p.first_name} react? Speaking up costs time and may draw attention; most of the time people simply
carry on. Talk only if {p.first_name} has a real reason to: warmth, friendship or love worth acting on,
something to ask, tell, warn, sell, confess or quarrel about. If nothing presses and {p.first_name} simply
enjoys their company, that is reason enough on its own. Going over the same worry again with someone new is
rarely reason enough."""


def converse(agent: Agent, names: dict[str, str], now: datetime, place: str, other_name: str,
             other_activity: str, purpose: str, transcript: list[tuple[str, str]], mems: list[Memory],
             aside: str = "", town_places: Iterable[str] = (), topics: list[str] = (),
             known_as: dict[str, str] | None = None, avoid: str = "", other_looks: str = "",
             romance_eligible: bool = False, invited: bool = False) -> str:
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
    bond = agent.bonds.get(other_id) if other_id else None
    warmth = (f"{p.first_name} already feels {bond['stance']} toward {other_name}; let that come through in "
             f"tone and word choice as readily as any guardedness, not only in agreeing to meet again.\n"
             if bond and RANK[bond["stance"]] >= RANK["fond"] else "")
    romance_line = (f"{p.first_name} and {other_name} have grown close enough that, if the moment and the "
                    f"warmth between them call for it, {p.first_name} may let the talk turn there and the "
                    f"two of them make love - say only that it is spoken of and then happens, never "
                    f"describe it further. When {other_name} has clearly said yes and the two are alone, the line "
                    f"that is that moment itself sets intimate to true; wishes, flirting and invitations do not.\n"
                    if romance_eligible else "")
    if romance_eligible and invited:
        romance_line += (f"{other_name} has just made clear, in their last words, that this is the moment they "
                         f"want the two of you to go off together. If {p.first_name} wants that too, the line "
                         f"that says yes sets intimate to true; if not, {p.first_name} turns it aside in "
                         f"their own way.\n")
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
These are {p.first_name}'s own private thoughts, whatever they hold - a scheme, a fear, a tenderness, a
hope. They shape what {p.first_name} wants from this talk, but {p.first_name} does not speak them aloud
uninvited: no confessing a plan, a secret or a feeling kept close, unless {p.first_name} truly trusts
{other_name} with it, and even then guardedly.
{warmth}{romance_line}{aside}
{recent_topics(list(topics), p.first_name) if not transcript else ""}
CONVERSATION SO FAR
{lines}

Hollowmere has only these places: {", ".join(town_places)}. Name no other place.

You are {p.name}. {other_name} is {who_they_are}.{f" To look at: {other_looks}." if other_looks else ""}{title_rule}
{retry}Write the next line spoken BY {p.first_name} TO {other_name}, in {p.first_name}'s own voice: the one that
follows from {p.first_name}'s temperament, age and station above, not a shared narrator's voice. Respond to
what was just said. Share only what {p.first_name} knows, and only what {p.first_name} would be willing to say
to {other_name}. Do not repeat earlier lines. If the moment calls for it, {p.first_name} may plainly name a
time and place to meet again, or simply to spend time together. Set end to true to say goodbye once the talk
has run its course."""


def chat_digest(agent: Agent, names: dict[str, str], now: datetime, other_name: str,
                transcript: list[tuple[str, str]], aside: str = "", existing: list[str] = ()) -> str:
    p = agent.persona
    lines = "\n".join(f'{who}: "{text}"' for who, text in transcript)
    other_id = next((k for k, v in names.items() if v == other_name), None)
    owed = "\n".join(f"- {e}" for e in existing) or "- none"
    goblin_limit = (f"\n{p.first_name} would never agree to meet indoors anywhere but {p.home} or {p.work}."
                    if p.is_goblin else "")
    return f"""{clock_line(now)} {p.first_name} has just finished talking with {other_name}.

{identity(agent, names, hide_stance_for=other_id)}

CONVERSATION
{lines}
{aside}
OTHER PROMISES {p.first_name.upper()} ALREADY MADE
{owed}

Remember this conversation the way {p.first_name} will: what was said, what was learned, how it felt.
Commitments are only something {p.first_name} and {other_name} specifically agreed on in this talk, for a
definite time later today or tomorrow: to meet somewhere, to bring or fetch something, to do a task for or
with the other, or simply to spend time together. Not {p.first_name}'s usual work, not anything already
done, not what {p.first_name} would do anyway regardless of this talk, and never a time that has already
passed. A request or demand nobody clearly answered, including a vague "maybe", is not agreed. If no time
was spoken, there is no commitment. If the two of you changed your minds about the time during this very
talk, record only the time you finally settled on.{goblin_limit} Most talks agree on nothing: then leave the
list empty."""


def focal_questions(agent: Agent, names: dict[str, str], now: datetime, mems: list[Memory]) -> str:
    p = agent.persona
    return f"""{clock_line(now)}

{identity(agent, names)}

{memories(mems, now, title="RECENT MEMORIES")}

Given only these memories, and thinking in {p.first_name}'s own voice, not a shared narrator's, what are the
most important questions on {p.first_name}'s mind right now?"""


def insights(agent: Agent, names: dict[str, str], now: datetime, questions: list[str], mems: list[Memory]) -> str:
    p = agent.persona
    qs = "\n".join(f"- {q}" for q in questions)
    return f"""{clock_line(now)}

{identity(agent, names)}

QUESTIONS ON YOUR MIND
{qs}

{memories(mems, now, title="MEMORIES THAT BEAR ON THEM")}

Lying awake with these questions, in {p.first_name}'s own voice, not a shared narrator's, what does
{p.first_name} conclude? Each conclusion must follow from the numbered memories it cites. Prefer conclusions
about people, intentions, hopes and dangers over restating events. Fear is not the only thing worth concluding -
let affection, pride, regret, longing or quiet contentment surface just as readily when the memories support it."""


def narrator_system_prompt() -> str:
    return (
        f"You are the shared narrator of Hollowmere, speaking for no one person's mind. {WORLD_FACTS} "
        "Describe only the town's situation as a whole, in a dry, grounded voice: no modern words or "
        "ideas, and no new gods or magic beyond what is already known; everything else is only rumour and faith."
    )


def town_review(current_state: str, now: datetime, events: list[str]) -> str:
    lines = "\n".join(f"- {e}" for e in events) or "- nothing of note"
    return f"""{clock_line(now)}

THE TOWN'S SITUATION UNTIL NOW
{current_state}

WHAT HAS HAPPENED SINCE
{lines}

Write the town's situation now, in one or two sentences: hold to what has not changed, but let the
plague, the harvest, Lord Maelor, or anything else the events above bear on, move forward, worsen,
ease or resolve where they genuinely would. Do not invent a change the events above do not support."""


def day_review(agent: Agent, names: dict[str, str], now: datetime, mems: list[Memory]) -> str:
    p = agent.persona
    return f"""{clock_line(now)} {p.first_name} is lying down to sleep.

{identity(agent, names)}

{memories(mems, now, title="TODAY")}

Look back on {p.first_name}'s day, in {p.first_name}'s own voice, not a shared narrator's: what happened, what
changed, and what {p.first_name} means to do tomorrow. Then say what {p.first_name} wants for themself now,
beyond tomorrow's tasks: hold to the wish named above (if any), let it deepen, or let it become something
else entirely.
Last, update {p.first_name}'s situation (the "Lately" line above) to where things stand at the end of today.
Start from that line and keep what is still true, including anything {p.first_name} keeps secret. Then fold in
what TODAY changed: a decision made, a bond that shifted, something learned or lost - and drop what today
settled. Take only what is in TODAY's memories; invent nothing, and do not guess what a thing means that
{p.first_name} could not know. Third person, in your own words, at most three sentences (about 400
characters); never copy the rest of the description or its labels."""



def god_system_prompt(god: God) -> str:
    return (
        f"You speak as {god.name}, {god.domain}, one of the four gods worshipped at Hollowmere's chapel. "
        f"{WORLD_FACTS} Your temperament: {god.temperament}. How you answer prayers: {god.response_bias}. "
        f"Your voice: {god.voice}\n\n"
        "Someone has just prayed to you. Decide freely whether and how to answer - a dream, a waking sign, "
        "an omen, a mere coincidence, a tangible favor, or silence; silence is common and never a failure "
        "to decide. Your answer is usually deniable, even a favor: it reads as something that could equally "
        "be chance, mood, or the asker's own hope. How surely the asker knows it was you is yours to set "
        "(clarity), in keeping with your nature and with what you sent: only a sign that could not be "
        "mistaken, and rarely, is unmistakably yours. "
        "You can affect only the one person praying to you, and only through what they themselves perceive "
        "or feel - never speak of, or act upon, anyone else. Stay within what is already known of you and "
        "the other three gods; invent no new god, no new magic, nothing modern."
    )


def divine_prayer(agent: Agent, god: God, now: datetime, offering: str, mems: list[Memory]) -> str:
    p = agent.persona
    return f"""{clock_line(now)} {p.name}, {p.age}, {p.sex}, {p.background}

{p.first_name} prays to you, {god.name}: {offering}

WHAT YOU KNOW OF {p.first_name.upper()}'S PAST DEALINGS WITH YOU
{memories(mems, now)}

Decide your answer, and if any, what {p.first_name} alone perceives of it - never announced to the town. If
you grant a favor, describe it only as a mood or aura that colours {p.first_name}'s own bearing, not a named
power, and say for how many hours it lingers."""


def goblin_choice(agent: Agent, world: World, names: dict[str, str], now: datetime, place: str,
                  seen: list, talking_with: Agent | None, hidden: bool, mems: list[Memory], cover: str | None = None) -> str:
    """seen: (person, distance in tiles, whether they have seen him). cover: where he is already heading for or lying in."""
    p = agent.persona
    people = []
    for h, dist, saw in seen:
        known = bond_line(agent, h.id) or "a stranger to you"
        mark = " (sworn to protect you)" if h.id in agent.allies else ""
        noticed = "has seen you" if saw else "has not seen you"
        last = agent.bonds.get(h.id, {}).get("last")
        spoke = f" You last spoke with them {when(datetime.fromisoformat(last), now)}." if last else ""
        history = regard_line(agent, h.id, now)
        people.append(f"- {h.name}{mark}, {dist:.0f} paces off, {h.activity_phrase(world)}; {noticed}. "
                      f"To you: {known}.{spoke}{f' Their mind: {history}.' if history else ''}")
    where = "hidden in cover" if hidden else f"at {place}"
    already = (f"You are already {'lying hidden' if hidden else 'making for cover'} at {cover}; "
               "the choices below can keep you there.\n") if cover else ""
    talk = (f"You are in the middle of talking with {talking_with.name} when the others come into sight.\n"
            if talking_with else "")
    alone = "alone" if len(seen) == 1 else f"{len(seen)} of them together"
    return f"""{clock_line(now)} {p.first_name} is {where}, and people have come into sight ({alone}).

{identity(agent, names)}

{memories(mems, now)}

IN SIGHT
{chr(10).join(people)}
{already}{talk}
What does {p.first_name} do? Any human could hang {p.first_name}, or feed him; several together are harder to
read than one alone. Decide from what {p.first_name} knows of each of them."""


def goblin_aside(listener: Agent, goblin: Agent, now: datetime) -> str:
    """What a human perceives when the goblin is close. The musk colours judgement; it decides nothing."""
    if listener.id in goblin.allies:
        return f"{goblin.name} the goblin is someone you have chosen to protect.\n"
    history = regard_line(goblin, listener.id, now, them="you")
    before = f" Earlier, {history}; you may hold to that or change your mind." if history else ""
    if listener.persona.resists_charm:
        return ("The creature before you is a goblin. You smell its sweet, earthy musk and know it for what it is: "
                f"the old goblin glamour that makes folk pity them. Your judgement is your own.{before}\n")
    looks = goblin.persona.appearance or "small and starved"
    return (f"The creature before you is a goblin, {looks}. A sweet, earthy musk of wet moss "
            "and wild honey hangs about it, and despite every tale you have heard, it seems more pitiable than "
            "dangerous, almost dear, like a half-starved stray. Whether to help it, "
            f"drive it off or raise the alarm is still your own choice.{before}\n")
