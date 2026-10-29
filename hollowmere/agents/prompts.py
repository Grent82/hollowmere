"""Prompt construction.

All prompts share one system prompt (the agent's voice and the world's tone)
and are assembled from the same blocks: identity, situation, memories, places.
The original kept ~40 near-duplicate text templates with positional !<INPUT n>!
slots; here each prompt is a small function and the shared parts are written once.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import TYPE_CHECKING, Iterable

from hollowmere.agents.gods import GOD_BY_ID, God
from hollowmere.agents.persona import DATA, RANK, Persona, first_name_of
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
        + (f"Since the story began: {'; '.join(f['text'].rstrip('.') for f in agent.history)}.\n"
           if agent.history else "")
        + f"Lately: {agent.situation or p.currently}\nHabits: {p.lifestyle}\n"
    )
    if agent.states:
        text += f"Right now: {', '.join(s['text'] for s in agent.states)}.\n"
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
        text += f"Let it colour how {p.first_name} speaks and chooses while it lasts.\n"
    if p.is_goblin:
        friends = ", ".join(names.get(a, a) for a in sorted(agent.allies)) or "no one yet"
        kind = ", ".join(f"{names.get(k, k)} ({REGARD_WORDS[r['step']]})" for k, r in sorted(agent.regard.items())
                         if r["step"] in ("tolerate", "feed")) or "no one yet"
        text += (f"Humans who have sworn to protect you: {friends}. Humans who have softened toward you: {kind}. "
                 "Anyone else could give you away; how far to trust each of them is your own judgement.\n")
    return text + f"People you know:\n{rel or '- no one well'}"


def source_label(m: Memory, names: dict[str, str]) -> str:
    """How the one who remembers came by it, as they would put it; nothing for a memory from before it was kept."""
    s = m.source
    if s == "saw":
        return "you saw it"
    if s == "lived":
        return "you lived it"
    if s.startswith("told:"):
        who = s.split(":", 1)[1]
        return f"{first_name_of(names.get(who, who))} told you"
    if s == "word":
        return "word going round"
    if s == "own":
        return "your own thought"
    if s == "divine":
        return "a sign from beyond"
    return ""


def memories(mems: Iterable[Memory], now: datetime, title: str = "WHAT YOU REMEMBER",
             names: dict[str, str] | None = None) -> str:
    # Observations are kept with their clock time no matter how old: knowing someone was at the
    # tavern "3 days ago" is useless for guessing where they'll be today, but "3 days ago at 19:00" lets
    # a pattern emerge across many such sightings. Thoughts and reflections don't need that precision.
    def stamp(m: Memory) -> str:
        label = source_label(m, names) if names is not None else ""
        return when(m.created, now, keep_time=m.kind == 'observation') + (f"; {label}" if label else "")
    lines = [f"[{m.id}] ({stamp(m)}) {m.text}" for m in mems]
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


def infant_line(agent: Agent) -> str:
    """A baby has no words: what it 'says' is the sound or small act itself, and it does not explain itself."""
    p = agent.persona
    if not p.infant:
        return ""
    return (f"{p.first_name} is only days old and has no words. What {p.first_name} says is only what a baby does: a "
            f"cry, a whimper, a coo, rooting for the breast, a grip on a finger. Write it as that sound or act "
            f"in a few words, for example \"a thin, hungry wail\", never as a sentence and never about being "
            f"unable to speak.\n")


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

{memories(known, now, title="WHAT YOU HAVE SEEN AND HEARD", names=names)}
Passing things on: what {p.first_name} says about a third person is a claim, not the truth. {p.first_name} may
pass it on, keep it back, bend it or make more of it, as {p.first_name}'s temperament and station would, depending on
how much {p.first_name} trusts or likes {other_name} and how {p.first_name} feels about the one spoken of. What
{p.first_name} only heard from someone, {p.first_name} says so ("Ros says ..."); what {p.first_name} saw, they may say
plainly.

{memories(private, now, title="YOUR PRIVATE THOUGHTS")}
These are {p.first_name}'s own private thoughts, whatever they hold - a scheme, a fear, a tenderness, a
hope. They shape what {p.first_name} wants from this talk, but {p.first_name} does not speak them aloud
uninvited: no confessing a plan, a secret or a feeling kept close, unless {p.first_name} truly trusts
{other_name} with it, and even then guardedly.
{warmth}{infant_line(agent)}{child_line(agent)}{romance_line}{aside}
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


def agreement_system_prompt() -> str:
    return "You read a conversation and report, as a neutral listener, what the speakers agreed."


def agreement(first_a: str, first_b: str, now: datetime, transcript: list[tuple[str, str]],
              places: dict[str, str]) -> str:
    lines = "\n".join(f'{who}: "{text}"' for who, text in transcript)
    where = "\n".join(f"- {pid}: {name}" for pid, name in places.items())
    return f"""{clock_line(now)} {first_a} and {first_b} have just finished talking.

CONVERSATION
{lines}

PLACES
{where}

Decide whether {first_a} and {first_b} agreed in this talk to meet or spend time together somewhere at a definite
time later today or tomorrow. Both must have said yes, or offered it themselves. Not their usual work, not
anything already done, not what they would do anyway, not a threat, an order nobody accepted, a vague "maybe",
or a time that has already passed. If no time was spoken, there is no meeting. Most talks agree on nothing: then
leave the list empty.

The activity is one phrase that both of them read as their own plan. Name the place for what it is ("visit the
nest together"), never as "home", "my" or "your" place: it is home to only one of them."""


def leaving_system_prompt() -> str:
    return "You read a conversation and report, as a neutral listener, what the speakers settled."


def leaving(first_a: str, first_b: str, now: datetime, transcript: list[tuple[str, str]]) -> str:
    lines = "\n".join(f'{who}: "{text}"' for who, text in transcript)
    return f"""{clock_line(now)} {first_a} and {first_b} have just finished talking.

CONVERSATION
{lines}

Decide whether in this talk either of them settled on leaving Hollowmere itself: setting out from the town
onto the road or into the wild, to be gone for days. Someone must say plainly that they are leaving, running away
from, or going out of the town (through a gate, onto the road, away for good), and, if they mean to take the other
along, the other agrees. Not a walk to meet or find someone, or to a place in or beside the town (the riverbank, the
reeds, the mill, the Hollow Keep, the burrow, the herb-wife's hut, the old stones), nor slipping past the Watch to
get there, hiding for a night, a wish, a complaint, a warning to flee, an offer someone refused, a plan only
talked over, a visit or going home.
Most talks settle nothing: then leave the list empty."""


def moving(first_a: str, first_b: str, now: datetime, transcript: list[tuple[str, str]]) -> str:
    lines = "\n".join(f'{who}: "{text}"' for who, text in transcript)
    return f"""{clock_line(now)} {first_a} and {first_b} have just finished talking.

CONVERSATION
{lines}

Decide whether in this talk one of them settled on moving into the other's home to live there for good, and the
other agreed. Not a visit, a night's shelter, a meeting at someone's home, a wish, an offer nobody accepted, or
staying a little longer. Most talks settle nothing: then leave the list empty."""


def child_line(agent: Agent) -> str:
    """A small child speaks in a few plain words, knows its parents' world only as a child does."""
    p = agent.persona
    if not p.child:
        return ""
    return (f"{p.first_name} is a small child, about {p.age} years old. {p.first_name} speaks in short, simple "
            f"sentences of a few words, is curious, asks why, and repeats what grown people say. {p.first_name} "
            f"knows nothing of coins, keys, the Watch, plots, the plague or what grown people mean, and does not "
            f"guess at them; the child stays close to whoever looks after it.\n")


def child_growth(child: Agent, mother: Persona, father: Persona, species: str, home_name: str = "") -> str:
    p = child.persona

    def who(q: Persona) -> str:
        return f"{q.name} ({q.species}). Temperament: {q.traits}." + (f" Looks: {q.appearance}" if q.appearance else "")
    facts = "; ".join(f["text"] for f in child.history) or "(none)"
    return f"""{p.name} was born as {'a goblin' if species == 'goblin' else 'a human'} a few days ago and is now growing
into a small child.

MOTHER
{who(mother)}

FATHER
{who(father)}

THE NEWBORN SO FAR
Looks: {p.appearance or "not yet described"}
{p.background}
What has happened to it: {facts}
{f"It lives at {home_name}." if home_name else ""}

The child is now about 3 years old: no longer an infant, it walks unsteadily and says a few words.
Say which of its parents' own traits the child shows (only from the choices given), how it looks and moves now
(write only what is new, such as size, hair and how it moves; do not repeat the earlier description), what it is and
does these days, and what a small child's day is like in its parents' care. Invent nothing new about who it is: it is
a child of these two, and only that. Set its day at its home and name no other place and no one but its parents."""


def infant_mind(agent: Agent) -> str:
    """What a baby can wonder and conclude: the body and the few people it knows, never the grown world's affairs."""
    p = agent.persona
    if p.child:
        return (f"\n{p.first_name} is a small child. All {p.first_name} can wonder or conclude is what a child feels "
                f"and notices: who is near, what is warm or hungry or frightening, what is played with. Say it in a "
                f"few plain words. {p.first_name} knows nothing of coins, keys, the Watch, plots, the plague or what "
                f"grown people mean, and does not guess at them.")
    if not p.infant:
        return ""
    return (f"\n{p.first_name} is only days old. All {p.first_name} can wonder or conclude is what a baby can feel: "
            f"warm or cold, hungry or fed, a voice or smell it knows, the dark, being held or left. Say it in a few "
            f"plain words. {p.first_name} knows nothing of coins, keys, the Watch, plots, the plague or what grown "
            f"people mean, and does not guess at them.")


def focal_questions(agent: Agent, names: dict[str, str], now: datetime, mems: list[Memory]) -> str:
    p = agent.persona
    return f"""{clock_line(now)}

{identity(agent, names)}

{memories(mems, now, title="RECENT MEMORIES")}

Given only these memories, and thinking in {p.first_name}'s own voice, not a shared narrator's, what are the
most important questions on {p.first_name}'s mind right now?{infant_mind(agent)}"""


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
let affection, pride, regret, longing or quiet contentment surface just as readily when the memories support it.{infant_mind(agent)}"""


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


def family_line(agent: Agent, names: dict[str, str]) -> str:
    """What a parent must not lose sight of while looking back on a day: a child in the house, or one on the way."""
    p = agent.persona
    children = [f"{names.get(k, k)} ({b['kin'].removeprefix('your ')})" for k, b in agent.bonds.items() if b.get("kin")]
    if not children and not agent.condition:
        return ""
    who = f"{p.first_name} has a child to look after: {', '.join(children)}." if children else \
        f"{p.first_name} is with child."
    return (f"\n{who} A baby cannot look after itself, and this weighs on {p.first_name}'s days whatever else fills "
            f"them: say in the situation and in tomorrow's plans how {p.first_name} means to carry that.\n")


def child_draft_system_prompt() -> str:
    return ("You write the newborn of a mother and a father in the town of Hollowmere, as a neutral storyteller. "
            f"{WORLD_FACTS}")


def child_draft(mother: Persona, father: Persona, species: str, sex: str, born_on: date, names: list[str],
                refused: str = "", keep_name: str = "") -> str:
    def who(p: Persona) -> str:
        return (f"{p.name}, {p.age}, {p.species}. Temperament: {p.traits}. "
                + (f"Looks: {p.appearance}. " if p.appearance else "") + p.background)
    kind = "a goblin" if species == "goblin" else "a human"
    surname = "" if species == "goblin" else f" A human child takes its father's family name ({father.name.split()[-1]})."
    return f"""A child is born on {born_on.isoformat()}: {'a son' if sex == 'male' else 'a daughter'}, {kind}.

MOTHER
{who(mother)}

FATHER
{who(father)}

FIRST NAMES ALREADY IN TOWN (do not reuse any of them)
{", ".join(names)}
{f"The name {refused} is taken: choose another." if refused else ""}{f"The child is already named {keep_name}: repeat that name as the name." if keep_name else ""}
Write the newborn: its name (a short, plain medieval name that fits its family and kind; a goblin child gets a
single short goblin name).{surname} Then what others see of it: {"small, grey-green and goblin" if species == "goblin" else "a human baby"},
with one or two features from its parents. Then one plain sentence of background: whose child it is, and nothing
else. Invent no events, no place, no midwife or other people present, no weather, no gifts and no temperament: it is
only days old."""


def daily_event_system_prompt() -> str:
    return (f"You are the shared narrator of Hollowmere. {WORLD_FACTS} You decide what small thing the day brings to "
            "the town, in a dry, grounded voice: no modern words or ideas, no new gods or magic beyond what is already "
            "known. Small only: nobody dies, is hurt, arrested or falls in love because of it.")


def daily_event(now: datetime, town_state: str, recent: list[str], town_events: list[str], gods: list[str]) -> str:
    earlier = "\n".join(f"- {e}" for e in recent) or "- nothing yet"
    lately = "\n".join(f"- {e}" for e in town_events[-12:]) or "- nothing of note"
    gods_text = "\n".join(gods) or "(no god may stir today)"
    return f"""{clock_line(now)} A new day begins in Hollowmere.

THE TOWN'S SITUATION
{town_state}

WHAT HAS HAPPENED IN TOWN LATELY
{lately}

SMALL EVENTS OF THE LAST DAYS (do not repeat any)
{earlier}

GODS WHO MAY STIR (only these; a god acts only through what anyone there can see or hear, never on one person)
{gods_text}

Decide the day's small event, or that nothing special happens. Make it everyday and believable: the weather, a
market or traders' day, a beast or bird, a small mishap, news from beyond the town. Nothing eerie or uncanny
unless a god below is behind it. It fits the town's situation, the season and what has happened lately, and differs
from the events of the last days. Say where it happens (a place of the town, or the whole town) and at what hour.
An omen is only the sign of a god listed below, suits that god's nature and is easy to take for chance; it never
names the god. Nobody is named in it."""


def life_facts_system_prompt() -> str:
    return ("You keep the lasting facts of one life in the town of Hollowmere, as a neutral recorder. You invent "
            "nothing: a fact rests on what the person's own memories show.")


def life_facts(agent: Agent, names: dict[str, str], now: datetime, mems: list[Memory]) -> str:
    p = agent.persona
    held = "\n".join(f"- {f['text']}" for f in agent.history) or "(none yet)"
    return f"""{clock_line(now)} {p.first_name}'s day is over.

{p.name}, {p.age}, {p.sex}. {p.traits}. {p.background}

LASTING FACTS HELD SO FAR
{held}

{memories(mems, now, title="TODAY'S IMPORTANT MEMORIES")}

Say whether today made something true of {p.first_name} that will still be true long from now: a birth, a break
with someone, a jailing, a flight, a lover, a secret out, a death, a new home or work. Only what is settled and
certain to {p.first_name}, never what someone suspects, fears, jokes about or has only heard. Not a mood, a worry or
any passing state (anything gone in a week), and not a condition that will end (pregnant, injured, hiding): wait
for how it ends (the birth, the healing). Not a plan or an intention (leaving, going to): only what has already
happened and stays so. Not what the description or the held facts already say, such as a trade, a rank, an age or
a family tie.
Answer with the list as it stands: repeat each held fact in the same words, and add at most one new fact (third
person, one plain sentence), only if the memories above show it, with a few words copied from them as evidence.
If the day makes a held fact obsolete, leave out a held fact (just that one) when you add the new one. Six facts at
most. Most days nothing qualifies: then repeat the list unchanged and leave the evidence empty."""


def states_block(agent: Agent, now: datetime) -> str:
    """The passing states held now, with their age, and what the review may do with them."""
    p = agent.persona
    held = "\n".join(f"- {s['text']} ({days} day{'s' * (days != 1)})" for s, days in agent.state_ages(now)) \
        or "(none)"
    return f"""
HOW {p.first_name.upper()} FEELS, PASSING STATES (not temperament)
{held}
"""


def day_review(agent: Agent, names: dict[str, str], now: datetime, mems: list[Memory]) -> str:
    p = agent.persona
    return f"""{clock_line(now)} {p.first_name} is lying down to sleep.

{identity(agent, names)}

{memories(mems, now, title="TODAY")}
{family_line(agent, names)}{states_block(agent, now)}
Look back on {p.first_name}'s day, in {p.first_name}'s own voice, not a shared narrator's: what happened, what
changed, and what {p.first_name} means to do tomorrow. Then say what {p.first_name} wants for themself now,
beyond tomorrow's tasks: hold to the wish named above (if any), let it deepen, or let it become something
else entirely.
Last, update {p.first_name}'s situation (the "Lately" line above) to where things stand at the end of today.
Start from that line and keep what is still true, including anything {p.first_name} keeps secret. Then fold in
what TODAY changed: a decision made, a bond that shifted, something learned or lost - and drop what today
settled. Take only what is in TODAY's memories; invent nothing, and do not guess what a thing means that
{p.first_name} could not know. Third person, in your own words, at most three sentences (about 400
characters); never copy the rest of the description or its labels.
And name how {p.first_name} is as the day ends, in the passing states list: repeat in the same words a state that
is still as true as it was, leave out one that has eased or ended - grief, fear and weariness do soften with
time unless something fresh keeps them alive - and add one that today brought. A state can be light as well as
dark: relieved, hopeful, content, rested, proud, in love, amused, at peace. A good day deserves a good state.
At most three; the list may be empty."""



def god_system_prompt(god: God) -> str:
    return (
        f"You speak as {god.name}, {god.domain}, one of the four gods worshipped at Hollowmere's chapel. "
        f"{WORLD_FACTS} Your temperament: {god.temperament}. How you answer prayers: {god.response_bias}. "
        f"Your voice: {god.voice}\n\n"
        "Someone has just prayed to you. Decide freely how to answer, for you always do - a dream, a waking sign, "
        "an omen, a mere coincidence, or a tangible favor. Your answer is usually deniable, even a favor: it reads as something that could equally "
        "be chance, mood, or the asker's own hope. How surely the asker knows it was you is yours to set "
        "(clarity), in keeping with your nature and with what you sent: only a sign that could not be "
        "mistaken, and rarely, is unmistakably yours. "
        "You can affect only the one person praying to you, and only through what they themselves perceive "
        "or feel - never speak of, or act upon, anyone else. Stay within what is already known of you and "
        "the other three gods; invent no new god, no new magic, nothing modern."
    )


def divine_prayer(agent: Agent, god: God, now: datetime, offering: str, mems: list[Memory],
                  recent: list[Memory] | None = None) -> str:
    p = agent.persona
    lately = (f"WHAT HAS WEIGHED ON {p.first_name.upper()} LATELY\n{memories(recent, now)}\n\n") if recent else ""
    return f"""{clock_line(now)} {p.name}, {p.age}, {p.sex}, {p.background}

HOW THINGS STAND FOR {p.first_name.upper()}
Temperament: {p.traits}.
{f"Right now: {', '.join(s['text'] for s in agent.states)}." + chr(10) if agent.states else ""}Lately: {agent.situation or p.currently}
{p.first_name} wants, for themself: {agent.drive or "nothing they could name"}
The town: {agent.town_state or DEFAULT_TOWN_STATE}

{lately}{p.first_name} prays to you, {god.name}: {offering}

WHAT YOU KNOW OF {p.first_name.upper()}'S PAST DEALINGS WITH YOU
{memories(mems, now)}

Decide your answer, and if any, what {p.first_name} alone perceives of it - never announced to the town. If
you grant a favor, describe it only as a mood or aura that colours {p.first_name}'s own bearing, not a named
power, and say for how many hours it lingers.
Your answer may also ease one of the passing states under "Right now" (name it exactly in `eases`) and/or leave
a new one in {p.first_name} (`lingers`), which lasts until {p.first_name}'s own days wear it away. Do so only
when your nature and what you sent truly would."""


def baby_home(agent: Agent, names: dict[str, str], now: datetime, child: Agent, father: Agent,
              mems: list[Memory], home_name: str, den_name: str) -> str:
    p, f = agent.persona, father.persona
    return f"""{clock_line(now)} {p.first_name} has just borne a child, {child.persona.first_name}, {f.name}'s: a goblin,
like its father. It lies in {p.first_name}'s arms, newborn.

{identity(agent, names)}

{memories(mems, now)}

WHERE THE CHILD COULD LIVE
- Keep it: the child lives in {p.first_name}'s own home, {home_name}, wherever {p.first_name} may be staying now.
  Anyone who comes there might see it, and {p.first_name} could not take it out into the town without everyone
  seeing what it is.
- With its father: {f.name} lives at {den_name}, a den in the wild, where goblins are at home and {p.first_name}
  may visit; the child would grow up among its own kind.

Decide where the child will live, as {p.first_name} would."""


def wishes_system_prompt(god: God) -> str:
    return (
        f"You speak as {god.name}, {god.domain}, one of the four gods worshipped at Hollowmere's chapel. "
        f"{WORLD_FACTS} Your temperament: {god.temperament}. Your voice: {god.voice}\n\n"
        "Mortals pray to you and you answer from afar, only through dreams, signs, omens and moods. Say what you "
        "would most like to do among them and in the world, if you could, and cannot yet."
    )


def god_wishes(god: God, now: datetime, prayers: list[dict], names: dict[str, str], existing: list[str]) -> str:
    lines = "\n".join(f"- {p['time'][:16]} {names.get(p['agent'], p['agent'])} ({p['form']}): {p['text']}"
                      for p in prayers)
    have = "\n".join(existing) or "(none yet)"
    return f"""{clock_line(now)} The day is ending.

PRAYERS YOU HAVE ANSWERED (oldest first)
{lines}

WISHES YOU HAVE ALREADY NAMED
{have}

From these prayers and what they show of the mortals, say what you would like to do among them or in the world
that you cannot yet: not what you did, but what you long for. Only wishes not already named; none if there is
nothing new."""


def world_ideas_system_prompt() -> str:
    return ("You read one person's day in the town of Hollowmere and report, as a neutral listener, how they "
            "wish to deal with the world around them. You invent nothing: every idea rests on words from their day.")


def world_ideas(agent: Agent, now: datetime, plan: list[str], mems: list[Memory], objects: list[str],
                existing: list[str], places: dict[str, str]) -> str:
    p = agent.persona
    have = "\n".join(existing) or "(none yet)"
    return f"""{clock_line(now)} {p.name}'s day is over.

THE DAY'S PLAN
{chr(10).join(f"- {x}" for x in plan) or "(none)"}

{memories(mems, now, title="TODAY'S MEMORIES")}

WHAT THE GAME HAS AT PRESENT (every kind of object in the town)
{", ".join(objects)}

THE TOWN'S PLACES (id: name)
{"; ".join(f"{k}: {v}" for k, v in places.items())}

IDEAS ALREADY COLLECTED
{have}

Find in today's plan and memories only physical things you could build or place in the town: tools, furniture,
goods, containers, buildings, rooms, paths. Three kinds of idea:
- change: {p.first_name} wants to alter a building, place or thing of the town.
- missing_item: a physical thing {p.first_name} handles or mentions that is NOT in the list of what the game has
  (check the list; skip it if something close is there).
- item_use: how {p.first_name} used or would use such a missing thing, in one short sentence.
Never a person, a god, an animal, a feeling, a smell, a sound, a saying or a metaphor. Name each in one to three words
and say in which of the places it belongs. Copy a few words from the plan or memories as evidence. Only ideas not already collected; none if there is nothing new."""


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
