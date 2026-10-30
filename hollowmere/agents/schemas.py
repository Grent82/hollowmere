"""Structured outputs for every cognitive call.

Field order matters: models generate fields in schema order, so reasoning-like
fields ("thought", "summary") come before decisions that should follow from them.
Field descriptions are part of the schema the model sees and double as
per-field instructions.

Schemas that mention places are built per agent with the agent's known place
ids as an enum, so constrained decoding cannot produce a place that does not
exist (the original needed three sequential LLM calls and fuzzy matching for this).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import BaseModel, Field, create_model

from hollowmere.agents.gods import GOD_IDS
from hollowmere.agents.persona import Stance

# How a human means to treat the goblin after a talk; a step can rise or fall from one talk to the next.
GoblinRegard = Literal["drive off", "undecided", "tolerate", "feed", "protect"]

PrayTo = Literal[GOD_IDS + ("none",)]
DivineForm = Literal["dream", "sign", "omen", "coincidence", "buff"]
DivineClarity = Literal["none", "doubt", "suspected", "certain"]

HHMM = r"^([01][0-9]|2[0-3]):[0-5][0-9]$"  # [0-9], not \d: llama.cpp's grammar lets anything through for \d


class FocalQuestions(BaseModel):
    questions: list[str] = Field(
        min_length=1, max_length=3,
        description="the most important questions on your mind given these memories, in your own words")


class Insight(BaseModel):
    text: str = Field(description="one conclusion you draw, in first person, one sentence")
    evidence: list[int] = Field(description="numbers of the memories this conclusion rests on")
    importance: int = Field(ge=1, le=10, description="how much this matters to you, 1 trivial to 10 life-changing")


class Insights(BaseModel):
    insights: list[Insight] = Field(min_length=1, max_length=4)


class Reaction(BaseModel):
    thought: str = Field(description="your private reaction to what you see, first person, one sentence")
    importance: int = Field(ge=1, le=10, description="how notable this sight is to you, 1 mundane to 10 shocking")
    talk: bool = Field(description="true if you go and speak to them now")
    reason: str = Field(description="if talking, what you want from the conversation; otherwise empty")


class Utterance(BaseModel):
    says: str = Field(description="what you say aloud, one to three sentences, in your own voice")
    end: bool = Field(description="true if this line ends the conversation (a farewell, walking off, nothing more to say)")


def utterance_schema(romance_eligible: bool = False) -> type[Utterance]:
    """What a speaker says; a pair that may make love is also asked, line by line, whether this line is the moment.
    Reading consent out of a transcript afterwards never worked (0 of 43 talks): the speaker knows when it is."""
    if not romance_eligible:
        return Utterance
    return create_model("RomanticUtterance", __base__=Utterance, intimate=(bool, Field(
        description="true only if this very line is the moment you and they go off together and give yourselves "
                    "to each other, because you both clearly want it and are alone. Flirting, longing, a "
                    "compliment or an invitation is not yet it: false by default, and never describe it")))


class TownState(BaseModel):
    text: str = Field(description="the town's situation now, in the shared narrator's voice, one or two "
                                  "sentences - not any one person's thought")


class DayReview(BaseModel):
    summary: str = Field(description="how your day went and what mattered, first person, two to three sentences")
    importance: int = Field(ge=1, le=10)
    intentions: list[str] = Field(max_length=3, description="things you mean to do tomorrow, first person")
    drive: str = Field(description="what you want for yourself now, beyond tomorrow's tasks - hold to what you "
                                   "wanted before, let it deepen, or let it become something else entirely, "
                                   "first person, one or two sentences")
    situation: str = Field(default="", description="where your life stands now, third person by name, one to "
                                                   "three sentences: the previous situation carried forward, "
                                                   "changed only where today changed it")
    states: list[Annotated[str, Field(max_length=60)]] = Field(max_length=6, description=(
        "the complete list of passing states of body or mind as the day ends, each a short phrase, lower case, "
        "without the name, e.g. \"grieving for her child\": repeat a state still true in exactly the same words, "
        "leave out one that has passed, add one today brought; at most three; may be empty"))


class GodWish(BaseModel):
    wish: str = Field(description="what you would most like to do among mortals or in the world but cannot yet, "
                                  "in your own voice, one or two sentences")
    towards: str = Field(description="whom it is about: a mortal's name, or \"the world\" or \"mortals\"")
    why: str = Field(description="one short sentence on what in the prayers stirred it")


class GodWishes(BaseModel):
    wishes: list[GodWish] = Field(max_length=3, description="only wishes not already on the list; may be empty")


@lru_cache(maxsize=8)
def world_ideas_schema(areas: tuple[str, ...]) -> type[BaseModel]:
    """The ideas a day holds for the world, each tied to one of the town's places."""
    idea = create_model(
        "WorldIdea",
        kind=(Literal["change", "missing_item", "item_use"], Field(description=(
            "change = they would like to alter a building, place or thing in the world; missing_item = something "
            "is mentioned that the game does not have; item_use = how they used or would use such a thing"))),
        item=(str, Field(max_length=40, description="the thing in one to three words, e.g. \"hopper\"")),
        where=(Literal[areas], Field(description="the place of the town where it belongs or would be used")),  # type: ignore[valid-type]
        what=(str, Field(description="the idea in one short sentence, plain words")),
        evidence=(str, Field(description="a few words copied exactly from today's memories or plan that show it")),
    )
    return create_model(
        "WorldIdeas",
        ideas=(list[idea], Field(max_length=4, description="only ideas not already collected; may be empty")))  # type: ignore[valid-type]


class ChildDraft(BaseModel):
    name: str = Field(max_length=40, description="the newborn's name, fitting its family and kind")
    appearance: str = Field(max_length=600, description="what others see of the baby, one or two short sentences, "
                                                         "with features from its parents")
    background: str = Field(max_length=500, description="whose child it is, in one plain sentence, third person; "
                                                         "no place, no people present, no weather")


@lru_cache(maxsize=64)
def child_growth_schema(pool: tuple[str, ...], low: int, high: int) -> type[BaseModel]:
    """A newborn written on as a child: its traits can only be words its parents already have."""
    return create_model(
        "ChildGrowth",
        traits=(list[Literal[pool]], Field(min_length=low, max_length=high, description=(  # type: ignore[valid-type]
            f"{low} to {high} of the parents' own traits that the child shows, the ones that fit it best"))),
        look_now=(str, Field(max_length=200, description="one short sentence on how the child looks and moves now; "
                                                         "only what is new, not the earlier description")),
        currently=(str, Field(max_length=200, description="one sentence: what the small child is, and does, these days")),
        lifestyle=(str, Field(max_length=250, description="one sentence on a small child's day in its parents' care")),
    )


@lru_cache(maxsize=16)
def daily_event_schema(areas: tuple[str, ...], gods: tuple[str, ...]) -> type[BaseModel]:
    """One small event for the day, at a place of the town (or for all of it), maybe a sign of a god."""
    return create_model(
        "DailyEvent",
        quiet=(bool, Field(description="true if nothing worth noting happens today")),
        kind=(Literal[(*(("weather", "market", "animal", "accident", "news") + (("omen",) if gods else ())),)],
              Field(description=(  # type: ignore[valid-type]
                  "weather; a market or traders' day; a beast or bird; a small mishap nobody is hurt by; a piece of "
                  "news from beyond the town" + ("; an omen, only as a sign of a god" if gods else "")))),
        text=(str, Field(max_length=300, description=(
            "what anyone there can see or hear, one or two plain sentences, no one named, never naming a god"))),
        where=(Literal[(*areas, "town")], Field(description=(  # type: ignore[valid-type]
            "the place of the town where it happens, or \"town\" if it is for all of it"))),
        hour=(int, Field(ge=5, le=21, description="the hour of the day it happens, 5 to 21")),
        god=(Literal[(*gods, "none")], Field(description=(  # type: ignore[valid-type]
            "a god whose doing this is, only from the choices; none for an ordinary event"))),
        clarity=(Literal["none", "doubt", "suspected"], Field(description=(
            "how plainly it reads as that god's doing: none (chance), doubt, suspected; none when there is no god"))),
        importance=(int, Field(ge=2, le=5, description="how much it weighs on those who see it, 2 faint to 5 striking")),
    )


class LifeFacts(BaseModel):
    facts: list[Annotated[str, Field(max_length=200)]] = Field(max_length=8, description=(
        "the lasting facts of this life as they stand: each held fact repeated in exactly the same words, plus at "
        "most one new one (third person, one plain sentence); six at most, so merge or leave one out to make room"))
    evidence: str = Field(description="a few words copied exactly from today's memories that show the new fact; "
                                      "empty if there is none")


class BabyHome(BaseModel):
    """Where a mother puts the goblin child she has just borne."""
    reason: str = Field(description="one or two sentences, in the mother's own voice, on what she fears and wants "
                                    "for the child")
    where: Literal["keep", "father"] = Field(description=(
        "keep = the child lives in the mother's own home; father = she gives the child to its father, to grow up "
        "in his home"))


class DivineAnswer(BaseModel):
    form: DivineForm = Field(description="how you answer: a dream, a waking sign, an omen, a mere "
                                         "coincidence, a tangible favor (buff) - choose freely; you "
                                         "always answer")
    text: str = Field(description="what the praying mortal perceives, in your own voice, kept deniable - "
                                  "something that could equally be chance, mood, or their own hope, never a "
                                  "flat statement that you have acted")
    clarity: DivineClarity = Field(
        default="none",
        description="how surely the mortal recognises this as your answer: none - it seems chance, mood or "
                    "an ordinary dream; doubt - a passing 'was that...?' they cannot hold on to; suspected - "
                    "they half believe you answered, without certainty; certain - unmistakably you, rare. "
                    "It must fit form and text: wording that could only be chance is never certain. Choose "
                    "it from your own nature")
    effect: str = Field(description="if form is buff, the mood or aura it leaves on them, a few words; "
                                     "otherwise empty")
    duration_hours: int = Field(ge=0, le=72, description="if form is buff, how many hours it lingers; "
                                                          "otherwise 0")
    importance: int = Field(ge=1, le=10, description="how much this should weigh on the mortal who "
                                                      "receives it, 1 faint to 10 unforgettable")
    eases: str = Field(default="", description="a passing state the mortal now carries, copied exactly as listed "
                                               "under 'Right now', that your answer lifts from them; otherwise empty")
    lingers: str = Field(default="", description="a short phrase for a passing state your answer leaves in them, "
                                                 "lower case, without their name, e.g. \"a quiet certainty that "
                                                 "someone heard\"; otherwise empty")


@lru_cache(maxsize=256)
def day_plan_schema(places: tuple[str, ...]) -> type[BaseModel]:
    place_t = Literal[places]  # type: ignore[valid-type]
    block = create_model(
        "PlanBlock",
        start=(str, Field(pattern=HHMM, description="24-hour start time, HH:MM")),
        end=(str, Field(pattern=HHMM, description="24-hour end time, HH:MM")),
        activity=(str, Field(description="what you do, a short present-tense phrase without your name")),
        pray_to=(PrayTo, Field(description="the one god this block is a prayer to, or 'none' for every "
                                           "ordinary block - most blocks are 'none'")),
        place=(place_t, Field(description="where, one id from the PLACES list")),
    )
    return create_model(
        "DayPlan",
        morning_thought=(str, Field(description="what is on your mind as you wake, first person, one or two sentences")),
        wake_up=(str, Field(pattern=HHMM)),
        blocks=(list[block], Field(min_length=3, max_length=10)),  # type: ignore[valid-type]
    )


@lru_cache(maxsize=256)
def decomposition_schema(places: tuple[str, ...], seekable: tuple[str, ...] = ()) -> type[BaseModel]:
    """seekable: the people you might go looking for during this stretch of the day."""
    place_t = Literal[places]  # type: ignore[valid-type]
    step = create_model(
        "Step",
        activity=(str, Field(description="a concrete action, short present-tense phrase without your name")),
        minutes=(int, Field(ge=5, le=240)),
        place=(place_t, Field(description="where exactly, one id from the list")),
        emoji=(str, Field(max_length=8, description="one or two emoji that picture the action")),
    )
    fields: dict = {}
    if seekable:
        fields["seek"] = (Literal[seekable + ("nobody",)], Field(  # type: ignore[valid-type]
            description="someone you mean to find during this time, or nobody"))
    return create_model("Decomposition", **fields,
                        steps=(list[step], Field(min_length=1, max_length=8)))  # type: ignore[valid-type]


# The parts of a day a talk can name; there are no clocks in Hollowmere, and asking for HH:MM made two readings
# of "when the bell rings" land hours apart. The code turns each into one fixed hour; None means "go now".
TIME_OF_DAY: dict[str, str | None] = {
    "straightaway": None, "dawn": "06:00", "morning": "09:00", "midday": "12:00", "afternoon": "15:00",
    "dusk": "18:00", "evening": "20:00", "night": "22:00", "deep night": "02:00",
}


@lru_cache(maxsize=256)
def agreement_schema(places: tuple[str, ...]) -> type[BaseModel]:
    """What two people agreed to do together in one talk, answered once for both of them."""
    place_t = Literal[places]  # type: ignore[valid-type]
    meeting = create_model(
        "Meeting",
        when_said=(str, Field(description="the words for the time exactly as spoken in the talk, "
                                          "e.g. 'at dusk', 'when the bell rings'")),
        both_agreed=(bool, Field(description="true only if one of them proposed it and the other said yes "
                                             "or offered the same")),
        day=(Literal["today", "tomorrow"], Field(
            description="today if that part of the day has not yet passed today, otherwise tomorrow")),
        time_of_day=(Literal[tuple(TIME_OF_DAY)], Field(  # type: ignore[valid-type]
            description="the part of the day when_said names: straightaway (going now), dawn, morning, midday, "
                        "afternoon, dusk, evening, night, deep night (after midnight). Bells and sunrise or "
                        "sunset words map to the nearest one")),
        activity=(str, Field(description="what the two will do together, a short phrase without either name, neutral for both readers: no \"home\", \"my\" or \"your\"")),
        place=(place_t, Field(description="where they will meet, one id from the PLACES list")),
    )
    return create_model(
        "Agreement",
        meetings=(list[meeting], Field(  # type: ignore[valid-type]
            max_length=2, description="only meetings the two agreed on in this talk, to spend time in the same "
                                      "place at a definite time later today or tomorrow; usually empty")))


@lru_cache(maxsize=256)
def leaving_schema(first_a: str, first_b: str) -> type[BaseModel]:
    """Whether anyone settled in one talk on leaving Hollowmere, answered once for both of them."""
    departure = create_model(
        "Departure",
        words=(str, Field(description="the words from the talk that settle it, copied exactly")),
        who=(Literal[first_a, first_b, "both"], Field(  # type: ignore[valid-type]
            description="who sets out from Hollowmere: one of them, or both together")),
        way_out=(str, Field(description="the gate, road or other way out of the town that was named in the talk, "
                                        "copied exactly; empty if none was named")),
        settled=(bool, Field(description="true only if whoever goes has firmly decided it in this talk and, if the "
                                         "other is to come along, the other said yes")),
    )
    return create_model(
        "Leaving",
        departures=(list[departure], Field(  # type: ignore[valid-type]
            max_length=1, description="a decision made in this talk to leave Hollowmere; almost always empty")))


@lru_cache(maxsize=256)
def moving_schema(first_a: str, first_b: str) -> type[BaseModel]:
    """Whether one of two settled in one talk on moving into the other's home, answered once for both."""
    move = create_model(
        "Move",
        words=(str, Field(description="the words from the talk that settle it, copied exactly")),
        who=(Literal[first_a, first_b], Field(  # type: ignore[valid-type]
            description="who moves out of their own home to live in the other's")),
        settled=(bool, Field(description="true only if the one who moves has firmly decided it in this talk and the "
                                         "other, whose home it will be, said yes")),
    )
    return create_model(
        "Moving",
        moves=(list[move], Field(  # type: ignore[valid-type]
            max_length=1, description="a decision made in this talk to live together under one roof for good; "
                                      "almost always empty")))


@lru_cache(maxsize=256)
def goblin_choice_schema(people: tuple[str, ...]) -> type[BaseModel]:
    return create_model(
        "GoblinChoice",
        thought=(str, Field(description="what goes through your head as you see them, first person, one sentence")),
        action=(Literal["approach", "watch", "hide", "flee"], Field(
            description="approach someone; watch from where you are, keeping still; hide in the nearest cover; "
                        "or flee as far as you can")),
        whom=(Literal[people + ("nobody",)], Field(  # type: ignore[valid-type]
            description="the person you approach, or the one you are most wary of")),
        reason=(str, Field(description="if approaching, what you want from them; otherwise empty")),
    )


@lru_cache(maxsize=256)
def chat_digest_schema(places: tuple[str, ...], about_goblin: bool = False,
                       romance_eligible: bool = False, people: tuple[str, ...] = ()) -> type[BaseModel]:
    place_t = Literal[places]  # type: ignore[valid-type]
    # The evidence comes first, so the model commits to it before inventing a time: the code then
    # checks `when_said` against the transcript and drops the other person's tasks and unanswered demands.
    commitment = create_model(
        "Commitment",
        when_said=(str, Field(description="the words for the time exactly as spoken in the talk, "
                                          "e.g. 'by noon', 'tomorrow at dusk'")),
        who=(Literal["me", "both", "them"], Field(
            description="who will do it: for a meeting, a walk, or any time spent in the same place, this is "
                        "almost always 'both' - even if you were the one who named the time or place, you "
                        "still have to be there too. Use 'me' only for a task you alone promised to do for "
                        "them with no meeting involved, and 'them' only for a task they alone promised to do "
                        "for you, elsewhere, that does not need you present")),
        accepted=(bool, Field(description="true only if whoever is to do it offered it or said yes")),
        day=(Literal["today", "tomorrow"], Field(
            description="today if that hour has not yet passed today, otherwise tomorrow")),
        time=(str, Field(pattern=HHMM, description="when_said as a 24-hour clock time (HH:MM). If the hour was "
                        "spoken vaguely ('by dusk', 'at six'), pick the single most natural reading for how the "
                        "day is passing right now - the other speaker is resolving the same moment independently "
                        "and must land on the same hour")),
        activity=(str, Field(description="what will be done, short phrase without your name")),
        place=(place_t, Field(description="where it will happen")),
    )
    fields: dict = dict(
        summary=(str, Field(description="what was said and what you learned, first person, one to three sentences")),
        importance=(int, Field(ge=1, le=10, description="how much this talk matters, 1 trivial to 10 life-changing - "
                               "judge it by what changed today, not by how long the talk ran")),
        feeling=(Stance, Field(
            description="how you feel about THEM now, judged from what THEY said and did in this talk toward "
                        "you. fear/dislike: they threatened, insulted or harmed you. wary: this talk gave you a "
                        "specific reason to suspect THIS person of lying or meaning harm. neutral: an ordinary "
                        "exchange, nothing notable either way. fond: you enjoyed their company, they were kind, "
                        "joked with you, or showed concern for you. trust: they confided in you, helped you, kept "
                        "a confidence or shared a risk with you. close/love: deep, long affection")),
        view_of_them=(str, Field(description="what you now make of them, first person, one short sentence")),
        commitments=(list[commitment], Field(  # type: ignore[valid-type]
            max_length=2, description="only what the two of you agreed on in this talk; usually empty. If more "
                        "than two things were truly agreed, keep only the two that matter most. A tentative "
                        "'maybe' that nobody confirmed can be left out entirely")),
    )
    if people:
        heard = create_model(
            "Heard",
            about=(Literal[people], Field(description="the third person the claim is about, by first name")),  # type: ignore[valid-type]
            claim=(str, Field(max_length=200, description="what THEY said about that person, a statement that names "
                                                          "that person, in a few words and in their own words where "
                                                          "you can; never what was said about you or about the talk")),
            trust=(Literal["believed", "unsure", "doubted"], Field(description=(
                "whether you take it for true: believed, unsure, or doubted"))),
        )
        fields["heard"] = (list[heard], Field(  # type: ignore[valid-type]
            default_factory=list, max_length=3, description=(
                "claims THEY made about other people that you did not know before; empty if they said nothing new of anyone. Only what they "
                "actually said, never your own guess")))
    fields["arrested"] = (Literal["no", "i_locked_them_up", "they_locked_me_up"], Field(
        default="no", description="only if, in this very conversation, someone was actually seized and locked up, "
                    "as an act the transcript shows: i_locked_them_up if you had them seized, they_locked_me_up if "
                    "they had you seized. A threat, a warning or a demand to be locked up is no. Default no"))
    if about_goblin:
        fields["toward_goblin"] = (GoblinRegard, Field(
            description="what you mean to do about the goblin now: drive it off or tell others; stay undecided; "
                        "tolerate it and keep quiet for now; feed it when it comes; or protect it and keep its "
                        "secret as your own"))
    if romance_eligible:
        fields["made_love"] = (bool, Field(
            description="true only if, in this very conversation, the two of you talked about it and then "
                        "made love - judge strictly from what the transcript actually shows; false by "
                        "default, including whenever nothing like that was said"))
    return create_model("ChatDigest", **fields)


@lru_cache(maxsize=32)
def glimpses_schema(n: int) -> type[BaseModel]:
    """The hour's sightings, each put in the watcher's words: as many lines as notes, in the same order."""
    return create_model(
        "Glimpses",
        lines=(list[Annotated[str, Field(max_length=300)]], Field(min_length=n, max_length=n, description=(
            f"exactly {n} lines, line k being note k in the watcher's own words: the same fact, one or two sentences, "
            "the person named"))),
    )
