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
from typing import Literal

from pydantic import BaseModel, Field, create_model

from hollowmere.agents.gods import GOD_IDS
from hollowmere.agents.persona import Stance

# How a human means to treat the goblin after a talk; a step can rise or fall from one talk to the next.
GoblinRegard = Literal["drive off", "undecided", "tolerate", "feed", "protect"]

PrayTo = Literal[GOD_IDS + ("none",)]
DivineForm = Literal["dream", "sign", "omen", "coincidence", "buff", "silence"]
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


class DivineAnswer(BaseModel):
    form: DivineForm = Field(description="how you answer: a dream, a waking sign, an omen, a mere "
                                         "coincidence, a tangible favor (buff), or silence - choose freely; "
                                         "silence is common and not a failure to decide")
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
                       romance_eligible: bool = False) -> type[BaseModel]:
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
