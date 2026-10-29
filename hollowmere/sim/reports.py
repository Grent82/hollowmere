"""Two evening collections, written to files and never fed back into the game: what each god would do among
mortals if it could, and how the townsfolk would deal with the world beyond what it has now.

They only read the save, run in the background after the day's end, and a failed call costs nothing but its entry.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime, time
from pathlib import Path

from hollowmere.agents import prompts
from hollowmere.agents.agent import DAY_START, day_key
from hollowmere.agents.gods import GODS
from hollowmere.agents.schemas import GodWishes, world_ideas_schema
from hollowmere.llm.client import LLMError

log = logging.getLogger(__name__)

REPORT_AT = time(23, 0)
KEPT = 40  # entries of its own file a call is shown, to add to and not repeat
WORD = re.compile(r"[a-z0-9']+")


def folder(engine) -> Path | None:
    """Beside the save: saves/hollowmere-litellm.db -> saves/hollowmere-litellm-reports/ (none for :memory:)."""
    path = engine.cfg.save_path
    return None if path == ":memory:" else Path(path).with_suffix("").parent / (Path(path).stem + "-reports")


def _existing(path: Path | None) -> list[str]:
    if path is None or not path.exists():
        return []
    return [x for x in path.read_text(encoding="utf-8").splitlines() if x.startswith("- ")][-KEPT:]


def _append(path: Path | None, title: str, day: date, bullets: list[str]) -> None:
    if path is None or not bullets:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = not path.exists()
    with path.open("a", encoding="utf-8") as f:
        if fresh:
            f.write(f"# {title}\n")
        f.write(f"\n## {day.isoformat()}\n" + "\n".join(bullets) + "\n")


def _norm(text: str) -> str:
    return " ".join(WORD.findall(text.lower()))


async def god_wishes(engine, now: datetime) -> None:
    out = folder(engine)
    path = out / "god_wishes.md" if out else None
    bullets: list[str] = []
    for god in GODS:
        prayers = engine.store.prayers_to(god.id)
        if not prayers:
            continue
        try:
            answer = await engine.ctx.llm.structured(
                prompts.wishes_system_prompt(god),
                prompts.god_wishes(god, now, prayers, engine.ctx.names, (_existing(path) + bullets)[-KEPT:]),
                GodWishes, label="god_wishes", temperature=0.8)
        except LLMError as e:
            log.warning("%s: wishes failed (%s)", god.id, e)
            continue
        bullets += [f"- **{god.name}** -> {w.towards}: {w.wish} _({w.why})_" for w in answer.wishes]
    _append(path, "What the gods would do", day_key(now), bullets)


async def world_ideas(engine, now: datetime) -> None:
    out = folder(engine)
    path = out / "world_ideas.md" if out else None
    objects = sorted({o.name for o in engine.world.objects.values()})
    have = [_norm(o) for o in objects]
    places = {a.id: a.name for a in engine.world.areas.values()}
    schema = world_ideas_schema(tuple(places))
    start = datetime.combine(day_key(now), DAY_START)
    bullets: list[str] = []
    for agent in engine.agents.values():
        if agent.persona.dependent or agent.is_away:
            continue
        mems = [m for m in agent.memory.items if m.created >= start and m.kind != "plan" and m.importance >= 3][-40:]
        if not mems:
            continue
        plan = [f"{b.start:%H:%M} {b.activity}" for b in agent.plan]
        corpus = _norm(" ".join([m.text for m in mems] + plan))
        try:
            answer = await engine.ctx.llm.structured(
                prompts.world_ideas_system_prompt(),
                prompts.world_ideas(agent, now, plan, mems, objects, (_existing(path) + bullets)[-KEPT:], places),
                schema, label="world_ideas", temperature=0.5)
        except LLMError as e:
            log.warning("%s: world ideas failed (%s)", agent.id, e)
            continue
        kept = [i for i in answer.ideas if _norm(i.evidence) and _norm(i.evidence) in corpus  # a quote that is there
                and (i.kind == "change" or not any(o in _norm(i.item + " " + i.what) for o in have))]  # not a thing the game has
        bullets += [f"- **{agent.first}** [{i.kind}] {i.item} ({places[i.where]}): {i.what} _(\"{i.evidence}\")_" for i in kept]
    _append(path, "What the townsfolk would change", day_key(now), bullets)


async def run(engine, now: datetime) -> None:
    await god_wishes(engine, now)
    await world_ideas(engine, now)
