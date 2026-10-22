# Hollowmere

A reimplementation of *Generative Agents: Interactive Simulacra of Human Behavior*
(Park et al., 2023), set in a plague-haunted medieval town and driven by a local LLM.

Eight townsfolk (an innkeeper and her restless daughter, a haunted smith and his
apprentice, a doubting priest, a gravedigger who dug a nameless grave by night, the watch
captain, and a herb-wife the town calls a witch), the women of their families, and
one goblin wake, plan their days, notice each other, talk, remember, and reflect. You watch.

**Nib, the last goblin** lives in a burrow under the Hollow Keep, a ruin deep in the
Witchwood. He is weak and wants friends. Whenever someone who could give him away comes into
sight, the model decides for him whether to approach, keep still and watch, hide or flee,
knowing who is there and how he feels about each of them. When planning a stretch of his day
he may also choose someone to go and find, and he creeps after them. His musk makes people
see him as pitiable, which colours their judgement but does not decide it: after talking with
him each person chooses for themselves whether to protect him (Wenna can tell the musk for
what it is). Strangers glimpse him only by chance, and a sighting becomes a vivid memory that
can spread as rumour; people who have met him see him plainly.

Everyone keeps a relationship with each person they know: it starts from the persona and is
rewritten by the model after every talk (fear, dislike, wary, neutral, fond, trust, plus a
sentence on what they now make of them). Click a villager to see theirs.

## Run

```sh
uv sync
cd web && npm install && npm run build && cd ..

uv run hollowmere check          # is the LLM server reachable, and which JSON mode works?
uv run hollowmere serve          # open http://127.0.0.1:8000
uv run hollowmere serve --fake   # no LLM: a scripted stand-in brain, for trying the UI
uv run hollowmere serve --fresh  # discard the saved run and start again
uv run hollowmere report         # judge the run: movement, talks, reflections, promises, LLM failures
uv run hollowmere report --hours 6   # only the last 6 game hours (e.g. after changing a prompt)
```

The LLM is configured in `config.toml`. The default points at llama.cpp's
`llama-server` on `http://127.0.0.1:8080/v1`, but any OpenAI-compatible chat server works.
Start the server with `./start-llama.ps1` (edit the paths at the top). It must be
`llama-server`, not `llama-cli`: the CLI is an interactive chat and opens no port.
The script passes `--jinja` (so Qwen's thinking can be switched off) and `--parallel 2`
to match `max_concurrency`. The model must fit in RAM + VRAM; one that does not is
read from disk and runs at about 1 token/s, far too slow for the town.
To use the adesso AI Hub (a LiteLLM proxy) instead, with Qwen 3.6 35B and real embeddings:

```powershell
$env:LITELLM_API_KEY = "sk-..."    # the key is read from the environment, never from a file
uv run hollowmere --config config.litellm.toml check
uv run hollowmere --config config.litellm.toml serve
uv run hollowmere --config config.litellm.toml report
```

That run keeps its own save (`saves/hollowmere-litellm.db`), so it never mixes with the local one.
Runs are saved to `saves/` and resume automatically. Failed LLM calls (invalid replies,
server errors) are logged with their prompt and reply to `saves/llm_log.jsonl`.

Viewer: drag to pan, scroll to zoom, click a villager (or a name in the list) to
follow them and read their plan, memories and reflections. Space pauses.

For frontend work, run `npm run dev` in `web/` (port 5173, proxies to the backend).

## What changed compared with the original

| | Original | Hollowmere |
|---|---|---|
| LLM output | free text parsed with `split()`, silent "failsafe" answers | JSON Schema-constrained replies validated by Pydantic; invalid replies are sent back to the model with the error |
| Choosing where to go | 3 sequential calls (sector → arena → object) + fuzzy matching | one enum field listing only places the agent knows |
| Calls per agent per day | ~670+ before conversations and reflection | ~50-90 in total |
| Agents | processed one after another | each tick's LLM calls run concurrently |
| Importance scoring | one LLM call per perceived event | heuristic for routine sights; the model rates importance inside calls it is already making |
| Recency | `0.995^rank` in a list sorted oldest-first (it favours old memories) | exponential decay in game hours since last access |
| Frontend ↔ backend | JSON files polled from disk every frame | WebSocket push |
| Storage | hundreds of thousands of JSON files | one SQLite database, resumable at any time |
| Conversations | whole dialogue generated in one step | turn by turn, each speaker drawing on their own memories, shown live |
| Commitments | buried in the transcript | extracted ("tomorrow 18:00, the vigil at the chapel") and inserted into plans |
| Map | Tiled export + 5 CSV layers | built in code (`hollowmere/world/town.py`) |

## Layout

```
hollowmere/
  llm/       client.py (OpenAI-compatible, structured output, downgrade/repair), fake.py
  memory/    stream.py (retrieval), store.py (SQLite), embed.py (server or hashing embeddings)
  agents/    agent.py (plan, decompose, react, converse, reflect), prompts.py, schemas.py, persona.py
  world/     world.py (grid, areas, objects, A*), town.py (the map)
  sim/       engine.py (tick loop), setup.py
  server/    app.py (FastAPI + WebSocket)
  data/      personas.json
web/         TypeScript + Canvas 2D viewer, procedural pixel art, no image assets
tests/
```

To add or change villagers, edit `hollowmere/data/personas.json`. Place ids refer to
areas and objects in `town.py`. Then run with `--fresh`.
