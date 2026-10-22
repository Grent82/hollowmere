"""Command line: `hollowmere serve`, `hollowmere check`, `hollowmere report`."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from hollowmere.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(prog="hollowmere")
    parser.add_argument("--config", default="config.toml")
    sub = parser.add_subparsers(dest="cmd", required=True)
    serve = sub.add_parser("serve", help="run the simulation and the web viewer")
    serve.add_argument("--fake", action="store_true", help="use the offline fake brain instead of the LLM")
    serve.add_argument("--fresh", action="store_true", help="discard the saved run and start over")
    sub.add_parser("check", help="test the connection to the LLM server")
    report = sub.add_parser("report", help="judge a run from its save: movement, talks, reflections, LLM")
    report.add_argument("--hours", type=float, help="only the last N game hours")
    report.add_argument("--fake", action="store_true", help="report on the fake-brain save")
    report.add_argument("--save", help="save file to read (default: from the config)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    if sys.platform == "win32":  # emoji in log lines
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    cfg = load_config(args.config)

    if args.cmd == "check":
        sys.exit(asyncio.run(check(cfg)))
    if args.cmd == "report":
        report_cmd(cfg, hours=args.hours, fake=args.fake, save=args.save)
    if args.cmd == "serve":
        serve_cmd(cfg, fake=args.fake, fresh=args.fresh)


def serve_cmd(cfg, *, fake: bool, fresh: bool) -> None:
    import uvicorn

    from hollowmere.server.app import create_app
    from hollowmere.sim.setup import build_engine

    async def runner():
        engine = await build_engine(cfg, fake=fake, fresh=fresh)
        app = create_app(engine)
        config = uvicorn.Config(app, host=cfg.server.host, port=cfg.server.port, log_level="warning")
        print(f"\n  Hollowmere is running: http://{cfg.server.host}:{cfg.server.port}\n", flush=True)
        await uvicorn.Server(config).serve()

    asyncio.run(runner())


def report_cmd(cfg, *, hours: float | None, fake: bool, save: str | None) -> None:
    from pathlib import Path

    from hollowmere.agents.persona import load_personas
    from hollowmere.memory.store import Store
    from hollowmere.report import Report
    from hollowmere.world.town import build_town

    path = save or (cfg.sim.save_path.replace(".db", "-fake.db") if fake else cfg.sim.save_path)
    if not Path(path).exists():
        sys.exit(f"no save at {path}")
    store = Store(path)
    try:
        print(Report(store, build_town(), load_personas(), hours).build())
    finally:
        store.close()


async def check(cfg) -> int:
    """Probe the server: plain chat, each structured-output mode, and embeddings."""
    import time

    import httpx
    from pydantic import BaseModel

    from hollowmere.llm.client import LLMError, OpenAICompatLLM
    from hollowmere.memory.embed import make_embedder

    class Probe(BaseModel):
        town: str
        mood: int

    print(f"LLM server: {cfg.llm.base_url}  model: {cfg.llm.model}")
    try:
        async with httpx.AsyncClient(timeout=10) as http:
            r = await http.get(cfg.llm.base_url.rstrip("/") + "/models",
                               headers={"Authorization": f"Bearer {cfg.llm.key}"})
            print(f"  /models -> HTTP {r.status_code}")
    except httpx.HTTPError as e:
        print(f"  cannot reach the server: {e!r}")
        return 1

    llm = OpenAICompatLLM(cfg.llm.model_copy(update={"log_path": ""}))
    try:
        t = time.perf_counter()
        out = await llm.structured("You describe places.", "Name a gloomy medieval town and rate its mood 1-10.",
                                   Probe, label="check")
        print(f"  structured output OK in {time.perf_counter() - t:.1f}s: {out}")
        print(f"  mode in use: {llm._mode}, thinking flag sent: {llm._send_thinking_flag}")
    except LLMError as e:
        print(f"  structured output FAILED: {e}")
        return 1
    finally:
        await llm.aclose()

    emb = await make_embedder(cfg.embeddings, cfg.llm)
    print(f"  embeddings: {emb.signature}")
    return 0


if __name__ == "__main__":
    main()
