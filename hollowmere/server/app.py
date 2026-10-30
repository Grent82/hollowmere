"""HTTP + WebSocket server: streams the simulation to the browser and takes controls."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from hollowmere.sim import knowledge
from hollowmere.sim.engine import Engine

log = logging.getLogger(__name__)

WEB_DIST = Path(__file__).resolve().parents[2] / "web" / "dist"


class Control(BaseModel):
    paused: bool | None = None
    speed: float | None = None


class Whisper(BaseModel):
    agent: str
    text: str
    kind: str


def create_app(engine: Engine) -> FastAPI:
    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        task = asyncio.create_task(engine.run())
        try:
            yield
        finally:
            engine.stop()
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            engine.save()
            await engine.ctx.llm.aclose()

    app = FastAPI(title="Hollowmere", lifespan=lifespan)

    @app.get("/api/world")
    async def world():
        return engine.world.to_dict()

    @app.get("/api/state")
    async def state():
        return engine.state_view()

    @app.get("/api/agents/{agent_id}")
    async def agent(agent_id: str):
        detail = engine.agent_detail(agent_id)
        if detail is None:
            raise HTTPException(404, "no such agent")
        return detail

    @app.get("/api/knowledge")
    async def knowledge_view():
        return knowledge.build(engine)

    @app.get("/api/events")
    async def events(limit: int = 200):
        return list(engine.events)[-limit:]

    @app.post("/api/control")
    async def control(c: Control):
        if c.paused is not None:
            engine.paused = c.paused
        if c.speed is not None:
            engine.speed = max(0.25, min(c.speed, 32.0))
        await engine._broadcast({"type": "tick", **engine.state_view()})
        return {"paused": engine.paused, "speed": engine.speed}

    @app.post("/api/whisper")
    async def whisper(w: Whisper):
        try:
            await engine.whisper(w.agent, w.text, w.kind)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        return {"ok": True}

    @app.websocket("/ws")
    async def ws(socket: WebSocket):
        await socket.accept()
        queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=200)

        async def listener(msg: dict) -> None:
            if queue.full():  # slow client: drop the oldest message rather than stall the sim
                queue.get_nowait()
            queue.put_nowait(msg)

        await socket.send_json({"type": "init", "world": engine.world.to_dict(),
                                "events": list(engine.events)[-150:], **engine.state_view()})
        engine.listeners.add(listener)
        try:
            while True:
                await socket.send_json(await queue.get())
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            engine.listeners.discard(listener)

    if WEB_DIST.exists():
        app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

        @app.get("/")
        async def index():
            return FileResponse(WEB_DIST / "index.html")

    return app
