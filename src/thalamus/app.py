"""THALAMUS in the browser: a local-only web app for setup and chat.

Run with `python -m thalamus.app` (the Start-THALAMUS launchers do this). The server listens on
127.0.0.1 only, and every API call must carry a random per-launch token that is embedded in the
page, so other websites open in the same browser cannot talk to it.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import secrets
import socket
import threading
import webbrowser
from collections.abc import Awaitable, Callable
from dataclasses import asdict
from importlib.resources import files
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from thalamus.brain import Brain, Response, create_brain
from thalamus.config import JEV_KEY_ENV, Settings, load_dotenv, load_settings, save_env
from thalamus.core.memory_store import MemoryStore
from thalamus.health import Check, explain, run_checks
from thalamus.providers.claude import WORKSPACE_ENV, ClaudeProvider, make_client
from thalamus.providers.jev import JevProvider

ANTHROPIC_ENV = "ANTHROPIC_API_KEY"
TOKEN_HEADER = "x-thalamus-token"
ALLOWED_HOSTS = ("127.0.0.1", "localhost")

BrainFactory = Callable[[Settings], Brain]
Checker = Callable[["SetupRequest", Settings], Awaitable[list[Check]]]


class SetupRequest(BaseModel):
    typesafe_key: str = ""
    anthropic_key: str = ""
    workspace_id: str = ""


class ChatRequest(BaseModel):
    message: str


async def check_keys(keys: SetupRequest, settings: Settings) -> list[Check]:
    """Validate keys against the real services before anything is saved."""
    from typesafe_sdk import AsyncTypeSafeClient

    typesafe_key = keys.typesafe_key.strip() or os.environ.get(JEV_KEY_ENV, "")
    anthropic_key = keys.anthropic_key.strip() or os.environ.get(ANTHROPIC_ENV, "")
    jev = JevProvider(settings.models.jev, client=AsyncTypeSafeClient(api_key=typesafe_key or "missing"))
    cortex = ClaudeProvider(
        settings.models,
        client=make_client(api_key=anthropic_key or None, workspace_id=keys.workspace_id.strip() or None),
    )
    try:
        return await run_checks(jev, cortex, settings.models.jev)
    finally:
        await jev.aclose()
        await cortex.aclose()


def serialize(response: Response) -> dict:
    return {
        "text": response.text,
        "action": response.action,
        "path": response.path,
        "context": response.context,
        "cost_usd": response.cost_usd,
        "modulators": response.modulators,
        "trace": [asdict(event) for event in response.trace.events],
    }


def key_status() -> dict:
    typesafe = bool(os.environ.get(JEV_KEY_ENV, "").strip())
    anthropic = bool(os.environ.get(ANTHROPIC_ENV, "").strip() or os.environ.get("ANTHROPIC_AUTH_TOKEN", "").strip())
    return {
        "typesafe": typesafe,
        "anthropic": anthropic,
        "workspace": bool(os.environ.get(WORKSPACE_ENV, "").strip()),
        "ready": typesafe and anthropic,
    }


def create_app(
    settings: Settings,
    *,
    env_path: Path = Path(".env"),
    brain_factory: BrainFactory = create_brain,
    checker: Checker = check_keys,
    token: str | None = None,
) -> FastAPI:
    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        await reset_brain()

    app = FastAPI(title="THALAMUS", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.token = token or secrets.token_urlsafe(24)
    app.state.brain = None
    lock = asyncio.Lock()
    page = files("thalamus.web").joinpath("index.html").read_text(encoding="utf-8")

    def brain() -> Brain:
        if app.state.brain is None:
            app.state.brain = brain_factory(settings)
        return app.state.brain

    async def reset_brain() -> None:
        if app.state.brain is not None:
            old: Brain = app.state.brain
            app.state.brain = None
            for close in (getattr(old.jev, "aclose", None), getattr(old.cortex, "aclose", None)):
                if close:
                    await close()
            old.memory.close()

    def error(exc: Exception, status: int = 502) -> JSONResponse:
        return JSONResponse({"error": asdict(explain(exc))}, status_code=status)

    @app.middleware("http")
    async def guard(request: Request, call_next):
        host = request.headers.get("host", "").rsplit(":", 1)[0]
        if host not in ALLOWED_HOSTS and host != "testserver":
            return JSONResponse({"error": {"message": "Forbidden host."}}, status_code=403)
        if request.url.path.startswith("/api/") and not secrets.compare_digest(
            request.headers.get(TOKEN_HEADER, ""), app.state.token
        ):
            return JSONResponse({"error": {"message": "Missing or invalid token."}}, status_code=403)
        return await call_next(request)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return page.replace("__THALAMUS_TOKEN__", app.state.token)

    @app.get("/api/status")
    async def status() -> dict:
        return key_status()

    @app.post("/api/setup")
    async def setup(keys: SetupRequest) -> JSONResponse:
        checks = await checker(keys, settings)
        ok = all(check.ok for check in checks)
        if ok:
            updates = {WORKSPACE_ENV: keys.workspace_id}
            if keys.typesafe_key.strip():
                updates[JEV_KEY_ENV] = keys.typesafe_key
            if keys.anthropic_key.strip():
                updates[ANTHROPIC_ENV] = keys.anthropic_key
            save_env(updates, env_path)
            async with lock:
                await reset_brain()
        return JSONResponse({"ok": ok, "checks": [c.to_dict() for c in checks], "status": key_status()})

    @app.post("/api/chat")
    async def chat(request: ChatRequest) -> JSONResponse:
        if not request.message.strip():
            return JSONResponse({"error": {"message": "Say something first."}}, status_code=400)
        async with lock:
            try:
                response = await brain().think(request.message)
            except Exception as exc:  # noqa: BLE001 - report provider failures, keep serving
                return error(exc)
        return JSONResponse(serialize(response))

    @app.get("/api/history")
    async def history() -> dict:
        current: Brain | None = app.state.brain
        if current is None:
            return {"turns": []}
        return {"turns": [{"role": t.role, "content": t.content, "meta": t.meta} for t in current.history]}

    @app.post("/api/conversation/new")
    async def new_conversation() -> dict:
        async with lock:
            if app.state.brain is not None:
                app.state.brain.new_conversation()
        return {"ok": True}

    @app.get("/api/mind")
    async def mind() -> dict:
        current: Brain | None = app.state.brain
        return current.snapshot() if current is not None else {}

    @app.get("/api/memory")
    async def memory() -> dict:
        store = app.state.brain.memory if app.state.brain else MemoryStore(settings.memory_path)
        try:
            return {
                "episodes": [dict(row) for row in store.episodes(50)],
                "values": [dict(row) for row in store.values()],
            }
        finally:
            if store is not getattr(app.state.brain, "memory", None):
                store.close()

    @app.post("/api/memory/clear")
    async def clear_memory() -> dict:
        async with lock:
            store = app.state.brain.memory if app.state.brain else MemoryStore(settings.memory_path)
            store.clear()
            if app.state.brain is not None:
                app.state.brain.prefrontal.working_memory.reset()
            if store is not getattr(app.state.brain, "memory", None):
                store.close()
        return {"ok": True}

    return app


def free_port(start: int = 8765, attempts: int = 50) -> int:
    for port in range(start, start + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("No free port found for THALAMUS.")


def main(open_browser: bool = True, config: Path | None = None) -> None:
    import uvicorn

    load_dotenv()
    settings = load_settings(config)
    app = create_app(settings)
    port = free_port()
    url = f"http://127.0.0.1:{port}/"
    print(f"\n  THALAMUS is running at {url}\n  Keep this window open while you chat. Close it to stop THALAMUS.\n")
    if open_browser:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
