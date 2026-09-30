"""THALAMUS in the browser: a local web app for setup and chat, optionally reachable from your phone.

Run with `python -m thalamus.app` (the Start-THALAMUS launchers do this). By default the server
listens on 127.0.0.1 only. With phone access on (see `thalamus.remote`) it also accepts your own
devices over Tailscale, after a passcode login. Every API call must carry a random per-launch token
that is embedded in the page, so other websites open in the same browser cannot talk to it.
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
from fastapi.responses import Response as HTTPResponse
from pydantic import BaseModel

from thalamus.brain import Brain, Response, create_brain
from thalamus.config import JEV_KEY_ENV, Settings, load_dotenv, load_settings, save_env
from thalamus.core.memory_store import MemoryStore
from thalamus.health import Check, explain, run_checks
from thalamus.providers.claude import WORKSPACE_ENV, ClaudeProvider, make_client
from thalamus.providers.jev import JevProvider
from thalamus.providers.search import SEARCH_KEY_ENV, TavilySearch
from thalamus.remote import (
    COOKIE,
    MIN_PASSCODE,
    PASSCODE_ENV,
    REMOTE_ENV,
    SECRET_ENV,
    KeepAwake,
    LoginThrottle,
    allowed_remote,
    check_passcode,
    hash_passcode,
    is_loopback,
    make_session,
    phone_urls,
    qr_svg,
    remote_enabled,
    tailscale_info,
    valid_session,
)

ANTHROPIC_ENV = "ANTHROPIC_API_KEY"
TOKEN_HEADER = "x-thalamus-token"
ALLOWED_HOSTS = ("127.0.0.1", "localhost")

BrainFactory = Callable[[Settings], Brain]
Checker = Callable[["SetupRequest", Settings], Awaitable[list[Check]]]


class SetupRequest(BaseModel):
    typesafe_key: str = ""
    anthropic_key: str = ""
    workspace_id: str = ""
    tavily_key: str = ""


class ChatRequest(BaseModel):
    message: str


class LoginRequest(BaseModel):
    passcode: str


class RemoteRequest(BaseModel):
    enabled: bool
    passcode: str = ""


PUBLIC_PATHS = {"/login", "/manifest.webmanifest", "/icon-180.png", "/icon-512.png"}
PC_ONLY = {"/api/setup", "/api/remote"}  # settings and keys can only be changed on the PC
MANIFEST = {
    "name": "THALAMUS",
    "short_name": "THALAMUS",
    "description": "A brain-modelled mind: JEV for intuition, Claude for thought.",
    "start_url": "/",
    "display": "standalone",
    "background_color": "#111117",
    "theme_color": "#6d5dfc",
    "icons": [
        {"src": "/icon-180.png", "sizes": "180x180", "type": "image/png"},
        {"src": "/icon-512.png", "sizes": "512x512", "type": "image/png"},
    ],
}


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
    tavily_key = keys.tavily_key.strip() or os.environ.get(SEARCH_KEY_ENV, "")
    search = TavilySearch(api_key=tavily_key, depth=settings.web.depth) if tavily_key else None
    try:
        return await run_checks(jev, cortex, settings.models.jev, search)
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
        "cost_breakdown": response.cost_breakdown,
        "sources": response.sources,
        "modulators": response.modulators,
        "trace": [asdict(event) for event in response.trace.events],
    }


def key_status(local: bool = True) -> dict:
    typesafe = bool(os.environ.get(JEV_KEY_ENV, "").strip())
    anthropic = bool(os.environ.get(ANTHROPIC_ENV, "").strip() or os.environ.get("ANTHROPIC_AUTH_TOKEN", "").strip())
    return {
        "typesafe": typesafe,
        "anthropic": anthropic,
        "workspace": bool(os.environ.get(WORKSPACE_ENV, "").strip()),
        "web": bool(os.environ.get(SEARCH_KEY_ENV, "").strip()),
        "ready": typesafe and anthropic,
        "local": local,
    }


def create_app(
    settings: Settings,
    *,
    env_path: Path = Path(".env"),
    brain_factory: BrainFactory = create_brain,
    checker: Checker = check_keys,
    token: str | None = None,
    remote_active: bool = False,
    port: int = 8765,
) -> FastAPI:
    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        await reset_brain()

    app = FastAPI(title="THALAMUS", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.token = token or secrets.token_urlsafe(24)
    app.state.brain = None
    app.state.remote_active = remote_active  # bound for phone access in this run
    throttle = LoginThrottle()
    lock = asyncio.Lock()
    web = files("thalamus.web")
    page = web.joinpath("index.html").read_text(encoding="utf-8")
    login_page = web.joinpath("login.html").read_text(encoding="utf-8")
    icons = {name: web.joinpath(name[1:]).read_bytes() for name in ("/icon-180.png", "/icon-512.png")}

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

    def is_local(request: Request) -> bool:
        return is_loopback(request.client.host if request.client else None)

    @app.middleware("http")
    async def guard(request: Request, call_next):
        path = request.url.path
        client = request.client.host if request.client else None
        if is_loopback(client):
            host = request.headers.get("host", "").rsplit(":", 1)[0]
            if host not in ALLOWED_HOSTS and host != "testserver":  # DNS-rebinding protection
                return JSONResponse({"error": {"message": "Forbidden host."}}, status_code=403)
        else:
            if not app.state.remote_active or not allowed_remote(client, settings.remote.allow_lan):
                return JSONResponse({"error": {"message": "Phone access is off."}}, status_code=403)
            if path not in PUBLIC_PATHS and not valid_session(
                request.cookies.get(COOKIE), os.environ.get(SECRET_ENV, "")
            ):
                if path == "/":
                    return HTMLResponse(login_page)
                return JSONResponse({"error": {"message": "Please log in again."}}, status_code=401)
            if path in PC_ONLY and request.method != "GET":
                return JSONResponse({"error": {"message": "Change settings on the PC."}}, status_code=403)
        if path.startswith("/api/") and not secrets.compare_digest(
            request.headers.get(TOKEN_HEADER, ""), app.state.token
        ):
            return JSONResponse({"error": {"message": "Missing or invalid token."}}, status_code=403)
        return await call_next(request)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return page.replace("__THALAMUS_TOKEN__", app.state.token)

    @app.get("/manifest.webmanifest")
    async def manifest() -> JSONResponse:
        return JSONResponse(MANIFEST, media_type="application/manifest+json")

    @app.get("/icon-180.png")
    async def icon_small() -> HTTPResponse:
        return HTTPResponse(icons["/icon-180.png"], media_type="image/png")

    @app.get("/icon-512.png")
    async def icon_large() -> HTTPResponse:
        return HTTPResponse(icons["/icon-512.png"], media_type="image/png")

    @app.get("/login", response_class=HTMLResponse)
    async def login_form() -> str:
        return login_page

    @app.post("/login")
    async def login(request: Request, body: LoginRequest) -> JSONResponse:
        client = request.client.host if request.client else "?"
        stored, secret = os.environ.get(PASSCODE_ENV, ""), os.environ.get(SECRET_ENV, "")
        if throttle.locked(client):
            return JSONResponse({"error": {"message": "Too many tries. Wait a few minutes."}}, status_code=429)
        if not stored or not secret:
            return JSONResponse({"error": {"message": "Set a passcode on the PC first."}}, status_code=403)
        if not check_passcode(body.passcode, stored):
            throttle.fail(client)
            return JSONResponse({"error": {"message": "Wrong passcode."}}, status_code=401)
        throttle.succeed(client)
        response = JSONResponse({"ok": True})
        response.set_cookie(
            COOKIE,
            make_session(secret, settings.remote.session_days),
            max_age=int(settings.remote.session_days * 86400),
            httponly=True,
            samesite="strict",
        )
        return response

    @app.get("/api/status")
    async def status(request: Request) -> dict:
        return key_status(local=is_local(request))

    @app.get("/api/remote")
    async def remote_status(request: Request) -> dict:
        info = await asyncio.to_thread(tailscale_info)
        urls = phone_urls(port, info) if app.state.remote_active else []
        return {
            "enabled": remote_enabled(),
            "active": app.state.remote_active,
            "passcode_set": bool(os.environ.get(PASSCODE_ENV, "").strip()),
            "port": port,
            "tailscale": info,
            "urls": urls,
            "qr": qr_svg(urls[0]) if urls else None,
            "local": is_local(request),
        }

    @app.post("/api/remote")
    async def remote_update(body: RemoteRequest) -> JSONResponse:
        passcode = body.passcode.strip()
        if passcode and len(passcode) < MIN_PASSCODE:
            message = f"Use a passcode of at least {MIN_PASSCODE} characters."
            return JSONResponse({"error": {"message": message}}, status_code=400)
        if body.enabled and not passcode and not os.environ.get(PASSCODE_ENV, "").strip():
            return JSONResponse({"error": {"message": "Choose a passcode first."}}, status_code=400)
        updates = {REMOTE_ENV: "1" if body.enabled else ""}
        if passcode:  # a new passcode also logs every phone out
            updates[PASSCODE_ENV] = hash_passcode(passcode)
            updates[SECRET_ENV] = secrets.token_urlsafe(32)
        save_env(updates, env_path)
        return JSONResponse(
            {"ok": True, "enabled": body.enabled, "restart_required": body.enabled != app.state.remote_active}
        )

    @app.post("/api/setup")
    async def setup(keys: SetupRequest) -> JSONResponse:
        checks = await checker(keys, settings)
        ok = all(check.ok for check in checks)
        if ok:
            updates = {}
            if keys.workspace_id.strip():  # blank means "keep what's saved", like the key fields
                updates[WORKSPACE_ENV] = keys.workspace_id
            if keys.typesafe_key.strip():
                updates[JEV_KEY_ENV] = keys.typesafe_key
            if keys.anthropic_key.strip():
                updates[ANTHROPIC_ENV] = keys.anthropic_key
            if keys.tavily_key.strip():
                updates[SEARCH_KEY_ENV] = keys.tavily_key
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
    remote = remote_enabled() and bool(os.environ.get(PASSCODE_ENV, "").strip())
    if remote:  # a fixed port, on every network interface (non-Tailscale devices are still refused)
        port, bind = settings.remote.port, "0.0.0.0"
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((bind, port))
            except OSError:
                print(f"\n  Port {port} is busy. Is THALAMUS already running in another window?\n")
                raise SystemExit(1) from None
    else:
        port, bind = free_port(settings.remote.port), "127.0.0.1"
    app = create_app(settings, remote_active=remote, port=port)
    url = f"http://127.0.0.1:{port}/"
    print(f"\n  THALAMUS is running at {url}\n  Keep this window open while you chat. Close it to stop THALAMUS.\n")
    awake = KeepAwake()
    if remote:
        urls = phone_urls(port, tailscale_info())
        print("  Phone access is ON." + (" Open on your phone: " + "  or  ".join(urls) if urls else ""))
        if not urls:
            print("  Tailscale doesn't seem to be connected on this PC; start it to reach THALAMUS from your phone.")
        if settings.remote.keep_awake:
            awake.start()
            if awake.active:
                print("  This PC will stay awake while THALAMUS runs.")
        print()
    if open_browser:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    try:
        uvicorn.run(app, host=bind, port=port, log_level="warning")
    finally:
        awake.stop()


if __name__ == "__main__":
    main()
