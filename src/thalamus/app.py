"""THALAMUS in the browser: a local web app for setup and chat, optionally reachable from your phone.

Run with `python -m thalamus.app` (the Start-THALAMUS launchers do this). By default the server
listens on 127.0.0.1 only. With phone access on (see `thalamus.remote`) it also accepts your own
devices over Tailscale, after a passcode login. Every API call must carry a random per-launch token
that is embedded in the page, so other websites open in the same browser cannot talk to it.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
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
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
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


class FactCreate(BaseModel):
    text: str
    category: str = "other"


class MailAccountCreate(BaseModel):
    provider: str
    address: str
    password: str = ""
    host: str = ""
    port: int = 0
    username: str = ""


class OutlookStart(BaseModel):
    address: str
    client_id: str


class NotifyUpdate(BaseModel):
    enabled: bool
    server: str = ""
    new_topic: bool = False


class FactUpdate(BaseModel):
    text: str | None = None
    pinned: bool | None = None
    category: str | None = None


PUBLIC_PATHS = {"/login", "/manifest.webmanifest", "/icon-180.png", "/icon-512.png"}
# Settings, keys and accounts can only be changed on the PC.
PC_ONLY = ("/api/setup", "/api/remote", "/api/mail/accounts", "/api/mail/outlook", "/api/notify")
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
    sleep_check_seconds: float = 60.0,
) -> FastAPI:
    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI):
        watchers = [asyncio.create_task(sleep_when_idle()), asyncio.create_task(check_mail())] \
            if sleep_check_seconds else []
        yield
        for watcher in watchers:
            watcher.cancel()
        await reset_brain()

    async def check_mail() -> None:
        """Check email on schedule while THALAMUS runs (read-only)."""
        while True:
            await asyncio.sleep(sleep_check_seconds)
            try:
                service = mail_service()
                if service is not None and service.due():
                    await service.check_all()
            except Exception:  # noqa: BLE001 - a failed check is recorded per account; keep the server up
                continue

    def mail_service():
        """The brain's mail service, if the brain can be loaded (keys present)."""
        try:
            return brain().mail
        except Exception:  # noqa: BLE001 - no keys yet
            return None

    async def sleep_when_idle() -> None:
        """Consolidate memories in the background once THALAMUS has been idle for a while."""
        while True:
            await asyncio.sleep(sleep_check_seconds)
            current: Brain | None = app.state.brain
            if current is None or lock.locked():
                continue
            try:
                if current.sleep_due():
                    async with lock:
                        await current.sleep(trigger="idle")
            except Exception:  # noqa: BLE001 - a failed sleep must never take the server down
                continue

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
            if path.startswith(PC_ONLY) and request.method != "GET":
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

    @app.post("/api/chat/stream")
    async def chat_stream(request: ChatRequest):
        """The reply streams as NDJSON lines: status notes, text deltas, then the full result."""
        if not request.message.strip():
            return JSONResponse({"error": {"message": "Say something first."}}, status_code=400)
        queue: asyncio.Queue[dict | None] = asyncio.Queue()

        async def on_text(delta: str) -> None:
            await queue.put({"type": "text", "delta": delta})

        async def on_status(note: str) -> None:
            await queue.put({"type": "status", "text": note})

        async def run() -> None:
            async with lock:
                try:
                    response = await brain().think(request.message, on_text=on_text, on_status=on_status)
                    await queue.put({"type": "done", **serialize(response)})
                except Exception as exc:  # noqa: BLE001 - report provider failures, keep serving
                    await queue.put({"type": "error", "error": asdict(explain(exc))})
                finally:
                    await queue.put(None)

        task = asyncio.create_task(run())

        async def lines():
            try:
                while (item := await queue.get()) is not None:
                    yield json.dumps(item) + "\n"
            finally:
                await task

        return StreamingResponse(lines(), media_type="application/x-ndjson")

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

    @contextlib.contextmanager
    def memory_store():
        """The running brain's memory, or the memory file itself if no brain is loaded yet."""
        current: Brain | None = app.state.brain
        if current is not None:
            yield current.memory
            return
        store = MemoryStore(settings.memory_path)
        try:
            yield store
        finally:
            store.close()

    @app.get("/api/facts")
    async def facts() -> dict:
        from thalamus.regions.sleep import pending_episodes

        with memory_store() as store:
            return {
                "facts": [dict(row) for row in store.facts()],
                "reports": store.sleep_reports(3),
                "pending": pending_episodes(store),
            }

    @app.post("/api/facts")
    async def add_fact(body: FactCreate) -> dict:
        if not body.text.strip():
            return JSONResponse({"error": {"message": "Write the fact first."}}, status_code=400)
        with memory_store() as store:
            return {"id": store.add_fact(body.text, body.category, [], origin="you")}

    @app.patch("/api/facts/{fact_id}")
    async def update_fact(fact_id: int, body: FactUpdate) -> dict:
        with memory_store() as store:
            store.update_fact(fact_id, text=body.text, pinned=body.pinned, category=body.category)
        return {"ok": True}

    @app.delete("/api/facts/{fact_id}")
    async def delete_fact(fact_id: int) -> dict:
        with memory_store() as store:
            store.retire_fact(fact_id)
        return {"ok": True}

    @app.post("/api/sleep")
    async def sleep_now() -> JSONResponse:
        async with lock:
            try:
                report = await brain().sleep(trigger="manual", force=True)
            except Exception as exc:  # noqa: BLE001
                return error(exc)
        return JSONResponse(report)

    def require_mail():
        service = mail_service()
        if service is None:
            raise_error = JSONResponse({"error": {"message": "Finish setting up your keys first."}}, status_code=400)
            return None, raise_error
        return service, None

    @app.get("/api/mail")
    async def mail_overview(surfaced: bool = False, hours: float = 72) -> JSONResponse:
        service, failure = require_mail()
        if failure:
            return failure
        return JSONResponse({
            "accounts": [a.public() for a in service.accounts()],
            "items": service.items(hours=hours, surfaced_only=surfaced),
            "notify": {k: v for k, v in service.notify_settings().items() if k != "topic"},
            "spent_today": service.spent_today(),
        })

    @app.post("/api/mail/check")
    async def mail_check() -> JSONResponse:
        service, failure = require_mail()
        return failure or JSONResponse(await service.check_all())

    @app.post("/api/mail/items/{item_id}/dismiss")
    async def mail_dismiss(item_id: int) -> JSONResponse:
        service, failure = require_mail()
        if failure:
            return failure
        service.dismiss(item_id)
        return JSONResponse({"ok": True})

    @app.post("/api/mail/accounts")
    async def mail_add(body: MailAccountCreate) -> JSONResponse:
        service, failure = require_mail()
        if failure:
            return failure
        if not body.address.strip() or not body.password.strip():
            return JSONResponse({"error": {"message": "Enter the email address and its app password."}},
                                status_code=400)
        draft = service.draft_account(body.provider, body.address, body.host, body.port, body.username)
        try:
            account = await service.add_password_account(draft, body.password.strip())
        except Exception as exc:  # noqa: BLE001 - shown on the settings page
            return JSONResponse({"error": {"message": str(exc)}}, status_code=400)
        return JSONResponse({"account": account.public()})

    @app.delete("/api/mail/accounts/{account_id}")
    async def mail_remove(account_id: int) -> JSONResponse:
        service, failure = require_mail()
        if failure:
            return failure
        service.remove_account(account_id)
        return JSONResponse({"ok": True})

    @app.post("/api/mail/outlook/start")
    async def outlook_start(body: OutlookStart) -> JSONResponse:
        service, failure = require_mail()
        if failure:
            return failure
        try:
            return JSONResponse(await service.outlook_start(body.address, body.client_id))
        except Exception as exc:  # noqa: BLE001
            return JSONResponse({"error": {"message": str(exc)}}, status_code=400)

    @app.get("/api/mail/outlook/{flow_id}")
    async def outlook_status(flow_id: str) -> JSONResponse:
        service, failure = require_mail()
        return failure or JSONResponse(service.outlook_status(flow_id))

    @app.get("/api/notify")
    async def notify_get(request: Request) -> JSONResponse:
        service, failure = require_mail()
        if failure:
            return failure
        config = service.notify_settings()
        if not is_local(request):
            config.pop("topic")
        return JSONResponse(config)

    @app.post("/api/notify")
    async def notify_set(body: NotifyUpdate) -> JSONResponse:
        service, failure = require_mail()
        return failure or JSONResponse(service.configure_notify(body.enabled, body.server, body.new_topic))

    @app.post("/api/notify/test")
    async def notify_test() -> JSONResponse:
        service, failure = require_mail()
        if failure:
            return failure
        try:
            await service.send_test()
        except Exception as exc:  # noqa: BLE001
            return JSONResponse({"error": {"message": f"Couldn't reach the ntfy server: {exc}"}}, status_code=502)
        return JSONResponse({"ok": True})

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
                app.state.brain.new_conversation()  # the thread is part of what was forgotten
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
