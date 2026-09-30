"""Phone access: reach the THALAMUS running on this PC from your phone over Tailscale.

Security model:
- The PC itself (loopback) needs no login, exactly as before.
- Other devices are only served when phone access is on, only from Tailscale's private address
  ranges (your own devices; everything is WireGuard-encrypted), and only after entering a passcode.
- The passcode is stored as a salted scrypt hash. A successful login sets a signed, HttpOnly cookie
  that lasts `session_days`. Changing the passcode rotates the signing secret, which logs out every
  device.
- Repeated wrong passcodes lock that address out for a while.
- Settings and keys can only be changed from the PC.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import ipaddress
import json
import os
import platform
import secrets
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field

REMOTE_ENV = "THALAMUS_REMOTE"
PASSCODE_ENV = "THALAMUS_PASSCODE_HASH"
SECRET_ENV = "THALAMUS_SECRET"
COOKIE = "thalamus_session"
MIN_PASSCODE = 6

TAILSCALE_NETS = (ipaddress.ip_network("100.64.0.0/10"), ipaddress.ip_network("fd7a:115c:a1e0::/48"))
LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient"}


def remote_enabled() -> bool:
    return os.environ.get(REMOTE_ENV, "").strip() == "1"


# ----- passcode -------------------------------------------------------------------------------
def hash_passcode(passcode: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(passcode.encode(), salt=salt, n=2**14, r=8, p=1)
    return f"scrypt${base64.b64encode(salt).decode()}${base64.b64encode(digest).decode()}"


def check_passcode(passcode: str, stored: str) -> bool:
    try:
        scheme, salt, digest = stored.split("$")
        if scheme != "scrypt":
            return False
        candidate = hashlib.scrypt(passcode.encode(), salt=base64.b64decode(salt), n=2**14, r=8, p=1)
        return hmac.compare_digest(candidate, base64.b64decode(digest))
    except (ValueError, TypeError):
        return False


# ----- sessions -------------------------------------------------------------------------------
def make_session(secret: str, days: float) -> str:
    expires = str(int(time.time() + days * 86400))
    signature = hmac.new(secret.encode(), expires.encode(), hashlib.sha256).hexdigest()
    return f"{expires}.{signature}"


def valid_session(cookie: str | None, secret: str) -> bool:
    if not cookie or not secret or "." not in cookie:
        return False
    expires, signature = cookie.split(".", 1)
    expected = hmac.new(secret.encode(), expires.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(signature, expected) and expires.isdigit() and int(expires) > time.time()


# ----- who is asking --------------------------------------------------------------------------
def is_loopback(host: str | None) -> bool:
    if not host:
        return False
    if host in LOOPBACK_HOSTS:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def allowed_remote(host: str | None, allow_lan: bool = False) -> bool:
    try:
        address = ipaddress.ip_address(host or "")
    except ValueError:
        return False
    if any(address in net for net in TAILSCALE_NETS):
        return True
    return allow_lan and address.is_private


@dataclass
class LoginThrottle:
    """Lock an address out after repeated wrong passcodes."""

    max_failures: int = 5
    lockout_seconds: float = 300.0
    failures: dict[str, list[float]] = field(default_factory=dict)

    def locked(self, host: str) -> bool:
        recent = [t for t in self.failures.get(host, []) if time.time() - t < self.lockout_seconds]
        self.failures[host] = recent
        return len(recent) >= self.max_failures

    def fail(self, host: str) -> None:
        self.failures.setdefault(host, []).append(time.time())

    def succeed(self, host: str) -> None:
        self.failures.pop(host, None)


# ----- finding this PC on the tailnet ----------------------------------------------------------
def _tailscale_cli() -> str | None:
    found = shutil.which("tailscale")
    if found:
        return found
    for candidate in (
        r"C:\Program Files\Tailscale\tailscale.exe",
        "/Applications/Tailscale.app/Contents/MacOS/Tailscale",
    ):
        if os.path.exists(candidate):
            return candidate
    return None


def tailscale_info() -> dict:
    """This PC's Tailscale address and name, if Tailscale is installed and connected."""
    info: dict = {"installed": False, "ip": None, "name": None}
    cli = _tailscale_cli()
    if cli:
        info["installed"] = True
        try:
            flags = {"creationflags": 0x08000000} if sys.platform == "win32" else {}  # no console flash
            out = subprocess.run([cli, "status", "--json"], capture_output=True, text=True, timeout=5, **flags)
            status = json.loads(out.stdout or "{}")
            me = status.get("Self") or {}
            ips = [ip for ip in me.get("TailscaleIPs", []) if ":" not in ip]
            info["ip"] = ips[0] if ips else None
            info["name"] = (me.get("DNSName") or "").rstrip(".") or None
            info["running"] = status.get("BackendState") == "Running"
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    if not info["ip"]:  # fall back to asking the OS which local address would reach the tailnet
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
                probe.connect(("100.100.100.100", 53))
                address = probe.getsockname()[0]
            if allowed_remote(address):
                info["ip"] = address
        except OSError:
            pass
    return info


def phone_urls(port: int, info: dict) -> list[str]:
    urls = []
    if info.get("name"):
        urls.append(f"http://{info['name']}:{port}/")
    if info.get("ip"):
        urls.append(f"http://{info['ip']}:{port}/")
    return urls


def qr_svg(text: str) -> str:
    import io

    import segno

    buffer = io.BytesIO()
    segno.make(text, error="m").save(buffer, kind="svg", scale=5, border=2, dark="#1b1b24", light="#ffffff")
    return buffer.getvalue().decode()


# ----- keep the PC awake while phone access is on ----------------------------------------------
class KeepAwake:
    def __init__(self) -> None:
        self._process: subprocess.Popen | None = None
        self.active = False

    def start(self) -> None:
        system = platform.system()
        try:
            if system == "Windows":
                import ctypes

                es_continuous, es_system_required = 0x80000000, 0x00000001
                ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required)
                self.active = True
            elif system == "Darwin" and shutil.which("caffeinate"):
                self._process = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
                self.active = True
            elif shutil.which("systemd-inhibit"):
                self._process = subprocess.Popen(
                    ["systemd-inhibit", "--what=sleep", "--why=THALAMUS phone access", "sleep", "infinity"]
                )
                self.active = True
        except (OSError, AttributeError):
            self.active = False

    def stop(self) -> None:
        if self._process is not None:
            self._process.terminate()
            self._process = None
        if platform.system() == "Windows" and self.active:
            import ctypes

            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        self.active = False
