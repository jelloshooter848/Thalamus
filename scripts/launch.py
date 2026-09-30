"""Shared launcher logic, run by Start-THALAMUS.bat / Start-THALAMUS.command inside .venv.

Installs (or updates) THALAMUS when pyproject.toml has changed since the last install, then starts
the browser app. Standard library only: it runs before THALAMUS's dependencies exist.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAMP = Path(sys.prefix) / ".thalamus-installed"


def fingerprint() -> str:
    return hashlib.sha256((ROOT / "pyproject.toml").read_bytes()).hexdigest()


def install() -> None:
    print("  Installing THALAMUS (first run or update). This takes a minute or two...\n")
    pip = [sys.executable, "-m", "pip", "--disable-pip-version-check"]
    subprocess.run([*pip, "install", "--upgrade", "pip", "-q"], check=True)
    subprocess.run([*pip, "install", "-e", str(ROOT), "-q"], check=True)
    STAMP.write_text(fingerprint())
    print("\n  Installed.\n")


def main() -> int:
    if not STAMP.is_file() or STAMP.read_text().strip() != fingerprint():
        install()
    return subprocess.call([sys.executable, "-m", "thalamus.app"], cwd=ROOT)


if __name__ == "__main__":
    sys.exit(main())
