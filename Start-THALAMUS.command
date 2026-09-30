#!/usr/bin/env bash
# THALAMUS launcher for Mac and Linux: double-click (Mac) or run ./Start-THALAMUS.command.
# First run: creates a private Python environment, installs THALAMUS, and opens it in your browser.
cd "$(dirname "$0")" || exit 1

pause_and_exit() {
  echo
  read -r -p "  Press Enter to close this window..." _
  exit "${1:-1}"
}

echo
echo "  THALAMUS"
echo "  --------"

PY=""
for cmd in python3.13 python3.12 python3.11 python3 python; do
  if command -v "$cmd" >/dev/null 2>&1 && "$cmd" -c 'import sys; sys.exit(sys.version_info < (3, 11))' >/dev/null 2>&1; then
    PY="$cmd"
    break
  fi
done

if [ -z "$PY" ]; then
  echo "  THALAMUS needs Python 3.11 or newer, and it isn't installed yet."
  echo "  Opening python.org: install the latest Python 3, then double-click this file again."
  if command -v open >/dev/null 2>&1; then open "https://www.python.org/downloads/"
  elif command -v xdg-open >/dev/null 2>&1; then xdg-open "https://www.python.org/downloads/"; fi
  pause_and_exit 1
fi

if [ ! -x .venv/bin/python ]; then
  echo "  Creating a private Python environment (.venv)..."
  "$PY" -m venv .venv || { echo "  Couldn't create the environment."; pause_and_exit 1; }
fi

.venv/bin/python scripts/launch.py || pause_and_exit 1
