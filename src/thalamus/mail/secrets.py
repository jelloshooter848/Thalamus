"""Where email passwords and tokens live: the OS keychain (Windows Credential Manager, macOS
Keychain), or, when no keychain is available, a private file readable only by this user."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol

SERVICE = "THALAMUS"


class Secrets(Protocol):
    def get(self, name: str) -> str | None: ...
    def set(self, name: str, value: str) -> None: ...
    def delete(self, name: str) -> None: ...


class MemorySecrets:
    """In-memory store, for tests."""

    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def get(self, name: str) -> str | None:
        return self.values.get(name)

    def set(self, name: str, value: str) -> None:
        self.values[name] = value

    def delete(self, name: str) -> None:
        self.values.pop(name, None)


class FileSecrets:
    def __init__(self, path: Path) -> None:
        self.path = path

    def _load(self) -> dict[str, str]:
        try:
            return json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {}

    def _save(self, data: dict[str, str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data))
        os.chmod(self.path, 0o600)

    def get(self, name: str) -> str | None:
        return self._load().get(name)

    def set(self, name: str, value: str) -> None:
        data = self._load()
        data[name] = value
        self._save(data)

    def delete(self, name: str) -> None:
        data = self._load()
        if data.pop(name, None) is not None:
            self._save(data)


class KeyringSecrets:
    def __init__(self) -> None:
        import keyring

        self._keyring = keyring

    def get(self, name: str) -> str | None:
        return self._keyring.get_password(SERVICE, name)

    def set(self, name: str, value: str) -> None:
        self._keyring.set_password(SERVICE, name, value)

    def delete(self, name: str) -> None:
        try:
            self._keyring.delete_password(SERVICE, name)
        except self._keyring.errors.PasswordDeleteError:
            pass


def default_secrets(fallback_dir: Path) -> Secrets:
    try:
        import keyring
        from keyring.backends import fail

        if not isinstance(keyring.get_keyring(), fail.Keyring):
            return KeyringSecrets()
    except Exception:  # noqa: BLE001 - any keyring trouble means use the private file
        pass
    return FileSecrets(fallback_dir / "secrets.json")
