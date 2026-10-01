"""Phone notifications through ntfy (free: install the ntfy app and subscribe to your topic)."""

from __future__ import annotations

import httpx

DEFAULT_SERVER = "https://ntfy.sh"


class Ntfy:
    def __init__(self, topic: str, server: str = DEFAULT_SERVER, client: httpx.AsyncClient | None = None) -> None:
        self.topic = topic
        self.server = server.rstrip("/")
        self._client = client

    async def send(self, title: str, message: str, *, priority: int = 3, click: str | None = None,
                   tags: str = "") -> None:
        headers = {"Title": title.encode("utf-8").decode("latin-1", "replace"), "Priority": str(priority)}
        if click:
            headers["Click"] = click
        if tags:
            headers["Tags"] = tags
        client = self._client or httpx.AsyncClient(timeout=15)
        try:
            response = await client.post(f"{self.server}/{self.topic}", content=message.encode(), headers=headers)
            response.raise_for_status()
        finally:
            if self._client is None:
                await client.aclose()
