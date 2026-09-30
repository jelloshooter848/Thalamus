"""Command line: `thalamus chat`, `thalamus doctor`, `thalamus memory ...`."""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.tree import Tree

from thalamus.brain import Brain, Response, create_brain
from thalamus.config import JEV_KEY_ENV, MissingJevKeyError, Settings, load_dotenv, load_settings
from thalamus.core.memory_store import MemoryStore

app = typer.Typer(help="THALAMUS: a brain-modelled agent (JEV System 1 + Claude System 2).", no_args_is_help=True)
memory_app = typer.Typer(help="Inspect or clear long-term memory.")
app.add_typer(memory_app, name="memory")
console = Console()

ConfigOption = typer.Option(None, "--config", "-c", help="Path to thalamus.toml.")


def _settings(config: Optional[Path]) -> Settings:
    load_dotenv()
    return load_settings(config)


def _brain(settings: Settings) -> Brain:
    try:
        return create_brain(settings)
    except MissingJevKeyError as error:
        console.print(f"[bold red]{error}[/]")
        raise typer.Exit(code=2) from None


def render_trace(response: Response) -> Tree:
    tree = Tree(
        f"[bold]cycle {response.trace.turn}[/]  context=[cyan]{response.context}[/]  "
        f"action=[magenta]{response.action}[/]  path=[{'yellow' if response.path == 'slow' else 'green'}]"
        f"{response.path}[/]  cost=${response.cost_usd:.5f}"
    )
    for event in response.trace.events:
        branch = tree.add(f"[bold]{event.region}[/] · {event.event}")
        for key, value in event.data.items():
            branch.add(f"{key}: {value}")
    return tree


@app.command()
def chat(
    trace: bool = typer.Option(False, "--trace", "-t", help="Show each cognitive cycle."),
    config: Optional[Path] = ConfigOption,
) -> None:
    """Talk with THALAMUS."""
    settings = _settings(config)
    brain = _brain(settings)
    console.print("[bold]THALAMUS[/] is awake. Type 'exit' to leave.\n")

    async def loop() -> None:
        try:
            while True:
                try:
                    message = await asyncio.to_thread(console.input, "[bold blue]you ›[/] ")
                except (EOFError, KeyboardInterrupt):
                    break
                if message.strip().lower() in {"exit", "quit"}:
                    break
                if not message.strip():
                    continue
                response = await brain.think(message)
                if trace:
                    console.print(render_trace(response))
                console.print(f"[bold green]thalamus ›[/] {response.text}\n")
        finally:
            await brain.jev.aclose()
            await brain.cortex.aclose()
            brain.memory.close()

    asyncio.run(loop())


@app.command()
def doctor(config: Optional[Path] = ConfigOption) -> None:
    """Check API keys and connectivity to JEV and Claude."""
    settings = _settings(config)
    if not os.environ.get(JEV_KEY_ENV, "").strip():
        console.print(f"[red]✗[/] {MissingJevKeyError()}")
        raise typer.Exit(code=2)
    console.print(f"[green]✓[/] {JEV_KEY_ENV} is set")
    brain = _brain(settings)

    async def check() -> int:
        from typesafe_sdk import Noul

        failures = 0
        try:
            decision = await brain.jev.decide(
                {"text": "Hello there!"}, {"greeting": Noul(instructions="Is text a greeting?")}
            )
            console.print(
                f"[green]✓[/] JEV ({settings.models.jev}) answered in {decision.latency_ms:.0f} ms "
                f"(P(greeting)={decision.nouls['greeting'].p:.2f})"
            )
        except Exception as error:  # noqa: BLE001 - report any failure to the user
            failures += 1
            console.print(f"[red]✗[/] JEV: {error}")
        for tier in ("fast", "deep"):
            started = time.perf_counter()
            try:
                generation = await brain.cortex.generate(
                    tier=tier,
                    system="Reply with the single word: ready",
                    messages=[{"role": "user", "content": "Status?"}],
                    max_tokens=64 if tier == "fast" else 2048,
                )
                console.print(
                    f"[green]✓[/] Claude {tier} ({generation.model}) replied in "
                    f"{(time.perf_counter() - started) * 1000:.0f} ms"
                )
            except Exception as error:  # noqa: BLE001
                failures += 1
                console.print(f"[red]✗[/] Claude {tier}: {error}")
        await brain.jev.aclose()
        await brain.cortex.aclose()
        brain.memory.close()
        return failures

    if asyncio.run(check()):
        raise typer.Exit(code=1)


@memory_app.command("list")
def memory_list(limit: int = 20, config: Optional[Path] = ConfigOption) -> None:
    """Show recent episodes and learned values."""
    settings = _settings(config)
    store = MemoryStore(settings.memory_path)
    episodes = Table("id", "speaker", "context", "importance", "recalls", "text", title="Episodes")
    for row in store.episodes(limit):
        episodes.add_row(
            str(row["id"]), row["speaker"], row["context"], f"{row['importance']:.2f}",
            str(row["recall_count"]), row["text"][:70],
        )
    values = Table("context", "option", "value", "visits", title="Learned values (striatum)")
    for row in store.values():
        values.add_row(row["context"], row["option"], f"{row['value']:+.3f}", str(row["visits"]))
    console.print(episodes, values)
    store.close()


@memory_app.command("clear")
def memory_clear(
    yes: bool = typer.Option(False, "--yes", help="Skip confirmation."), config: Optional[Path] = ConfigOption
) -> None:
    """Erase all episodes and learned values."""
    settings = _settings(config)
    if not yes and not typer.confirm(f"Erase all memory in {settings.memory_path}?"):
        raise typer.Exit()
    store = MemoryStore(settings.memory_path)
    store.clear()
    store.close()
    console.print("Memory cleared.")


if __name__ == "__main__":
    app()
