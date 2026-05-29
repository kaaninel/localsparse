"""LocalSparse CLI.

Subcommands:
    localsparse chat                      interactive REPL (needs a backend)
    localsparse wks list                  list workspaces
    localsparse wks create NAME [TEXT]    create + optional seed text
    localsparse wks append NAME TEXT      append to a workspace
    localsparse wks delete NAME
    localsparse wks pin NAME [-w WEIGHT]
    localsparse wks unpin NAME
    localsparse wks consolidate SRC REGION DST [--mode research|rewrite]
    localsparse info                      print env + config summary
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .config import LocalSparseConfig, default_config
from .agent import LocalSparseAgent
from .workspace import WorkspaceManager, ConsolidationOrchestrator
from .workspace.consolidation import MockSearcher


app = typer.Typer(help="LocalSparse: small dense LLM with workspace-extended context.")
wks_app = typer.Typer(help="Workspace commands.")
app.add_typer(wks_app, name="wks")

console = Console()


def _agent() -> LocalSparseAgent:
    return LocalSparseAgent(config=default_config())


# ---------------------------------------------------------------------------
# wks subcommands
# ---------------------------------------------------------------------------
@wks_app.command("list")
def wks_list():
    with _agent() as a:
        items = a.manager.list()
        if not items:
            console.print("[dim]no workspaces[/dim]")
            return
        t = Table(title="Workspaces")
        for col in ("name", "slots", "cap", "tier", "pinned", "last_used"):
            t.add_column(col)
        for m in items:
            tier = ("L2" if m.tier_flags == 0b001
                    else "L1" if m.tier_flags == 0b011
                    else "L0")
            t.add_row(
                m.name, f"{m.slot_count:,}", f"{m.slot_cap:,}",
                tier, "✓" if m.pinned else "",
                f"{m.last_used_at:.0f}",
            )
        console.print(t)


@wks_app.command("create")
def wks_create(name: str, text: Optional[str] = typer.Argument(None)):
    with _agent() as a:
        meta = a.manager.create(name, source=text)
        console.print(f"created [bold]{meta.name}[/bold] (slots={meta.slot_count})")


@wks_app.command("append")
def wks_append(name: str, text: str):
    with _agent() as a:
        start, end = a.manager.append(name, text)
        console.print(f"appended {end - start} slots to [bold]{name}[/bold] (now {end})")


@wks_app.command("delete")
def wks_delete(name: str):
    with _agent() as a:
        a.manager.delete(name)
        console.print(f"deleted [bold]{name}[/bold]")


@wks_app.command("pin")
def wks_pin(name: str, weight: float = typer.Option(1.0, "-w")):
    with _agent() as a:
        a.manager.pin(name, weight=weight)
        console.print(f"pinned [bold]{name}[/bold] (weight={weight})")


@wks_app.command("unpin")
def wks_unpin(name: str):
    with _agent() as a:
        a.manager.unpin(name)
        console.print(f"unpinned [bold]{name}[/bold]")


@wks_app.command("consolidate")
def wks_consolidate(src: str, region: str, dst: str,
                    mode: str = typer.Option("research", "--mode")):
    with _agent() as a:
        res = a.orchestrator.consolidate(src=src, region=region, dst=dst, mode=mode)
        console.print(
            f"consolidated {src}/{region} → {dst}: "
            f"cid={res.consolidation_id} +{res.appended_tokens} slots")


@wks_app.command("candidates")
def wks_candidates():
    with _agent() as a:
        cands = a.orchestrator.pending_candidates()
        if not cands:
            console.print("[dim]no pending candidates[/dim]")
            return
        t = Table(title="Consolidation candidates")
        for col in ("src", "region", "dst", "hits", "first_seen"):
            t.add_column(col)
        for c in cands:
            t.add_row(c.src_wks, c.src_region, c.dst_wks, str(c.hits),
                      f"{c.first_seen_at:.0f}")
        console.print(t)


# ---------------------------------------------------------------------------
# top-level commands
# ---------------------------------------------------------------------------
@app.command()
def info():
    cfg = default_config()
    console.print("[bold]LocalSparse config[/bold]")
    console.print(cfg.to_json())


@app.command()
def chat(model_path: Optional[Path] = typer.Option(None, "--model", "-m"),
         max_new_tokens: int = 512):
    """Interactive REPL chat. Requires `--model` to be supplied with a
    weights directory loadable via HuggingFace transformers, or omit to
    run in dry-run mode that prints prompts without generating."""
    with _agent() as a:
        backend = _load_backend(model_path)
        a.backend = backend
        console.print("[bold green]LocalSparse chat[/bold green] — type 'exit' to quit.")
        while True:
            try:
                msg = console.input("[bold cyan]you[/bold cyan] » ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not msg or msg.lower() in {"exit", "quit"}:
                break
            try:
                resp = a.chat(msg)
            except RuntimeError as e:
                console.print(f"[red]{e}[/red]")
                continue
            console.print(f"[bold magenta]agent[/bold magenta] » {resp}")


def _load_backend(model_path: Optional[Path]):
    if model_path is None:
        console.print("[yellow]No --model given; using dry-run backend.[/yellow]")
        from .agent import MockBackend
        return MockBackend(["[no model loaded; this is a dry-run reply]"])
    try:
        from .model.hf_backend import HFBackend
    except ImportError as e:
        console.print(f"[red]transformers backend not available: {e}[/red]")
        raise typer.Exit(1)
    return HFBackend(model_path)


if __name__ == "__main__":
    app()
