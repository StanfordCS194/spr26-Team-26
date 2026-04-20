"""
Feature 5 — Observability
Owner: Team

Shared utility module — every agent imports and calls log_event().
No agent writes to stdout or disk directly.
"""

from __future__ import annotations

from src.types import LogEntry


def log_event(
    agent: str,
    level: str,
    message: str,
    metadata: dict = {},
    log_path: str = "outputs/logs/run.jsonl",
) -> None:
    """Central logging function called by every agent. Writes JSON to disk and colored line to stdout."""
    raise NotImplementedError


def format_cli_line(entry: LogEntry) -> str:
    """Returns formatted CLI string: [AgentName] message."""
    raise NotImplementedError


def emit_cli(entry: LogEntry) -> None:
    """Prints the formatted CLI line to stdout with ANSI color coding by agent and level."""
    raise NotImplementedError


def write_json_log(entry: LogEntry, log_path: str) -> None:
    """Appends the LogEntry as a JSON line to the structured log file."""
    raise NotImplementedError


def get_budget_display(spent: float, budget: float) -> str:
    """Returns a formatted budget string. e.g. 'Spend: $9.20 / $50.00 (18% used)'."""
    raise NotImplementedError
