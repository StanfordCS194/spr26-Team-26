"""
Feature 5 — Observability
Owner: Team

Shared utility module — every agent imports and calls log_event().
No agent writes to stdout or disk directly.

Designed so swapping to a fully structured JSON-only handler later is a
one-line change: replace emit_cli with a JSON-only emitter and nothing else
changes.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.types import LogEntry

# ANSI color codes by agent name
_AGENT_COLORS: dict[str, str] = {
    "Manager":        "\033[34m",   # blue
    "DataGen":        "\033[36m",   # cyan
    "DecisionEngine": "\033[35m",   # magenta
    "AutoResearch":   "\033[33m",   # yellow
    "CostManager":    "\033[31m",   # red
    "TinkerAPI":      "\033[32m",   # green
}
_LEVEL_COLORS: dict[str, str] = {
    "INFO":  "\033[37m",   # white
    "WARN":  "\033[33m",   # yellow
    "ERROR": "\033[31m",   # red
}
_RESET = "\033[0m"


def log_event(
    agent: str,
    level: str,
    message: str,
    metadata: dict = {},
    log_path: str = "outputs/logs/run.jsonl",
) -> None:
    """Central logging function called by every agent. Writes JSON to disk and colored line to stdout."""
    entry: LogEntry = {
        "agent": agent,
        "level": level,
        "message": message,
        "metadata": metadata,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    emit_cli(entry)
    write_json_log(entry, log_path)


def format_cli_line(entry: LogEntry) -> str:
    """Returns formatted CLI string: [AgentName] LEVEL timestamp — message."""
    agent_color = _AGENT_COLORS.get(entry["agent"], "")
    level_color = _LEVEL_COLORS.get(entry["level"], "")
    ts = entry["timestamp"][:19].replace("T", " ")
    return (
        f"{agent_color}[{entry['agent']}]{_RESET} "
        f"{level_color}{entry['level']}{_RESET} "
        f"{ts} — {entry['message']}"
    )


def emit_cli(entry: LogEntry) -> None:
    """Prints the formatted CLI line to stdout with ANSI color coding by agent and level."""
    print(format_cli_line(entry), file=sys.stdout, flush=True)


def write_json_log(entry: LogEntry, log_path: str) -> None:
    """Appends the LogEntry as a JSON line to the structured log file."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def get_budget_display(spent: float, budget: float) -> str:
    """Returns a formatted budget string. e.g. 'Spend: $9.20 / $50.00 (18% used)'."""
    pct = (spent / budget * 100) if budget > 0 else 0.0
    return f"Spend: ${spent:.2f} / ${budget:.2f} ({pct:.0f}% used)"
