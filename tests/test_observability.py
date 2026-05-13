"""Tests for Feature 5 — Observability (owner: Team)"""

import json
import pytest
from src.observability.observability import (
    log_event,
    format_cli_line,
    emit_cli,
    write_json_log,
    get_budget_display,
)
from src.types import AgentName, LogLevel


def _make_entry(agent=AgentName.MANAGER, level=LogLevel.INFO, message="test msg"):
    return {
        "agent": agent,
        "level": level,
        "message": message,
        "metadata": {},
        "timestamp": "2026-05-08T00:00:00+00:00",
    }


def test_format_cli_line_includes_agent_name():
    entry = _make_entry(agent=AgentName.MANAGER, message="hello")
    line = format_cli_line(entry)
    assert "Manager" in line
    assert "hello" in line


def test_get_budget_display_formats_correctly():
    result = get_budget_display(9.20, 50.0)
    assert "$9.20" in result
    assert "$50.00" in result
    assert "18%" in result


def test_write_json_log_appends_valid_json(tmp_path):
    log_file = str(tmp_path / "run.jsonl")
    entry = _make_entry(message="line one")
    write_json_log(entry, log_file)
    write_json_log(_make_entry(message="line two"), log_file)

    lines = open(log_file).readlines()
    assert len(lines) == 2
    parsed = json.loads(lines[0])
    assert parsed["message"] == "line one"
    assert parsed["agent"] == AgentName.MANAGER


def test_log_event_writes_to_disk(tmp_path):
    log_file = str(tmp_path / "run.jsonl")
    log_event(AgentName.DATA_GEN, LogLevel.INFO, "dataset found", log_path=log_file)

    lines = open(log_file).readlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["agent"] == AgentName.DATA_GEN
    assert parsed["message"] == "dataset found"
    assert "timestamp" in parsed
