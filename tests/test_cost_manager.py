"""Tests for Feature 4 — Cost Manager (owner: Sid Potti)"""

import pytest
from src.cost_manager.cost_manager import (
    start_cost_monitor,
    poll_spend,
    check_budget_status,
    save_checkpoint,
    kill_job,
    generate_cost_report,
)
from src.types import BudgetStatus


def test_check_budget_status_ok_below_90_percent():
    raise NotImplementedError


def test_check_budget_status_warning_at_90_percent():
    raise NotImplementedError


def test_check_budget_status_exceeded_at_100_percent():
    raise NotImplementedError


def test_start_cost_monitor_returns_thread():
    raise NotImplementedError


def test_save_checkpoint_returns_path():
    raise NotImplementedError


def test_generate_cost_report_totals_match():
    raise NotImplementedError
