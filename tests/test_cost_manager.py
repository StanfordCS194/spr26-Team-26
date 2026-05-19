"""Tests for Feature 4 — Cost Manager (owner: Sid Potti)"""

import threading
import pytest
from unittest.mock import patch
from src.cost_manager.cost_manager import (
    CostManager,
    check_budget_status,
    generate_cost_report,
    save_checkpoint,
    start_cost_monitor,
)
from src.types import BudgetStatus


def test_check_budget_status_ok_below_90_percent():
    assert check_budget_status(40.0, 100.0) == BudgetStatus.OK
    assert check_budget_status(0.0, 100.0) == BudgetStatus.OK
    assert check_budget_status(89.9, 100.0) == BudgetStatus.OK


def test_check_budget_status_warning_at_90_percent():
    assert check_budget_status(90.0, 100.0) == BudgetStatus.WARNING
    assert check_budget_status(95.0, 100.0) == BudgetStatus.WARNING
    assert check_budget_status(99.9, 100.0) == BudgetStatus.WARNING


def test_check_budget_status_exceeded_at_100_percent():
    assert check_budget_status(100.0, 100.0) == BudgetStatus.EXCEEDED
    assert check_budget_status(110.0, 100.0) == BudgetStatus.EXCEEDED


def test_start_cost_monitor_returns_thread():
    with patch("src.cost_manager.cost_manager.poll_spend", return_value=0.0):
        thread = start_cost_monitor("job-123", 50.0, poll_interval_sec=999)
    assert isinstance(thread, threading.Thread)
    assert thread.is_alive()
    thread.stop_event.set()  # type: ignore[attr-defined]
    thread.join(timeout=1)


def test_cost_manager_stop_signals_monitor_thread():
    with patch("src.cost_manager.cost_manager.poll_spend", return_value=0.0):
        manager = CostManager(50.0)
        manager.start("job-456")
        thread = manager._thread
        assert isinstance(thread, threading.Thread)
        assert thread.is_alive()

        manager.stop()

    assert manager._thread is None
    assert not thread.is_alive()


def test_save_checkpoint_returns_path(tmp_path):
    path = save_checkpoint("job-abc", str(tmp_path))
    assert path.endswith("checkpoint_job-abc.pt")
    signal = tmp_path / ".save_signal_job-abc"
    assert signal.exists()


def test_generate_cost_report_totals_match():
    with patch("src.tinker_api.tinker_api.get_cumulative_spend", return_value=20.0):
        report = generate_cost_report("job-xyz")
    assert abs(report["total_usd"] - 20.0) < 0.01
    assert report["training_usd"] > report["llm_calls_usd"]
    assert report["termination_reason"] == "training_complete"
