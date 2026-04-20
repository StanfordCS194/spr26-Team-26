"""
Feature 4 — Cost Manager
Owner: Sid Potti

Background thread that polls Tinker billing every 30s.
Saves checkpoint at 90% budget; kills job at 100%.
Plain Python — no LangGraph.
"""

from __future__ import annotations

import threading

from src.types import BudgetStatus, CostBreakdown


def start_cost_monitor(
    job_id: str,
    budget: float,
    poll_interval_sec: int = 30,
) -> threading.Thread:
    """Starts a background thread that polls Tinker billing every poll_interval_sec seconds.
    Calls save_checkpoint at 90% and kill_job at 100%. Returns the running thread."""
    raise NotImplementedError


def poll_spend(job_id: str) -> float:
    """Calls Tinker billing API to fetch cumulative USD spend for a job."""
    raise NotImplementedError


def check_budget_status(spent: float, budget: float) -> str:
    """Returns BudgetStatus: OK (< 90%), WARNING (90–99%), EXCEEDED (>= 100%)."""
    raise NotImplementedError


def save_checkpoint(job_id: str, output_dir: str) -> str:
    """Saves current model state_dict to disk. Returns absolute path to checkpoint file."""
    raise NotImplementedError


def kill_job(job_id: str) -> None:
    """Calls Tinker API to immediately terminate the GPU instance for job_id."""
    raise NotImplementedError


def generate_cost_report(job_id: str) -> CostBreakdown:
    """Fetches final cost breakdown from Tinker. Splits into data_gen, training, llm_calls components."""
    raise NotImplementedError


class CostManager:
    """Convenience wrapper used by orchestrate_node to pass a cost manager instance to AutoResearch."""

    def __init__(self, budget: float):
        self.budget = budget
        self._thread: threading.Thread | None = None

    def start(self, job_id: str) -> None:
        self._thread = start_cost_monitor(job_id, self.budget)

    def stop(self) -> None:
        if self._thread:
            self._thread.join(timeout=5)
