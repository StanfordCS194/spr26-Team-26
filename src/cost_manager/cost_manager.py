"""
Feature 4 — Cost Manager
Owner: Sid Potti

Background thread that polls Tinker billing every 30s.
Saves checkpoint at 90% budget; kills job at 100%.
Plain Python — no LangGraph.
"""

from __future__ import annotations

import os
import threading

from src.types import BudgetStatus, CostBreakdown


def start_cost_monitor(
    job_id: str,
    budget: float,
    poll_interval_sec: int = 30,
    output_dir: str = "outputs/checkpoints",
) -> threading.Thread:
    """Starts a background thread that polls Tinker billing every poll_interval_sec seconds.
    Saves checkpoint at 90% budget and kills job at 100%. Returns the running thread."""

    stop_event = threading.Event()

    def _monitor():
        while not stop_event.is_set():
            try:
                spent = poll_spend(job_id)
                status = check_budget_status(spent, budget)
                if status == BudgetStatus.WARNING:
                    save_checkpoint(job_id, output_dir)
                elif status == BudgetStatus.EXCEEDED:
                    save_checkpoint(job_id, output_dir)
                    kill_job(job_id)
                    stop_event.set()
                    return
            except Exception:
                pass  # don't crash the monitor on transient errors
            if stop_event.wait(poll_interval_sec):
                return

    thread = threading.Thread(target=_monitor, daemon=True, name=f"cost-monitor-{job_id}")
    thread.stop_event = stop_event  # type: ignore[attr-defined]
    thread.start()
    return thread


def poll_spend(job_id: str) -> float:
    """Calls Tinker billing API to fetch cumulative USD spend for a job."""
    from src.tinker_api.tinker_api import get_cumulative_spend
    return get_cumulative_spend(job_id)


def check_budget_status(spent: float, budget: float) -> str:
    """Returns BudgetStatus: OK (< 90%), WARNING (90–99%), EXCEEDED (>= 100%)."""
    if budget <= 0:
        return BudgetStatus.EXCEEDED if spent > 0 else BudgetStatus.OK
    ratio = spent / budget if budget > 0 else 0.0
    if ratio >= 1.0:
        return BudgetStatus.EXCEEDED
    if ratio >= 0.9:
        return BudgetStatus.WARNING
    return BudgetStatus.OK


def save_checkpoint(job_id: str, output_dir: str) -> str:
    """Signals the training process to save a checkpoint. Returns the checkpoint path."""
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, f"checkpoint_{job_id}.pt")
    # Write a signal file that train.py polls for — the actual state_dict save
    # happens inside the training script when it detects this file.
    signal_path = os.path.join(output_dir, f".save_signal_{job_id}")
    with open(signal_path, "w") as f:
        f.write("save")
    return os.path.abspath(checkpoint_path)


def kill_job(job_id: str) -> None:
    """Calls Tinker API to immediately terminate the GPU instance for job_id."""
    from src.tinker_api.tinker_api import cancel_job
    cancel_job(job_id)


def generate_cost_report(job_id: str) -> CostBreakdown:
    """Fetches final cost breakdown from Tinker and returns a CostBreakdown."""
    from src.tinker_api.tinker_api import get_cumulative_spend
    total = get_cumulative_spend(job_id)
    # Approximate breakdown: 80% training, 15% data gen, 5% LLM calls
    return CostBreakdown(
        data_gen_usd=round(total * 0.15, 4),
        training_usd=round(total * 0.80, 4),
        llm_calls_usd=round(total * 0.05, 4),
        total_usd=round(total, 4),
        termination_reason="training_complete",
    )


class CostManager:
    """Convenience wrapper used by orchestrate_node to pass a cost manager instance to AutoResearch."""

    def __init__(self, budget: float):
        self.budget = budget
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._category_totals: dict[str, float] = {}

    @property
    def category_totals(self) -> dict[str, float]:
        """Returns a snapshot of in-process spend by category."""
        with self._lock:
            return dict(self._category_totals)

    @property
    def spent_usd(self) -> float:
        """Returns total in-process spend recorded through this manager."""
        with self._lock:
            return round(sum(self._category_totals.values()), 6)

    @property
    def remaining_budget(self) -> float:
        """Returns the remaining budget after in-process spend."""
        return max(0.0, round(self.budget - self.spent_usd, 6))

    @property
    def status(self) -> str:
        """Returns the current in-process budget status."""
        return check_budget_status(self.spent_usd, self.budget)

    def can_start_run(self, estimated_cost: float = 0.0) -> bool:
        """Returns whether a new run can start without exceeding remaining budget."""
        try:
            estimate = max(0.0, float(estimated_cost))
        except (TypeError, ValueError):
            estimate = 0.0

        if self.status == BudgetStatus.EXCEEDED:
            return False
        if self.remaining_budget <= 0:
            return False
        return estimate <= self.remaining_budget

    def record_spend(self, amount: float, category: str = "training") -> str:
        """Records completed in-process spend and returns the updated budget status."""
        amount = float(amount)
        if amount < 0:
            raise ValueError("CostManager.record_spend amount must be non-negative")
        with self._lock:
            self._category_totals[category] = (
                self._category_totals.get(category, 0.0) + amount
            )
        return self.status

    def cost_breakdown(self, termination_reason: str | None = None) -> CostBreakdown:
        """Returns a CostBreakdown from in-process spend totals."""
        totals = self.category_totals
        data_gen = totals.get("data_gen", 0.0) + totals.get("data_generation", 0.0)
        training = totals.get("training", 0.0)
        llm_calls = totals.get("llm_calls", 0.0) + totals.get("llm", 0.0)
        total = self.spent_usd
        other = max(0.0, total - data_gen - training - llm_calls)
        training += other
        reason = termination_reason or (
            "budget_limit" if self.status == BudgetStatus.EXCEEDED else "training_complete"
        )
        return {
            "data_gen_usd": round(data_gen, 6),
            "training_usd": round(training, 6),
            "llm_calls_usd": round(llm_calls, 6),
            "total_usd": total,
            "termination_reason": reason,
        }

    def start(self, job_id: str) -> None:
        self.stop()
        monitor_budget = self.remaining_budget if self.spent_usd > 0 else self.budget
        self._thread = start_cost_monitor(job_id, monitor_budget)

    def stop(self) -> None:
        if self._thread:
            stop_event = getattr(self._thread, "stop_event", None)
            if stop_event is not None:
                stop_event.set()
            self._thread.join(timeout=5)
            self._thread = None
