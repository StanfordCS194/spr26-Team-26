"""
CLI-runnable scaffold for testing the AutoResearch PROPOSE phase in isolation.

Orchestrates one PROPOSE iteration: load config → generate proposal →
apply patch → log to diary. RUN/EVALUATE/DECIDE are stubbed with TODOs
pointing to the Tinker API, Evaluator sub-feature, and Cost Manager.

The full LangGraph graph (build_autoresearch_graph) supersedes this for
production use; this scaffold is useful for fast iteration without API costs.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.autoresearch.config import TrainingConfig
from src.autoresearch.diff_utils import format_patch_as_diff
from src.autoresearch.proposer import ProposalStrategy
from src.observability.observability import log_event
from src.types import AgentName, IterationRecord, LogLevel

_HISTORY_WINDOW = 10


class AutoResearchLoop:
    """Orchestrates one PROPOSE iteration. Call run_iteration() to execute a single step."""

    def __init__(
        self,
        proposer: ProposalStrategy,
        diary_path: Path,
        current_config_path: Path,
        logger: logging.Logger | None = None,
    ) -> None:
        self.proposer = proposer
        self.diary_path = Path(diary_path)
        self.current_config_path = Path(current_config_path)
        self.logger = logger or logging.getLogger("autoresearch.loop")
        self.diary_path.parent.mkdir(parents=True, exist_ok=True)

    def run_iteration(self) -> None:
        """
        Runs one PROPOSE iteration: loads config, generates a proposal, validates it,
        and appends a PENDING entry to the diary.
        """
        config = TrainingConfig.load(self.current_config_path)
        history = self._load_history()
        iteration = len(history) + 1

        log_event(
            AgentName.AUTORESEARCH,
            LogLevel.INFO,
            f"Starting iteration {iteration}",
            metadata={
                "iteration": iteration,
                "strategy": type(self.proposer).__name__,
            },
        )

        proposal = self.proposer.propose(config, history)

        if not proposal.patch:
            raise ValueError(
                f"{type(self.proposer).__name__}.propose() returned an empty patch. "
                "Every proposal must change at least one hyperparameter."
            )

        # Validate the patch before writing anything
        candidate = config.apply_patch(proposal.patch)
        diff = format_patch_as_diff(proposal.patch, config)

        entry = self._make_pending_entry(iteration, proposal.hypothesis, diff)
        self._append_diary(entry)

        param = next(iter(proposal.patch))
        new_val = proposal.patch[param]

        log_event(
            AgentName.AUTORESEARCH,
            LogLevel.INFO,
            f"Iteration {iteration}: {proposal.hypothesis}",
            metadata={
                "iteration": iteration,
                "param": param,
                "old_value": getattr(config, param, None),
                "new_value": new_val,
                "strategy": proposal.metadata.get("strategy"),
                "seed": proposal.metadata.get("seed"),
                "candidate_config": candidate.to_dict(),
            },
        )

        print(f"\n[AutoResearch] Iteration {iteration}: Testing {param}={new_val}")
        print(f"[AutoResearch] Hypothesis: {proposal.hypothesis}")
        print(f"[AutoResearch] Patch: {json.dumps(proposal.patch)}")
        print(f"[AutoResearch] Diff:\n{diff}\n")

    def run_experiment(self, config: TrainingConfig) -> None:
        """
        TODO: Submit config to Tinker and block until the job finishes.

        When wired up: serialise config → tinker_api.submit_job() → poll status →
        respect CostManager budget → return ExperimentResult.
        See src/tinker_api/tinker_api.py and src/cost_manager/cost_manager.py.
        """
        raise NotImplementedError("Requires Tinker job submission (F4).")

    def evaluate_experiment(self, experiment_result: Any) -> None:
        """
        TODO: Score the trained model using run_evals() and compute a ScoreDelta.

        When wired up: run_evals(model_path, eval_suite) → compare_scores() →
        flag_regression() → return (EvalScore, ScoreDelta).
        See src/autoresearch/autoresearch.py.
        """
        raise NotImplementedError("Requires Evaluator sub-feature.")

    def decide(self, score_delta: Any, diary_entry: IterationRecord) -> None:
        """
        TODO: Commit or roll back the patch and update the diary entry.

        When wired up: decide_keep_or_revert(delta) → overwrite config or
        call revert_patch() → update diary entry from PENDING → KEPT/REVERTED.
        See src/autoresearch/autoresearch.py and diff_utils.py.
        """
        raise NotImplementedError("Requires EVALUATE phase and Cost Manager (F4).")

    def _load_history(self) -> list[IterationRecord]:
        """Loads the last _HISTORY_WINDOW entries from the diary. Silently skips corrupted lines."""
        if not self.diary_path.exists():
            return []
        entries: list[IterationRecord] = []
        with open(self.diary_path) as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    self.logger.warning(
                        "Skipping corrupted diary line %d in %s",
                        lineno,
                        self.diary_path,
                    )
        return entries[-_HISTORY_WINDOW:]

    def _make_pending_entry(
        self, iteration: int, hypothesis: str, diff: str
    ) -> IterationRecord:
        return {
            "iteration": iteration,
            "hypothesis": hypothesis,
            "patch": diff,
            "cost_usd": 0.0,
            "metrics": {
                "train_loss": None,   # type: ignore[typeddict-item]
                "val_loss": None,     # type: ignore[typeddict-item]
                "test_loss": None,    # type: ignore[typeddict-item]
                "primary_metric": None,  # type: ignore[typeddict-item]
            },
            "decision": "PENDING",
            "notes": f"Proposed at {datetime.now(timezone.utc).isoformat()}",
        }

    def _append_diary(self, entry: IterationRecord) -> None:
        with open(self.diary_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
