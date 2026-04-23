"""
AutoResearchLoop — PROPOSE-phase scaffold.

Orchestrates one iteration of the PROPOSE step:
  1. Load current TrainingConfig from disk.
  2. Load recent ResearchDiary history.
  3. Call ProposalStrategy.propose() → Proposal.
  4. apply_patch() to produce a candidate TrainingConfig.
  5. format_patch_as_diff() for human-readable logging.
  6. Append a PENDING diary entry to outputs/logs/research_diary.jsonl.
  7. Emit structured logs via Observability (F5).

RUN / EVALUATE / DECIDE are stubbed with explicit TODO docstrings that
reference the Tinker API (F4), the Evaluator sub-feature, and the Cost
Manager so future implementers know exactly what to wire in.

Relationship to the LangGraph graph (autoresearch.py):
  This loop is the CLI-runnable scaffold used for testing the PROPOSE
  infra in isolation. The full LangGraph graph (build_autoresearch_graph)
  will call propose_node → run_node → evaluate_node → decide_node → log_node
  in a cycle. propose_node internally calls propose_hypothesis() which uses
  Claude; this scaffold uses ProposalStrategy directly (no API cost).
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
    """
    Thin orchestrator for the AutoResearch PROPOSE phase.

    Initialised with a ProposalStrategy and two paths (diary, config).
    Call run_iteration() to run one PROPOSE step.
    """

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

    # ──────────────────────────────────────────────────────────────────────────
    # PROPOSE (implemented)
    # ──────────────────────────────────────────────────────────────────────────

    def run_iteration(self) -> None:
        """
        Run a single PROPOSE-phase iteration.

        Side effects:
          - Appends one PENDING IterationRecord to research_diary.jsonl.
          - Appends one LogEntry to outputs/logs/run.jsonl.
          - Prints colored status to stdout.
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

        # Human-readable stdout summary (in addition to structured log)
        print(f"\n[AutoResearch] Iteration {iteration}: Testing {param}={new_val}")
        print(f"[AutoResearch] Hypothesis: {proposal.hypothesis}")
        print(f"[AutoResearch] Patch: {json.dumps(proposal.patch)}")
        print(f"[AutoResearch] Diff:\n{diff}\n")

    # ──────────────────────────────────────────────────────────────────────────
    # RUN (stub)
    # ──────────────────────────────────────────────────────────────────────────

    def run_experiment(self, config: TrainingConfig) -> None:
        """
        TODO: Submit job to Tinker and block until completion.

        Not implemented in the PROPOSE-only phase.

        When RUN is ready, this should:
          1. Serialise config to a temp JSON and pass to tinker_api.submit_job().
          2. Poll tinker_api.get_job_status(job_id) until COMPLETED or FAILED.
          3. Respect Cost Manager (F4): if CostManager signals budget exceeded,
             call tinker_api.cancel_job(job_id) and raise BudgetExceededError.
          4. Return ExperimentResult (job_id, metrics, model_path, cost_usd).

        See: src/tinker_api/tinker_api.py, src/cost_manager/cost_manager.py.
        """
        raise NotImplementedError(
            "run_experiment not yet implemented. Requires Tinker job submission (F4)."
        )

    # ──────────────────────────────────────────────────────────────────────────
    # EVALUATE (stub)
    # ──────────────────────────────────────────────────────────────────────────

    def evaluate_experiment(self, experiment_result: Any) -> None:
        """
        TODO: Use the Evaluator sub-feature to compute metrics and a scalar score.

        Not implemented in the PROPOSE-only phase.

        When EVALUATE is ready, this should:
          1. Call autoresearch.run_evals(model_path, eval_suite) → EvalScore.
          2. Call autoresearch.compare_scores(new_score, baseline_score) → ScoreDelta.
          3. Call autoresearch.flag_regression(delta) — if True, skip DECIDE and revert.
          4. Return (EvalScore, ScoreDelta) for pass to decide().

        See: src/autoresearch/autoresearch.py — run_evals, compare_scores, flag_regression.
        """
        raise NotImplementedError(
            "evaluate_experiment not yet implemented. Requires Evaluator sub-feature."
        )

    # ──────────────────────────────────────────────────────────────────────────
    # DECIDE (stub)
    # ──────────────────────────────────────────────────────────────────────────

    def decide(self, score_delta: Any, diary_entry: IterationRecord) -> None:
        """
        TODO: Decide KEPT/REVERTED, merge diffs, update research diary and current config.

        Not implemented in the PROPOSE-only phase.

        When DECIDE is ready, this should:
          1. Call autoresearch.decide_keep_or_revert(delta) → 'KEEP' | 'REVERT'.
          2. If KEEP:
               - Overwrite configs/current.json with candidate config.
               - Update diary entry decision field from PENDING → KEPT.
          3. If REVERT:
               - Call autoresearch.revert_patch(script_path, original_content).
               - Update diary entry decision field from PENDING → REVERTED.
          4. Log outcome with AgentName.AUTORESEARCH and full metrics metadata.

        See: src/autoresearch/autoresearch.py — decide_keep_or_revert, revert_patch.
             src/autoresearch/diff_utils.py — parse_diff_to_patch (for replay).
        """
        raise NotImplementedError(
            "decide not yet implemented. Requires EVALUATE phase and Cost Manager (F4)."
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _load_history(self) -> list[IterationRecord]:
        """Load the last _HISTORY_WINDOW entries from the diary. Skips corrupted lines."""
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
