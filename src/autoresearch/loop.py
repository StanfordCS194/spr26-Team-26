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

RUN / EVALUATE / DECIDE are intentionally not implemented here. This class is
kept as the CLI-runnable proposal-only loop for fast local inspection and
backwards-compatible tests. The production/full run path lives in the
LangGraph graph in autoresearch.py.

Relationship to the LangGraph graph (autoresearch.py):
  This loop is the CLI-runnable scaffold used for testing the PROPOSE
  infra in isolation. The full LangGraph graph (build_autoresearch_graph)
  calls propose_node → run_node → evaluate_node → decide_node → log_node in a
  cycle. That graph can use local or Claude-backed proposal modes, and its RUN
  node calls the SDK-native Tinker SFT runner. This scaffold uses
  ProposalStrategy directly and never spends API tokens.
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
        Compatibility stub for older proposal-only callers.

        The full AutoResearch graph already runs bounded SDK-native Tinker SFT
        experiments through run_tinker_sft_experiment(), records costs, and
        handles cancellation. This class stays proposal-only so the legacy CLI
        can inspect candidate patches without launching training.

        See: src/autoresearch/autoresearch.py and src/tinker_api/sft_runner.py.
        """
        raise NotImplementedError(
            "AutoResearchLoop is proposal-only; use build_autoresearch_graph() "
            "for SDK-native Tinker runs."
        )

    # ──────────────────────────────────────────────────────────────────────────
    # EVALUATE (stub)
    # ──────────────────────────────────────────────────────────────────────────

    def evaluate_experiment(self, experiment_result: Any) -> None:
        """
        Compatibility stub for older proposal-only callers.

        The full graph evaluates Tinker runner metrics into EvalScore values and
        compares against the current best. This CLI scaffold deliberately stops
        after writing a PENDING proposal entry.

        See: src/autoresearch/autoresearch.py — run_evals, compare_scores, flag_regression.
        """
        raise NotImplementedError(
            "AutoResearchLoop is proposal-only; use build_autoresearch_graph() "
            "for evaluation."
        )

    # ──────────────────────────────────────────────────────────────────────────
    # DECIDE (stub)
    # ──────────────────────────────────────────────────────────────────────────

    def decide(self, score_delta: Any, diary_entry: IterationRecord) -> None:
        """
        Compatibility stub for older proposal-only callers.

        The full graph owns KEEP/REVERT decisions and writes terminal diary
        status. This class leaves entries PENDING by design for proposal review.

        See: src/autoresearch/autoresearch.py — decide_keep_or_revert, revert_patch.
             src/autoresearch/diff_utils.py — parse_diff_to_patch (for replay).
        """
        raise NotImplementedError(
            "AutoResearchLoop is proposal-only; use build_autoresearch_graph() "
            "for KEEP/REVERT decisions."
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
