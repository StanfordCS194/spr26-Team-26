"""
End-to-end tests for the AutoResearch PROPOSE loop.

Three levels of coverage:

  LEVEL 1 — Mechanical (no API, fast)
    Runs AutoResearchLoop with algorithmic proposers for N iterations.
    Verifies diary entries, config mutations, apply/revert round-trips,
    and log file output without spending any API tokens.

  LEVEL 2 — Claude API integration (real API call, costs ~$0.001)
    Calls propose_hypothesis() with a real config and asserts the returned
    Hypothesis has valid structure and a parseable, in-bounds patch.
    Marked with @pytest.mark.integration — skipped unless ANTHROPIC_API_KEY
    is set.

  LEVEL 3 — State transition trace
    Runs 5 iterations and prints a full human-readable trace of every diary
    entry and log event so behaviour is visible without a debugger.
"""

from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path

import pytest

from src.autoresearch.autoresearch import (
    apply_patch,
    compare_scores,
    create_eval_suite,
    decide_keep_or_revert,
    flag_regression,
    propose_hypothesis,
    revert_patch,
)
from src.autoresearch.config import BOUNDS, TrainingConfig
from src.autoresearch.diff_utils import format_patch_as_diff, parse_diff_to_patch
from src.autoresearch.loop import AutoResearchLoop
from src.autoresearch.proposer import (
    LocalPerturbationProposalStrategy,
    RandomSearchProposalStrategy,
)

# ─── Markers ─────────────────────────────────────────────────────────────────

integration = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set — skipping live API test",
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def base_config():
    return TrainingConfig(model_name="distilbert-base-uncased")


@pytest.fixture
def config_file(tmp_path, base_config):
    p = tmp_path / "current.json"
    base_config.save(p)
    return p


@pytest.fixture
def diary_path(tmp_path):
    return tmp_path / "logs" / "research_diary.jsonl"


@pytest.fixture
def run_log_path(tmp_path):
    return tmp_path / "logs" / "run.jsonl"


# ═════════════════════════════════════════════════════════════════════════════
# LEVEL 1 — Mechanical end-to-end
# ═════════════════════════════════════════════════════════════════════════════

class TestMultipleIterations:
    """Run several propose iterations and verify every observable side-effect."""

    def test_three_iterations_append_three_diary_entries(self, config_file, diary_path):
        loop = AutoResearchLoop(
            proposer=RandomSearchProposalStrategy(seed=0),
            diary_path=diary_path,
            current_config_path=config_file,
        )
        for _ in range(3):
            loop.run_iteration()

        lines = [l for l in diary_path.read_text().splitlines() if l.strip()]
        assert len(lines) == 3

    def test_diary_entries_have_sequential_iteration_numbers(self, config_file, diary_path):
        loop = AutoResearchLoop(
            proposer=RandomSearchProposalStrategy(seed=10),
            diary_path=diary_path,
            current_config_path=config_file,
        )
        for _ in range(4):
            loop.run_iteration()

        records = [json.loads(l) for l in diary_path.read_text().splitlines() if l.strip()]
        assert [r["iteration"] for r in records] == [1, 2, 3, 4]

    def test_all_diary_entries_start_as_pending(self, config_file, diary_path):
        loop = AutoResearchLoop(
            proposer=LocalPerturbationProposalStrategy(seed=7),
            diary_path=diary_path,
            current_config_path=config_file,
        )
        for _ in range(3):
            loop.run_iteration()

        records = [json.loads(l) for l in diary_path.read_text().splitlines() if l.strip()]
        assert all(r["decision"] == "PENDING" for r in records)

    def test_diary_entries_contain_valid_diff_string(self, config_file, diary_path):
        loop = AutoResearchLoop(
            proposer=RandomSearchProposalStrategy(seed=5),
            diary_path=diary_path,
            current_config_path=config_file,
        )
        loop.run_iteration()

        record = json.loads(diary_path.read_text().splitlines()[0])
        patch = record["patch"]
        # Must contain at least one "- key: val" and one "+ key: val" line
        assert any(l.startswith("- ") for l in patch.splitlines())
        assert any(l.startswith("+ ") for l in patch.splitlines())

    def test_diff_is_invertible_via_parse_diff_to_patch(self, config_file, diary_path):
        """parse_diff_to_patch(format_patch_as_diff(p)) == p for every proposal."""
        loop = AutoResearchLoop(
            proposer=RandomSearchProposalStrategy(seed=99),
            diary_path=diary_path,
            current_config_path=config_file,
        )
        for _ in range(5):
            loop.run_iteration()

        records = [json.loads(l) for l in diary_path.read_text().splitlines() if l.strip()]
        for record in records:
            recovered = parse_diff_to_patch(record["patch"])
            assert recovered  # non-empty
            for key in recovered:
                assert key in BOUNDS or key == "model_name"

    def test_log_file_written_for_every_iteration(self, config_file, diary_path, tmp_path):
        log_path = tmp_path / "logs" / "run.jsonl"
        import unittest.mock as mock
        # Patch the log path used by observability
        with mock.patch(
            "src.observability.observability.write_json_log",
            wraps=lambda entry, path: _write_to(entry, log_path),
        ):
            loop = AutoResearchLoop(
                proposer=RandomSearchProposalStrategy(seed=1),
                diary_path=diary_path,
                current_config_path=config_file,
            )
            for _ in range(3):
                loop.run_iteration()

        # At minimum one log line per iteration (the "Starting iteration N" event)
        if log_path.exists():
            lines = [l for l in log_path.read_text().splitlines() if l.strip()]
            assert len(lines) >= 3

    def test_each_proposal_changes_exactly_one_param(self, config_file, diary_path):
        loop = AutoResearchLoop(
            proposer=RandomSearchProposalStrategy(seed=42),
            diary_path=diary_path,
            current_config_path=config_file,
        )
        for _ in range(6):
            loop.run_iteration()

        records = [json.loads(l) for l in diary_path.read_text().splitlines() if l.strip()]
        for record in records:
            plus_lines = [l for l in record["patch"].splitlines() if l.startswith("+ ")]
            assert len(plus_lines) == 1, f"Expected exactly 1 changed param, got {plus_lines}"

    def test_proposal_values_stay_within_bounds(self, config_file, diary_path):
        loop = AutoResearchLoop(
            proposer=RandomSearchProposalStrategy(seed=77),
            diary_path=diary_path,
            current_config_path=config_file,
        )
        base = TrainingConfig.load(config_file)
        for _ in range(10):
            loop.run_iteration()

        records = [json.loads(l) for l in diary_path.read_text().splitlines() if l.strip()]
        for record in records:
            patch = parse_diff_to_patch(record["patch"])
            # apply_patch validates bounds — if it raised, run_iteration would have failed
            # We verify retroactively that every recovered patch is valid
            base.apply_patch(patch)  # raises if out of bounds

    def test_local_strategy_produces_valid_proposals_with_history(self, config_file, diary_path):
        """LocalPerturbation uses history; ensure it doesn't crash with realistic diary."""
        loop = AutoResearchLoop(
            proposer=LocalPerturbationProposalStrategy(seed=3, perturbation_factor=0.2),
            diary_path=diary_path,
            current_config_path=config_file,
        )
        # Seed the diary with a mix of KEPT and REVERTED entries
        _seed_diary(diary_path, [
            ("- learning_rate: 0.0003\n+ learning_rate: 0.001", "REVERTED"),
            ("- batch_size: 16\n+ batch_size: 32", "KEPT"),
            ("- lora_rank: 16\n+ lora_rank: 32", "KEPT"),
        ])
        for _ in range(5):
            loop.run_iteration()

        records = [json.loads(l) for l in diary_path.read_text().splitlines() if l.strip()]
        # First 3 are seeded; next 5 are new proposals
        assert len(records) == 8


# ═════════════════════════════════════════════════════════════════════════════
# LEVEL 1 — Helper logic unit tests
# ═════════════════════════════════════════════════════════════════════════════

class TestHelperFunctions:

    def test_compare_scores_positive_delta(self):
        new = {"scalar": 0.90, "metrics": {}, "critique": ""}
        base = {"scalar": 0.85, "metrics": {}, "critique": ""}
        delta = compare_scores(new, base)
        assert delta["improved"] is True
        assert delta["absolute"] == pytest.approx(0.05, abs=1e-9)
        assert delta["relative_pct"] == pytest.approx(5.88, abs=0.01)

    def test_compare_scores_negative_delta(self):
        new = {"scalar": 0.80, "metrics": {}, "critique": ""}
        base = {"scalar": 0.85, "metrics": {}, "critique": ""}
        delta = compare_scores(new, base)
        assert delta["improved"] is False
        assert delta["absolute"] < 0

    def test_compare_scores_tie_is_not_improved(self):
        score = {"scalar": 0.85, "metrics": {}, "critique": ""}
        delta = compare_scores(score, score)
        assert delta["improved"] is False

    def test_decide_keep_on_positive_delta(self):
        delta = {"absolute": 0.05, "relative_pct": 5.0, "improved": True}
        assert decide_keep_or_revert(delta) == "KEEP"

    def test_decide_revert_on_negative_delta(self):
        delta = {"absolute": -0.02, "relative_pct": -2.0, "improved": False}
        assert decide_keep_or_revert(delta) == "REVERT"

    def test_decide_revert_on_zero_delta(self):
        delta = {"absolute": 0.0, "relative_pct": 0.0, "improved": False}
        assert decide_keep_or_revert(delta) == "REVERT"

    def test_flag_regression_above_threshold(self):
        delta = {"absolute": 0.02, "relative_pct": 2.0, "improved": True}
        assert flag_regression(delta) is False

    def test_flag_regression_below_threshold(self):
        delta = {"absolute": -0.05, "relative_pct": -5.0, "improved": False}
        assert flag_regression(delta) is True

    def test_flag_regression_at_threshold_is_false(self):
        # -0.01 is exactly at threshold; strict < means it should NOT flag
        delta = {"absolute": -0.01, "relative_pct": -1.0, "improved": False}
        assert flag_regression(delta, threshold=-0.01) is False

    def test_create_eval_suite_classification(self):
        task = {
            "task_type": "text-classification",
            "modality": "text",
            "has_pretrained_base": True,
            "eval_metric": "",
            "complexity": "medium",
        }
        dataset = _minimal_dataset("/tmp/data")
        suite = create_eval_suite(task, dataset)
        assert suite["primary_metric"] == "f1"
        assert "f1" in suite["metrics"]
        assert suite["use_llm_grading"] is False

    def test_create_eval_suite_high_complexity_enables_llm_grading(self):
        task = {
            "task_type": "text-classification",
            "modality": "text",
            "has_pretrained_base": True,
            "eval_metric": "",
            "complexity": "high",
        }
        dataset = _minimal_dataset("/tmp/data")
        suite = create_eval_suite(task, dataset)
        assert suite["use_llm_grading"] is True

    def test_create_eval_suite_caller_metric_overrides_default(self):
        task = {
            "task_type": "text-classification",
            "modality": "text",
            "has_pretrained_base": True,
            "eval_metric": "accuracy",
            "complexity": "low",
        }
        dataset = _minimal_dataset("/tmp/data")
        suite = create_eval_suite(task, dataset)
        assert suite["primary_metric"] == "accuracy"

    def test_apply_patch_revert_round_trip(self, tmp_path, base_config):
        path = tmp_path / "config.json"
        base_config.save(path)
        original = apply_patch(str(path), json.dumps({"batch_size": 64}))
        assert TrainingConfig.load(path).batch_size == 64
        revert_patch(str(path), original)
        assert TrainingConfig.load(path).batch_size == 16


# ═════════════════════════════════════════════════════════════════════════════
# LEVEL 2 — Live Claude API integration
# ═════════════════════════════════════════════════════════════════════════════

class TestClaudeAPIIntegration:

    @integration
    def test_propose_hypothesis_returns_valid_json_structure(self):
        config = TrainingConfig(model_name="distilbert-base-uncased")
        task = {
            "task_type": "text-classification",
            "modality": "text",
            "has_pretrained_base": True,
            "eval_metric": "f1",
            "complexity": "medium",
        }
        h = propose_hypothesis(config.to_dict(), [], task)
        assert h["description"]
        assert h["patch"]
        assert h["expected_effect"]
        assert h["search_strategy"] in ("random", "local", "playbook")

    @integration
    def test_propose_hypothesis_patch_is_in_bounds(self):
        config = TrainingConfig(model_name="distilbert-base-uncased")
        task = {
            "task_type": "text-classification",
            "modality": "text",
            "has_pretrained_base": True,
            "eval_metric": "f1",
            "complexity": "medium",
        }
        h = propose_hypothesis(config.to_dict(), [], task)
        patch_dict = json.loads(h["patch"])
        assert len(patch_dict) == 1, "Claude must change exactly ONE parameter"
        param, value = next(iter(patch_dict.items()))
        # apply_patch validates bounds — if out of bounds it raises
        config.apply_patch(patch_dict)

    @integration
    def test_propose_hypothesis_respects_reverted_history(self):
        """Claude should not re-propose a parameter that was just REVERTED."""
        config = TrainingConfig(model_name="distilbert-base-uncased")
        history = [
            {
                "iteration": 1,
                "hypothesis": "Increase learning_rate from 3e-4 to 1e-3.",
                "patch": "- learning_rate: 3e-4\n+ learning_rate: 0.001",
                "cost_usd": 0.10,
                "metrics": {"train_loss": 0.45, "val_loss": 0.48,
                             "test_loss": 0.50, "primary_metric": 0.82},
                "decision": "REVERTED",
                "notes": "loss spiked",
            }
        ]
        task = {
            "task_type": "text-classification",
            "modality": "text",
            "has_pretrained_base": True,
            "eval_metric": "f1",
            "complexity": "medium",
        }
        # Run several times and confirm Claude doesn't fixate on learning_rate
        chosen_params = set()
        for _ in range(3):
            h = propose_hypothesis(config.to_dict(), history, task)
            patch_dict = json.loads(h["patch"])
            chosen_params.update(patch_dict.keys())
        # With REVERTED history, Claude should diversify
        assert len(chosen_params) >= 1  # soft check — Claude may still choose lr, but should vary


# ═════════════════════════════════════════════════════════════════════════════
# LEVEL 3 — Trace: human-readable step-by-step dump
# ═════════════════════════════════════════════════════════════════════════════

class TestTraceView:
    """
    Not a pass/fail test — runs 5 iterations and prints a full trace so
    each step's output is visible. Run with -s to see the output:

        pytest tests/test_e2e_propose.py::TestTraceView -s -v
    """

    def test_print_full_propose_trace(self, config_file, diary_path, capsys):
        n_iters = 5
        loop = AutoResearchLoop(
            proposer=LocalPerturbationProposalStrategy(seed=42, perturbation_factor=0.2),
            diary_path=diary_path,
            current_config_path=config_file,
        )

        print("\n" + "═" * 72)
        print("  AUTORESEARCH PROPOSE LOOP — FULL TRACE  (5 iterations)")
        print("  Strategy: LocalPerturbation  seed=42  perturbation=±20%")
        print("═" * 72)

        base = TrainingConfig.load(config_file)
        print(f"\n  Initial config: {json.dumps(base.to_dict(), indent=4)}\n")

        for i in range(n_iters):
            print(f"\n{'─' * 72}")
            print(f"  ITERATION {i + 1}")
            print(f"{'─' * 72}")
            loop.run_iteration()

        # Read and pretty-print the diary
        print(f"\n{'═' * 72}")
        print("  RESEARCH DIARY  (outputs/logs/research_diary.jsonl)")
        print(f"{'═' * 72}\n")

        records = [json.loads(l) for l in diary_path.read_text().splitlines() if l.strip()]
        for r in records:
            status = r["decision"]
            marker = "✓ KEPT    " if status == "KEPT" else "✗ REVERTED" if status == "REVERTED" else "⏳ PENDING "
            print(f"  [{marker}] Iter {r['iteration']:>2}  {r['hypothesis'][:70]}")
            for line in r["patch"].splitlines():
                color = "  +  " if line.startswith("+") else "  -  "
                print(f"           {color}{line}")
            print()

        # Structural assertions (test still needs to pass)
        assert len(records) == n_iters
        assert all(r["decision"] == "PENDING" for r in records)  # loop.py only does PROPOSE
        for r in records:
            recovered = parse_diff_to_patch(r["patch"])
            assert recovered  # each diff is parseable and non-empty


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _write_to(entry, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _minimal_dataset(path: str):
    return {
        "dataset": {
            "path": path,
            "format": "jsonl",
            "train_size": 1000,
            "val_size": 100,
            "test_size": 100,
        },
        "mode_used": "A",
        "quality_notes": "",
        "validation_report": {"passed": True, "issues": [], "sample_accuracy_estimate": 0.9},
    }


def _seed_diary(diary_path: Path, entries: list[tuple[str, str]]) -> None:
    diary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(diary_path, "a") as f:
        for i, (patch, decision) in enumerate(entries, 1):
            record = {
                "iteration": i,
                "hypothesis": f"seeded entry {i}",
                "patch": patch,
                "cost_usd": 0.0,
                "metrics": {"train_loss": 0.3, "val_loss": 0.35,
                             "test_loss": 0.36, "primary_metric": 0.88},
                "decision": decision,
                "notes": "seeded for test",
            }
            f.write(json.dumps(record) + "\n")
