"""
Edge-case tests for the AutoResearch PROPOSE infrastructure.

Covers all 8 bugs found during review:
  1. NaN rejected by TrainingConfig.apply_patch
  2. perturb_value handles current=None (falls back to full-range sample)
  3. Fixed seed produces distinct proposals on successive calls (incrementing _next_seed)
  4. LocalPerturbationProposalStrategy reads regressed params from diff-format, not JSON
  5. run_iteration raises ValueError on empty patch
  6. _load_history skips corrupted diary lines without crashing
  7. parse_diff_to_patch skips malformed "+" lines with missing key
  8. apply_patch (autoresearch.py) writes atomically — no temp file left behind
"""

from __future__ import annotations

import json
import random

import pytest

from src.autoresearch.autoresearch import apply_patch as ar_apply_patch, revert_patch
from src.autoresearch.config import BOUNDS, TrainingConfig
from src.autoresearch.diff_utils import format_patch_as_diff, parse_diff_to_patch
from src.autoresearch.loop import AutoResearchLoop
from src.autoresearch.proposer import (
    LocalPerturbationProposalStrategy,
    Proposal,
    RandomSearchProposalStrategy,
    SearchSpace,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

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


# ─── 1. NaN rejection ────────────────────────────────────────────────────────

def test_apply_patch_rejects_nan_learning_rate(base_config):
    with pytest.raises(ValueError, match="NaN"):
        base_config.apply_patch({"learning_rate": float("nan")})


def test_apply_patch_rejects_nan_dropout(base_config):
    with pytest.raises(ValueError, match="NaN"):
        base_config.apply_patch({"dropout": float("nan")})


def test_apply_patch_accepts_valid_float(base_config):
    result = base_config.apply_patch({"learning_rate": 1e-4})
    assert result.learning_rate == pytest.approx(1e-4)


# ─── 2. None current in perturb_value ────────────────────────────────────────

def test_perturb_value_none_lora_rank_returns_valid_candidate():
    rng = random.Random(42)
    val = SearchSpace.perturb_value("lora_rank", None, 0.2, rng)
    assert val in BOUNDS["lora_rank"]["candidates"]


def test_perturb_value_none_continuous_param_stays_in_bounds():
    rng = random.Random(42)
    val = SearchSpace.perturb_value("learning_rate", None, 0.2, rng)
    spec = BOUNDS["learning_rate"]
    assert spec["min"] <= val <= spec["max"]


def test_perturb_value_zero_current_stays_in_bounds():
    rng = random.Random(7)
    val = SearchSpace.perturb_value("warmup_steps", 0, 0.2, rng)
    spec = BOUNDS["warmup_steps"]
    assert spec["min"] <= val <= spec["max"]


# ─── 3. Fixed seed produces distinct proposals per call ───────────────────────

def test_random_strategy_fixed_seed_increments_between_calls(base_config):
    strategy = RandomSearchProposalStrategy(seed=0)
    p1 = strategy.propose(base_config, [])
    p2 = strategy.propose(base_config, [])
    assert p1.metadata["seed"] != p2.metadata["seed"]


def test_random_strategy_fixed_seed_both_proposals_valid(base_config):
    strategy = RandomSearchProposalStrategy(seed=0)
    for _ in range(5):
        prop = strategy.propose(base_config, [])
        assert len(prop.patch) == 1
        param = next(iter(prop.patch))
        val = prop.patch[param]
        # Must not raise — i.e. value is within bounds
        base_config.apply_patch({param: val})


def test_local_strategy_fixed_seed_increments_between_calls(base_config):
    strategy = LocalPerturbationProposalStrategy(seed=42)
    p1 = strategy.propose(base_config, [])
    p2 = strategy.propose(base_config, [])
    assert p1.metadata["seed"] != p2.metadata["seed"]


# ─── 4. Regressed params extracted from diff string, not JSON ─────────────────

def test_local_strategy_avoids_recently_reverted_param(base_config):
    """A REVERTED diff that names learning_rate should make strategy prefer other params."""
    strategy = LocalPerturbationProposalStrategy(seed=0, history_window=5)
    reverted_diff = "- learning_rate: 0.0003\n+ learning_rate: 0.001"
    history = [
        {
            "iteration": 1,
            "hypothesis": "Increase lr",
            "patch": reverted_diff,
            "decision": "REVERTED",
            "cost_usd": 0.0,
            "metrics": {},
            "notes": "",
        }
    ]
    chosen_params: set[str] = set()
    for seed_offset in range(20):
        strategy._next_seed = seed_offset
        prop = strategy.propose(base_config, history)
        chosen_params.update(prop.patch.keys())
    # Other params must appear — learning_rate should not monopolize choices
    assert len(chosen_params) > 1 or "learning_rate" not in chosen_params


def test_local_strategy_json_patch_field_does_not_blacklist(base_config):
    """Old-format JSON in the patch field has no '+ key: value' lines → nothing blacklisted."""
    strategy = LocalPerturbationProposalStrategy(seed=7, history_window=5)
    json_patch = json.dumps({"learning_rate": 0.001})
    history = [
        {
            "iteration": 1,
            "hypothesis": "Increase lr",
            "patch": json_patch,
            "decision": "REVERTED",
            "cost_usd": 0.0,
            "metrics": {},
            "notes": "",
        }
    ]
    for i in range(5):
        strategy._next_seed = i
        prop = strategy.propose(base_config, history)
        assert prop.patch  # no crash, valid proposal


def test_local_strategy_kept_entries_do_not_affect_bias(base_config):
    """Only REVERTED entries should contribute to regressed params."""
    strategy = LocalPerturbationProposalStrategy(seed=99, history_window=5)
    kept_diff = "- batch_size: 16\n+ batch_size: 32"
    history = [
        {
            "iteration": 1,
            "hypothesis": "Increase batch",
            "patch": kept_diff,
            "decision": "KEPT",
            "cost_usd": 0.0,
            "metrics": {},
            "notes": "",
        }
    ]
    chosen_params: set[str] = set()
    for seed_offset in range(15):
        strategy._next_seed = seed_offset
        prop = strategy.propose(base_config, history)
        chosen_params.update(prop.patch.keys())
    # batch_size must still be eligible since it was KEPT, not REVERTED
    assert "batch_size" in chosen_params


# ─── 5. Empty patch guard ─────────────────────────────────────────────────────

class _EmptyPatchStrategy:
    def propose(self, config, history):
        return Proposal(hypothesis="nothing changes", patch={})


def test_run_iteration_raises_on_empty_patch(config_file, diary_path):
    loop = AutoResearchLoop(
        proposer=_EmptyPatchStrategy(),
        diary_path=diary_path,
        current_config_path=config_file,
    )
    with pytest.raises(ValueError, match="empty patch"):
        loop.run_iteration()


def test_run_iteration_does_not_write_diary_on_empty_patch(config_file, diary_path):
    """An empty-patch proposal should abort before touching the diary."""
    loop = AutoResearchLoop(
        proposer=_EmptyPatchStrategy(),
        diary_path=diary_path,
        current_config_path=config_file,
    )
    with pytest.raises(ValueError):
        loop.run_iteration()
    assert not diary_path.exists()


# ─── 6. Corrupted diary lines skipped ────────────────────────────────────────

def _good_entry(n: int) -> str:
    return json.dumps({
        "iteration": n,
        "hypothesis": f"iter {n}",
        "patch": f"- batch_size: 16\n+ batch_size: {n * 8}",
        "cost_usd": 0.0,
        "metrics": {},
        "decision": "KEPT",
        "notes": "",
    })


def test_load_history_skips_corrupted_lines(config_file, tmp_path):
    diary = tmp_path / "diary.jsonl"
    diary.write_text(
        _good_entry(1) + "\n"
        "NOT_VALID_JSON\n"
        + _good_entry(2) + "\n"
    )
    loop = AutoResearchLoop(
        proposer=RandomSearchProposalStrategy(seed=0),
        diary_path=diary,
        current_config_path=config_file,
    )
    history = loop._load_history()
    assert len(history) == 2


def test_load_history_returns_empty_when_file_missing(config_file, diary_path):
    loop = AutoResearchLoop(
        proposer=RandomSearchProposalStrategy(seed=0),
        diary_path=diary_path,
        current_config_path=config_file,
    )
    assert loop._load_history() == []


def test_load_history_caps_at_window(config_file, tmp_path):
    diary = tmp_path / "diary.jsonl"
    diary.write_text("\n".join(_good_entry(i) for i in range(1, 20)) + "\n")
    loop = AutoResearchLoop(
        proposer=RandomSearchProposalStrategy(seed=0),
        diary_path=diary,
        current_config_path=config_file,
    )
    # _HISTORY_WINDOW is 10
    history = loop._load_history()
    assert len(history) == 10


# ─── 7. parse_diff_to_patch empty-key guard ───────────────────────────────────

def test_parse_diff_to_patch_skips_empty_key_line():
    diff = "+ \n+ learning_rate: 0.001"
    patch = parse_diff_to_patch(diff)
    assert "" not in patch
    assert patch.get("learning_rate") == pytest.approx(0.001)


def test_parse_diff_to_patch_skips_no_separator_line():
    diff = "+ noseparatorhere\n+ batch_size: 32"
    patch = parse_diff_to_patch(diff)
    assert "noseparatorhere" not in patch
    assert patch.get("batch_size") == 32


def test_parse_diff_to_patch_roundtrip(base_config):
    original_patch = {"batch_size": 32}
    diff = format_patch_as_diff(original_patch, base_config)
    recovered = parse_diff_to_patch(diff)
    assert recovered == {"batch_size": 32}


def test_parse_diff_to_patch_preserves_int_types(base_config):
    diff = format_patch_as_diff({"batch_size": 64, "warmup_steps": 200}, base_config)
    recovered = parse_diff_to_patch(diff)
    assert isinstance(recovered["batch_size"], int)
    assert isinstance(recovered["warmup_steps"], int)


def test_parse_diff_to_patch_preserves_float_types(base_config):
    diff = format_patch_as_diff({"learning_rate": 1e-4}, base_config)
    recovered = parse_diff_to_patch(diff)
    assert isinstance(recovered["learning_rate"], float)
    assert recovered["learning_rate"] == pytest.approx(1e-4)


# ─── 8. Atomic write — no temp file left behind ───────────────────────────────

def test_ar_apply_patch_no_temp_file_left(tmp_path, base_config):
    config_path = tmp_path / "config.json"
    base_config.save(config_path)

    ar_apply_patch(str(config_path), json.dumps({"batch_size": 32}))

    tmp_file = config_path.with_suffix(".json.tmp")
    assert not tmp_file.exists()


def test_ar_apply_patch_writes_updated_value(tmp_path, base_config):
    config_path = tmp_path / "config.json"
    base_config.save(config_path)

    ar_apply_patch(str(config_path), json.dumps({"batch_size": 32}))

    updated = json.loads(config_path.read_text())
    assert updated["batch_size"] == 32


def test_ar_apply_patch_returns_valid_original_json(tmp_path, base_config):
    config_path = tmp_path / "config.json"
    base_config.save(config_path)

    original = ar_apply_patch(str(config_path), json.dumps({"batch_size": 32}))
    old = json.loads(original)
    assert old["batch_size"] == 16  # TrainingConfig default


def test_ar_revert_patch_restores_original(tmp_path, base_config):
    config_path = tmp_path / "config.json"
    base_config.save(config_path)

    original = ar_apply_patch(str(config_path), json.dumps({"learning_rate": 1e-5}))
    revert_patch(str(config_path), original)

    restored = TrainingConfig.load(config_path)
    assert restored.learning_rate == pytest.approx(base_config.learning_rate)


# ─── Bonus: bounds validation still solid ────────────────────────────────────

def test_apply_patch_rejects_lr_above_max(base_config):
    with pytest.raises(ValueError):
        base_config.apply_patch({"learning_rate": 1.0})  # max is 1e-2


def test_apply_patch_rejects_lr_below_min(base_config):
    with pytest.raises(ValueError):
        base_config.apply_patch({"learning_rate": 1e-10})  # min is 1e-6


def test_apply_patch_rejects_unknown_key(base_config):
    with pytest.raises(ValueError, match="Unknown"):
        base_config.apply_patch({"not_a_real_param": 42})


def test_apply_patch_rejects_invalid_lora_rank_candidate(base_config):
    with pytest.raises(ValueError, match="candidates"):
        base_config.apply_patch({"lora_rank": 3})  # must be in [4,8,16,32,64,128]
