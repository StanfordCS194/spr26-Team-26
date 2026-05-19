from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.autoresearch.config import TrainingConfig


def test_record_to_conversation_accepts_messages():
    from src.tinker_api.sft_runner import record_to_conversation

    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]
    assert record_to_conversation({"messages": messages}) == messages


def test_record_to_conversation_converts_input_output():
    from src.tinker_api.sft_runner import record_to_conversation

    assert record_to_conversation({"input": "Classify this", "output": "positive"}) == [
        {"role": "user", "content": "Classify this"},
        {"role": "assistant", "content": "positive"},
    ]


def test_record_to_conversation_converts_input_label():
    from src.tinker_api.sft_runner import record_to_conversation

    assert record_to_conversation({"input": "Classify this", "label": 1})[-1] == {
        "role": "assistant",
        "content": "1",
    }


def test_record_to_conversation_rejects_malformed_record():
    from src.tinker_api.sft_runner import record_to_conversation

    with pytest.raises(ValueError, match="missing target"):
        record_to_conversation({"input": "No answer"})


def test_load_conversations_returns_empty_for_empty_jsonl(tmp_path):
    from src.tinker_api.sft_runner import load_conversations

    path = tmp_path / "train.jsonl"
    path.write_text("")
    assert load_conversations(path) == []


def test_resolve_renderer_fallback_for_qwen35_9b():
    from src.tinker_api.sft_runner import DEFAULT_TINKER_MODEL, resolve_renderer_name

    model_info = MagicMock()
    model_info.get_recommended_renderer_name.side_effect = KeyError(DEFAULT_TINKER_MODEL)

    assert resolve_renderer_name(DEFAULT_TINKER_MODEL, model_info) == "qwen3_5_disable_thinking"


def test_tinker_search_space_can_be_restricted_to_supported_tunables():
    from src.autoresearch.proposer import SearchSpace
    from src.tinker_api.sft_runner import SUPPORTED_TINKER_TUNABLES

    assert set(SearchSpace.tunable_params("tinker_sft")) == set(SUPPORTED_TINKER_TUNABLES)
    assert "dropout" not in SearchSpace.tunable_params("tinker_sft")
    assert "warmup_steps" not in SearchSpace.tunable_params("tinker_sft")


def test_run_tinker_sft_experiment_writes_artifacts(tmp_path, monkeypatch):
    state = _install_fake_tinker_stack(monkeypatch, losses=[0.5, 0.25])
    from src.tinker_api.sft_runner import run_tinker_sft_experiment

    data_path = _write_jsonl(
        tmp_path / "train.jsonl",
        [
            {"input": "Say hello", "output": "hello"},
            {
                "messages": [
                    {"role": "user", "content": "2+2"},
                    {"role": "assistant", "content": "4"},
                ]
            },
        ],
    )

    result = run_tinker_sft_experiment(
        TrainingConfig(model_name="Qwen/Qwen3.5-9B", batch_size=1),
        str(data_path),
        run_id="unit-run",
        max_steps=2,
        output_dir=str(tmp_path / "experiments"),
    )

    run_dir = tmp_path / "experiments" / "unit-run"
    assert result["status"] == "COMPLETED"
    assert result["model_path"] == str(run_dir)
    assert json.loads((run_dir / "metrics.json").read_text())["val_loss"] == pytest.approx(0.25)
    assert (run_dir / "metrics.jsonl").read_text().count("\n") == 2
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["checkpoints"]["state_path"].startswith("tinker://state/")
    assert manifest["checkpoints"]["sampler_path"].startswith("tinker://sampler/")
    assert manifest["train_on_what"] == "last_assistant_message"
    assert manifest["training_examples"] == 2
    assert json.loads((run_dir / "sample.json").read_text())["text"] == "sample text"
    assert state.training_client.forward_backward_calls == 2
    assert state.training_client.optim_step_calls == 2
    assert {call["train_on_what"] for call in state.conversions} == {
        "last_assistant_message"
    }


def test_run_tinker_sft_experiment_splits_multi_assistant_conversations(
    tmp_path,
    monkeypatch,
):
    state = _install_fake_tinker_stack(monkeypatch, losses=[0.5])
    from src.tinker_api.sft_runner import run_tinker_sft_experiment

    data_path = _write_jsonl(
        tmp_path / "train.jsonl",
        [
            {
                "messages": [
                    {"role": "system", "content": "Be brief."},
                    {"role": "user", "content": "First"},
                    {"role": "assistant", "content": "One"},
                    {"role": "user", "content": "Second"},
                    {"role": "assistant", "content": "Two"},
                ]
            },
        ],
    )

    result = run_tinker_sft_experiment(
        TrainingConfig(model_name="Qwen/Qwen3.5-9B", batch_size=4),
        str(data_path),
        run_id="multi-assistant-run",
        max_steps=1,
        output_dir=str(tmp_path / "experiments"),
    )

    run_dir = tmp_path / "experiments" / "multi-assistant-run"
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert result["status"] == "COMPLETED"
    assert manifest["training_examples"] == 2
    assert len(state.conversions) == 4
    converted_targets = {
        call["conversation"][-1]["content"]
        for call in state.conversions
    }
    assert converted_targets == {"One", "Two"}
    assert {call["train_on_what"] for call in state.conversions} == {
        "last_assistant_message"
    }


def test_run_tinker_sft_experiment_reads_tinker_loss_sum_metric(
    tmp_path,
    monkeypatch,
):
    _install_fake_tinker_stack(
        monkeypatch,
        losses=[{"loss:sum": 3.5}, {"loss:sum": 2.25}],
    )
    from src.tinker_api.sft_runner import run_tinker_sft_experiment

    data_path = _write_jsonl(
        tmp_path / "train.jsonl",
        [{"input": "Question", "output": "Answer"}],
    )

    result = run_tinker_sft_experiment(
        TrainingConfig(model_name="Qwen/Qwen3.5-9B", batch_size=1),
        str(data_path),
        run_id="loss-sum-run",
        max_steps=2,
        output_dir=str(tmp_path / "experiments"),
    )

    run_dir = tmp_path / "experiments" / "loss-sum-run"
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert result["metrics"]["train_loss"] == pytest.approx(2.25)
    assert metrics["primary_metric"] == pytest.approx(1.0 / 3.25)


def test_run_tinker_sft_experiment_stops_on_cancel_and_writes_artifacts(tmp_path, monkeypatch):
    _install_fake_tinker_stack(monkeypatch, losses=[0.5, 0.25, 0.1])
    from src.tinker_api import tinker_api
    from src.tinker_api.sft_runner import run_tinker_sft_experiment

    original_record_tokens = tinker_api.record_tokens

    def cancel_after_first_step(job_id: str, n_tokens: int) -> None:
        original_record_tokens(job_id, n_tokens)
        tinker_api.cancel_job(job_id)

    monkeypatch.setattr("src.tinker_api.sft_runner.record_tokens", cancel_after_first_step)
    data_path = _write_jsonl(
        tmp_path / "train.jsonl",
        [{"input": "Question", "output": "Answer"}],
    )

    result = run_tinker_sft_experiment(
        TrainingConfig(model_name="Qwen/Qwen3.5-9B", batch_size=1),
        data_path,
        run_id="cancel-run",
        max_steps=3,
        output_dir=str(tmp_path / "experiments"),
    )

    run_dir = tmp_path / "experiments" / "cancel-run"
    assert result["status"] == "CANCELLED"
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "sample.json").exists()
    assert (run_dir / "metrics.jsonl").read_text().count("\n") == 1


def test_run_evals_scores_tinker_metrics(tmp_path):
    from src.autoresearch.autoresearch import run_evals

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metrics.json").write_text(
        json.dumps({"train_loss": 0.5, "val_loss": 1.0, "test_loss": 1.2})
    )

    score = run_evals(
        str(run_dir),
        {
            "primary_metric": "primary_metric",
            "metrics": [],
            "test_split_path": "",
            "use_llm_grading": False,
        },
    )

    assert score["scalar"] == pytest.approx(0.5)
    assert score["metrics"]["primary_metric"] == pytest.approx(0.5)


def test_autoresearch_run_node_calls_tinker_runner(monkeypatch, tmp_path):
    import src.autoresearch.autoresearch as ar

    data_path = _write_jsonl(tmp_path / "train.jsonl", [{"input": "x", "output": "y"}])
    captured = {}

    def fake_runner(config, dataset, *, max_steps=None, **kwargs):
        captured["config"] = config
        captured["dataset"] = dataset
        captured["max_steps"] = max_steps
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "metrics.json").write_text(
            json.dumps({"train_loss": 0.5, "val_loss": 0.5, "test_loss": 0.5})
        )
        return {
            "job_id": "fake-run",
            "status": "COMPLETED",
            "metrics": {
                "train_loss": 0.5,
                "val_loss": 0.5,
                "test_loss": 0.5,
                "primary_metric": 2 / 3,
            },
            "model_path": str(run_dir),
            "cost_usd": 0.01,
            "logs_path": str(run_dir / "metrics.jsonl"),
        }

    monkeypatch.setattr(ar, "run_tinker_sft_experiment", fake_runner)
    state = _autoresearch_state(str(data_path))

    result = ar.run_node(state)

    assert result["last_result"]["job_id"] == "fake-run"
    assert captured["max_steps"] == 5
    assert captured["dataset"]["dataset"]["path"] == str(data_path)
    assert captured["config"].model_name == "Qwen/Qwen3.5-9B"


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def _autoresearch_state(dataset_path: str) -> dict:
    return {
        "plan": {
            "strategy": "fine-tune",
            "base_model": "Qwen/Qwen3.5-9B",
            "lora_config": {"rank": 8, "alpha": 16, "dropout": 0.05, "target_modules": []},
            "estimated_cost": 1.0,
            "estimated_time_min": 5,
            "training_script_path": "outputs/scripts/train.py",
            "eval_metric": "primary_metric",
            "backend": "tinker_sft",
            "dataset_path": dataset_path,
        },
        "config": {
            "data": False,
            "prompt": "test",
            "compute_budget": 10.0,
            "training_procedure": {
                "task_type": "text-classification",
                "data_format": "jsonl",
                "training_type": "SFT",
                "base_model": "Qwen/Qwen3.5-9B",
                "hyperparameters": {"learning_rate": 1e-4, "batch_size": 1},
                "notes": "",
            },
        },
        "eval_suite": {
            "primary_metric": "primary_metric",
            "metrics": ["primary_metric"],
            "test_split_path": "",
            "use_llm_grading": False,
        },
        "current_script": "outputs/scripts/train.py",
        "current_config": {"learning_rate": 1e-4, "batch_size": 1},
        "current_patch": None,
        "last_description": None,
        "original_content": None,
        "diary": [],
        "baseline_score": None,
        "best_score": None,
        "best_script": "outputs/scripts/train.py",
        "last_result": None,
        "last_score": None,
        "last_delta": None,
        "iteration": 0,
        "no_improve_streak": 0,
        "should_stop": False,
    }


class _FakeFuture:
    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value


class _FakeModelInput:
    def __init__(self, length: int):
        self.length = length


class _FakeDatum:
    def __init__(self, length: int):
        self.model_input = _FakeModelInput(length)


class _FakeRenderer:
    def build_generation_prompt(self, messages):
        return _FakeModelInput(len(messages) + 1)

    def get_stop_sequences(self):
        return [0]

    def parse_response(self, tokens):
        return {"role": "assistant", "content": "sample text"}, "stop"


class _FakeTrainingClient:
    def __init__(self, losses):
        self.losses = list(losses)
        self.forward_backward_calls = 0
        self.optim_step_calls = 0
        self.batches = []
        self.sampling_client = _FakeSamplingClient()

    def forward_backward(self, batch, loss_fn):
        self.forward_backward_calls += 1
        self.batches.append(batch)
        loss = self.losses[min(self.forward_backward_calls - 1, len(self.losses) - 1)]
        if isinstance(loss, dict):
            return _FakeFuture(types.SimpleNamespace(metrics=loss))
        return _FakeFuture(types.SimpleNamespace(loss=loss))

    def optim_step(self, adam_params):
        self.optim_step_calls += 1
        return _FakeFuture(types.SimpleNamespace(metrics={}))

    def save_state(self, name):
        return _FakeFuture(types.SimpleNamespace(path=f"tinker://state/{name}"))

    def save_weights_for_sampler(self, name):
        return _FakeFuture(types.SimpleNamespace(path=f"tinker://sampler/{name}"))

    def save_weights_and_get_sampling_client(self, name=None):
        return self.sampling_client


class _FakeSamplingClient:
    def sample(self, prompt, num_samples, sampling_params):
        sequence = types.SimpleNamespace(tokens=[10, 11, 12])
        return _FakeFuture(types.SimpleNamespace(sequences=[sequence]))


class _FakeServiceClient:
    def __init__(self, training_client):
        self.training_client = training_client

    def create_lora_training_client(self, base_model, rank):
        self.base_model = base_model
        self.rank = rank
        return self.training_client


def _install_fake_tinker_stack(monkeypatch, losses):
    training_client = _FakeTrainingClient(losses)
    service_client = _FakeServiceClient(training_client)

    tinker_mod = types.ModuleType("tinker")
    tinker_types_mod = types.ModuleType("tinker.types")

    class AdamParams:
        def __init__(self, learning_rate):
            self.learning_rate = learning_rate

    class SamplingParams:
        def __init__(self, max_tokens, stop):
            self.max_tokens = max_tokens
            self.stop = stop

    tinker_types_mod.AdamParams = AdamParams
    tinker_types_mod.SamplingParams = SamplingParams
    tinker_mod.types = tinker_types_mod
    tinker_mod.ServiceClient = lambda: service_client

    cookbook = types.ModuleType("tinker_cookbook")
    model_info = types.ModuleType("tinker_cookbook.model_info")
    model_info.get_recommended_renderer_name = MagicMock(side_effect=KeyError("missing"))
    renderers = types.ModuleType("tinker_cookbook.renderers")
    renderers.TrainOnWhat = types.SimpleNamespace(
        ALL_ASSISTANT_MESSAGES="all_assistant_messages",
        LAST_ASSISTANT_MESSAGE="last_assistant_message",
    )
    renderers.get_renderer = MagicMock(return_value=_FakeRenderer())
    tokenizer_utils = types.ModuleType("tinker_cookbook.tokenizer_utils")
    tokenizer_utils.get_tokenizer = MagicMock(return_value=object())
    supervised_pkg = types.ModuleType("tinker_cookbook.supervised")
    supervised_data = types.ModuleType("tinker_cookbook.supervised.data")
    conversions = []

    def conversation_to_datum(conversation, renderer, max_length, train_on_what):
        conversions.append(
            {
                "conversation": conversation,
                "max_length": max_length,
                "train_on_what": train_on_what,
            }
        )
        return _FakeDatum(length=len(json.dumps(conversation)))

    supervised_data.conversation_to_datum = conversation_to_datum
    supervised_pkg.data = supervised_data
    cookbook.model_info = model_info
    cookbook.renderers = renderers
    cookbook.tokenizer_utils = tokenizer_utils

    for name in [
        "tinker",
        "tinker.types",
        "tinker_cookbook",
        "tinker_cookbook.model_info",
        "tinker_cookbook.renderers",
        "tinker_cookbook.tokenizer_utils",
        "tinker_cookbook.supervised",
        "tinker_cookbook.supervised.data",
    ]:
        sys.modules.pop(name, None)

    monkeypatch.setitem(sys.modules, "tinker", tinker_mod)
    monkeypatch.setitem(sys.modules, "tinker.types", tinker_types_mod)
    monkeypatch.setitem(sys.modules, "tinker_cookbook", cookbook)
    monkeypatch.setitem(sys.modules, "tinker_cookbook.model_info", model_info)
    monkeypatch.setitem(sys.modules, "tinker_cookbook.renderers", renderers)
    monkeypatch.setitem(sys.modules, "tinker_cookbook.tokenizer_utils", tokenizer_utils)
    monkeypatch.setitem(sys.modules, "tinker_cookbook.supervised", supervised_pkg)
    monkeypatch.setitem(sys.modules, "tinker_cookbook.supervised.data", supervised_data)

    importlib.reload(importlib.import_module("src.tinker_api.sft_runner"))
    return types.SimpleNamespace(
        training_client=training_client,
        service_client=service_client,
        conversions=conversions,
    )
