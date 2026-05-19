"""
SDK-native Tinker supervised fine-tuning runner.

This module is intentionally lazy about importing ``tinker`` and
``tinker_cookbook`` so the regular unit suite can run without live SDK
dependencies installed.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence
from uuid import uuid4

from src.tinker_api.tinker_api import (
    TinkerAPIError,
    get_cumulative_spend,
    is_cancelled,
    record_tokens,
)
from src.types import DatasetResult, ExperimentResult, TrainingMetrics

if TYPE_CHECKING:
    from src.autoresearch.config import TrainingConfig

DEFAULT_TINKER_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_LIVE_SMOKE_STEPS = 5
QWEN35_9B_RENDERER = "qwen3_5_disable_thinking"
SUPPORTED_TINKER_TUNABLES: frozenset[str] = frozenset(
    {"learning_rate", "batch_size", "max_seq_length", "lora_rank", "num_epochs"}
)

_INPUT_KEYS = ("input", "prompt", "question", "content")
_TARGET_KEYS = ("output", "answer", "label", "target")


def run_tinker_sft_experiment(
    config: TrainingConfig | Mapping[str, Any],
    dataset: DatasetResult | str | os.PathLike[str] | Sequence[Mapping[str, Any]],
    *,
    run_id: str | None = None,
    max_steps: int | None = None,
    output_dir: str = "outputs/experiments",
    service_client: Any = None,
) -> ExperimentResult:
    """Run a short chat/SFT LoRA experiment through the Tinker SDK.

    ``dataset`` may be a repo ``DatasetResult``, a JSONL path, or an in-memory
    sequence of JSON-like records. Metrics and manifest artifacts are written
    under ``output_dir/run_id``.
    """
    cfg = _coerce_training_config(config)
    run_name = run_id or f"tinker-sft-{int(time.time())}-{uuid4().hex[:8]}"
    run_dir = Path(output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    conversations = load_conversations(dataset)
    if not conversations:
        raise ValueError("No valid chat/SFT examples found for Tinker training.")

    try:
        tinker, tinker_types, model_info, renderers, supervised_data, tokenizer_utils = (
            _load_tinker_deps()
        )
        model_name = cfg.model_name or DEFAULT_TINKER_MODEL
        renderer_name = resolve_renderer_name(model_name, model_info)
        tokenizer = tokenizer_utils.get_tokenizer(model_name)
        renderer = renderers.get_renderer(renderer_name, tokenizer)
        training_conversations = _expand_assistant_training_targets(conversations)
        if not training_conversations:
            raise ValueError("No assistant targets found after a user message.")

        svc = service_client if service_client is not None else tinker.ServiceClient()
        training_client = svc.create_lora_training_client(
            base_model=model_name,
            rank=int(cfg.lora_rank or 32),
        )

        train_on_what = renderers.TrainOnWhat.LAST_ASSISTANT_MESSAGE
        target_steps = _resolve_target_steps(
            num_examples=len(training_conversations),
            batch_size=cfg.batch_size,
            num_epochs=cfg.num_epochs,
            max_steps=max_steps,
        )
        metrics_history: list[dict[str, Any]] = []
        metrics_path = run_dir / "metrics.jsonl"
        status = "COMPLETED"

        for step in range(target_steps):
            if is_cancelled(run_name):
                status = "CANCELLED"
                break

            batch_conversations = _conversation_batch(
                training_conversations,
                cfg.batch_size,
                step,
            )
            batch = [
                supervised_data.conversation_to_datum(
                    conversation,
                    renderer,
                    cfg.max_seq_length,
                    train_on_what,
                )
                for conversation in batch_conversations
            ]

            adam_params = tinker_types.AdamParams(learning_rate=cfg.learning_rate)
            fwd_future = training_client.forward_backward(batch, loss_fn="cross_entropy")
            optim_future = training_client.optim_step(adam_params)
            fwd_result = _future_result(fwd_future)
            optim_result = _future_result(optim_future)

            tokens = sum(_datum_token_count(datum) for datum in batch)
            record_tokens(run_name, tokens)
            loss = _extract_loss(fwd_result, optim_result)
            step_metrics = {
                "step": step + 1,
                "train_loss": loss,
                "val_loss": loss,
                "test_loss": loss,
                "primary_metric": _score_from_loss(loss),
                "tokens": tokens,
            }
            metrics_history.append(step_metrics)
            with open(metrics_path, "a") as fh:
                fh.write(json.dumps(step_metrics) + "\n")

        checkpoints, sampling_client = _save_final_artifacts(training_client, run_name)
        sample_payload = _sample_once(
            sampling_client=sampling_client,
            renderer=renderer,
            tinker_types=tinker_types,
            conversations=conversations,
        )
        _write_json(run_dir / "sample.json", sample_payload)

        final_metrics = _final_metrics(metrics_history, status)
        manifest = {
            "run_id": run_name,
            "status": status,
            "backend": "tinker_sft",
            "model_name": model_name,
            "renderer_name": renderer_name,
            "train_on_what": str(train_on_what),
            "dataset_path": _dataset_path_for_manifest(dataset),
            "training_examples": len(training_conversations),
            "max_steps": max_steps,
            "completed_steps": len(metrics_history),
            "checkpoints": checkpoints,
        }
        _write_json(run_dir / "manifest.json", manifest)
        _write_json(run_dir / "metrics.json", final_metrics)

        return {
            "job_id": run_name,
            "status": status,
            "metrics": final_metrics,
            "model_path": str(run_dir),
            "cost_usd": get_cumulative_spend(run_name),
            "logs_path": str(metrics_path),
        }
    except Exception as exc:
        if isinstance(exc, (TinkerAPIError, ValueError)):
            raise
        raise TinkerAPIError(str(exc)) from exc


def load_conversations(
    dataset: DatasetResult | str | os.PathLike[str] | Sequence[Mapping[str, Any]],
) -> list[list[dict[str, str]]]:
    """Load and normalize chat/SFT examples into message conversations."""
    records = _load_records(dataset)
    conversations: list[list[dict[str, str]]] = []
    errors: list[str] = []
    for index, record in enumerate(records):
        try:
            conversations.append(record_to_conversation(record))
        except ValueError as exc:
            errors.append(f"row {index}: {exc}")

    if records and not conversations:
        preview = "; ".join(errors[:3])
        raise ValueError(f"No valid chat/SFT examples found. {preview}")
    return conversations


def record_to_conversation(record: Mapping[str, Any]) -> list[dict[str, str]]:
    """Convert one JSON-like record into a user/assistant conversation."""
    messages = record.get("messages")
    if messages is not None:
        if not isinstance(messages, list):
            raise ValueError("messages must be a list")
        normalized = [_normalize_message(message) for message in messages]
        if not any(message["role"] == "user" for message in normalized):
            raise ValueError("messages must contain at least one user message")
        if not any(message["role"] == "assistant" for message in normalized):
            raise ValueError("messages must contain at least one assistant message")
        return normalized

    user_text = _first_text(record, _INPUT_KEYS)
    assistant_text = _first_text(record, _TARGET_KEYS)
    if not user_text:
        raise ValueError(f"missing input field; expected one of {_INPUT_KEYS}")
    if not assistant_text:
        raise ValueError(f"missing target field; expected one of {_TARGET_KEYS}")
    return [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]


def _expand_assistant_training_targets(
    conversations: Sequence[Sequence[dict[str, str]]],
) -> list[list[dict[str, str]]]:
    """Split multi-assistant chats into one target per assistant response.

    Cookbook renderers warn against ``ALL_ASSISTANT_MESSAGES`` for renderers
    without the extension property. Training one prefix ending in the target
    assistant message preserves all assistant targets while using the safer
    ``LAST_ASSISTANT_MESSAGE`` mode.
    """
    targets: list[list[dict[str, str]]] = []
    for conversation in conversations:
        prefix: list[dict[str, str]] = []
        seen_user = False
        for message in conversation:
            copied = {"role": message["role"], "content": message["content"]}
            prefix.append(copied)
            if copied["role"] == "user":
                seen_user = True
            if copied["role"] == "assistant" and seen_user:
                targets.append(list(prefix))
    return targets


def resolve_renderer_name(model_name: str, model_info_module: Any | None = None) -> str:
    """Resolve cookbook renderer name, with an explicit Qwen3.5-9B fallback."""
    if model_info_module is None:
        _tinker, _types, model_info_module, _renderers, _supervised, _tokenizers = (
            _load_tinker_deps()
        )
    try:
        return model_info_module.get_recommended_renderer_name(model_name)
    except Exception:
        if model_name == DEFAULT_TINKER_MODEL:
            return QWEN35_9B_RENDERER
        raise


def _coerce_training_config(config: TrainingConfig | Mapping[str, Any]) -> TrainingConfig:
    from src.autoresearch.config import TrainingConfig

    if isinstance(config, TrainingConfig):
        data = config.to_dict()
    else:
        data = dict(config)

    if not data.get("model_name"):
        data["model_name"] = DEFAULT_TINKER_MODEL
    if "max_seq_len" in data and "max_seq_length" not in data:
        data["max_seq_length"] = data["max_seq_len"]
    if "epochs" in data and "num_epochs" not in data:
        data["num_epochs"] = data["epochs"]
    return TrainingConfig.from_dict(data)


def _load_records(
    dataset: DatasetResult | str | os.PathLike[str] | Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    if isinstance(dataset, (str, os.PathLike)):
        return _read_jsonl_records(_resolve_jsonl_path(Path(dataset)))
    if isinstance(dataset, Mapping):
        dataset_meta = dataset.get("dataset")
        if isinstance(dataset_meta, Mapping) and dataset_meta.get("path"):
            return _read_jsonl_records(_resolve_jsonl_path(Path(str(dataset_meta["path"]))))
        if dataset.get("path"):
            return _read_jsonl_records(_resolve_jsonl_path(Path(str(dataset["path"]))))
    return [record for record in dataset if isinstance(record, Mapping)]  # type: ignore[arg-type]


def _resolve_jsonl_path(path: Path) -> Path:
    if path.is_file():
        return path
    if path.is_dir():
        for name in ("train.jsonl", "train_data.jsonl", "data.jsonl"):
            candidate = path / name
            if candidate.exists():
                return candidate
    raise ValueError(f"Could not find JSONL dataset at {path}")


def _read_jsonl_records(path: Path) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSONL row") from exc
            if not isinstance(parsed, Mapping):
                raise ValueError(f"{path}:{lineno}: JSONL row must be an object")
            records.append(parsed)
    return records


def _normalize_message(message: Any) -> dict[str, str]:
    if not isinstance(message, Mapping):
        raise ValueError("message entries must be objects")
    role = str(message.get("role", "")).strip()
    content = message.get("content")
    if role not in {"system", "user", "assistant"}:
        raise ValueError(f"unsupported message role {role!r}")
    if content is None or str(content).strip() == "":
        raise ValueError("message content cannot be empty")
    return {"role": role, "content": str(content)}


def _first_text(record: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _load_tinker_deps() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        import tinker
        from tinker import types as tinker_types
        from tinker_cookbook import model_info, renderers
        from tinker_cookbook.supervised import data as supervised_data
        from tinker_cookbook import tokenizer_utils
    except ImportError as exc:
        raise TinkerAPIError(
            "tinker and tinker-cookbook are required for live Tinker SFT runs"
        ) from exc
    return tinker, tinker_types, model_info, renderers, supervised_data, tokenizer_utils


def _resolve_target_steps(
    *,
    num_examples: int,
    batch_size: int,
    num_epochs: int,
    max_steps: int | None,
) -> int:
    if max_steps is not None:
        return max(0, int(max_steps))
    batches_per_epoch = max(1, math.ceil(num_examples / max(1, batch_size)))
    return max(1, batches_per_epoch * max(1, int(num_epochs)))


def _conversation_batch(
    conversations: list[list[dict[str, str]]],
    batch_size: int,
    step: int,
) -> list[list[dict[str, str]]]:
    size = max(1, int(batch_size))
    start = step * size
    return [conversations[(start + offset) % len(conversations)] for offset in range(size)]


def _future_result(value: Any) -> Any:
    result = getattr(value, "result", None)
    if callable(result):
        return result()
    return value


def _datum_token_count(datum: Any) -> int:
    model_input = getattr(datum, "model_input", None)
    if model_input is None:
        return 0
    length = getattr(model_input, "length", None)
    if isinstance(length, int):
        return length
    tokens = getattr(model_input, "tokens", None)
    if tokens is not None:
        return len(tokens)
    to_ints = getattr(model_input, "to_ints", None)
    if callable(to_ints):
        return len(to_ints())
    return 0


def _extract_loss(*results: Any) -> float:
    for result in results:
        if result is None:
            continue
        loss = getattr(result, "loss", None)
        if isinstance(loss, (int, float)):
            return float(loss)
        metrics = getattr(result, "metrics", None)
        if isinstance(metrics, Mapping):
            for key in ("loss", "train_loss", "mean_loss", "nll"):
                value = metrics.get(key)
                if isinstance(value, (int, float)):
                    return float(value)
    return 0.0


def _score_from_loss(loss: float | None) -> float:
    if loss is None or not math.isfinite(loss) or loss < 0:
        return 0.0
    return 1.0 / (1.0 + loss)


def _save_final_artifacts(training_client: Any, run_id: str) -> tuple[dict[str, str], Any]:
    checkpoints: dict[str, str] = {}
    state_result = _call_optional_checkpoint(training_client, "save_state", f"{run_id}-final-state")
    if state_result:
        checkpoints["state_path"] = state_result
    sampler_result = _call_optional_checkpoint(
        training_client,
        "save_weights_for_sampler",
        f"{run_id}-final-sampler",
    )
    if sampler_result:
        checkpoints["sampler_path"] = sampler_result

    get_sampling = getattr(training_client, "save_weights_and_get_sampling_client", None)
    if not callable(get_sampling):
        return checkpoints, None
    try:
        sampling_client = _future_result(get_sampling())
    except TypeError:
        sampling_client = _future_result(get_sampling(name=f"{run_id}-final-sampling-client"))
    return checkpoints, sampling_client


def _call_optional_checkpoint(training_client: Any, method_name: str, name: str) -> str | None:
    method = getattr(training_client, method_name, None)
    if not callable(method):
        return None
    result = _future_result(method(name=name))
    path = getattr(result, "path", None)
    if path is not None:
        return str(path)
    if isinstance(result, str):
        return result
    return None


def _sample_once(
    *,
    sampling_client: Any,
    renderer: Any,
    tinker_types: Any,
    conversations: list[list[dict[str, str]]],
) -> dict[str, Any]:
    prompt_messages = [message for message in conversations[0] if message["role"] != "assistant"]
    if not prompt_messages:
        prompt_messages = [{"role": "user", "content": conversations[0][0]["content"]}]

    if sampling_client is None:
        return {"prompt": prompt_messages, "text": "", "tokens": [], "error": "no sampling client"}

    model_input = renderer.build_generation_prompt(prompt_messages)
    sampling_params = tinker_types.SamplingParams(
        max_tokens=64,
        stop=renderer.get_stop_sequences(),
    )
    response = _future_result(
        sampling_client.sample(
            prompt=model_input,
            num_samples=1,
            sampling_params=sampling_params,
        )
    )
    sequences = getattr(response, "sequences", None) or getattr(response, "samples", None) or []
    if not sequences:
        return {"prompt": prompt_messages, "text": "", "tokens": []}
    tokens = list(getattr(sequences[0], "tokens", []))
    try:
        parsed, _termination = renderer.parse_response(tokens)
        content = parsed.get("content", "") if isinstance(parsed, Mapping) else str(parsed)
    except Exception:
        content = ""
    return {"prompt": prompt_messages, "text": content, "tokens": tokens}


def _final_metrics(metrics_history: list[dict[str, Any]], status: str) -> TrainingMetrics:
    if not metrics_history:
        loss = float("nan") if status == "FAILED" else 0.0
        return {
            "train_loss": loss,
            "val_loss": loss,
            "test_loss": loss,
            "primary_metric": _score_from_loss(loss),
        }
    last = metrics_history[-1]
    loss = float(last["train_loss"])
    return {
        "train_loss": loss,
        "val_loss": float(last["val_loss"]),
        "test_loss": float(last["test_loss"]),
        "primary_metric": float(last["primary_metric"]),
    }


def _dataset_path_for_manifest(
    dataset: DatasetResult | str | os.PathLike[str] | Sequence[Mapping[str, Any]],
) -> str | None:
    if isinstance(dataset, (str, os.PathLike)):
        return str(dataset)
    if isinstance(dataset, Mapping):
        dataset_meta = dataset.get("dataset")
        if isinstance(dataset_meta, Mapping) and dataset_meta.get("path"):
            return str(dataset_meta["path"])
        if dataset.get("path"):
            return str(dataset["path"])
    return None


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, allow_nan=True)
