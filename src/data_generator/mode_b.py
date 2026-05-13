from __future__ import annotations

import os
import re
from typing import Any

from src.types import HFCandidate, OrchestrationConfig, RawData


def parse_explicit_hf_dataset_ids(config: OrchestrationConfig, data_path: str | None = None) -> list[str]:
    """Extract explicit HF dataset IDs from config/prompt/path."""
    raw_items: list[str] = []

    for key in ("hf_dataset_ids", "hf_dataset_urls", "explicit_hf_datasets", "dataset_ids", "sources"):
        value = config.get(key)  # type: ignore[arg-type]
        raw_items.extend(_coerce_source_tokens(value))

    tp = config.get("training_procedure", {})
    if isinstance(tp, dict):
        for key in ("hf_dataset_ids", "hf_dataset_urls", "dataset_ids", "sources", "notes"):
            value = tp.get(key)
            raw_items.extend(_coerce_source_tokens(value))

    # New orchestrator envelope format support.
    data_request = config.get("data_request")  # type: ignore[arg-type]
    if isinstance(data_request, dict):
        raw_items.extend(_coerce_source_tokens(data_request.get("sources")))

    prompt = str(config.get("prompt", ""))
    raw_items.extend(_extract_hf_mentions(prompt))

    if data_path and str(data_path).startswith("hf://"):
        raw_items.append(str(data_path).replace("hf://", "", 1))

    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        ds_id = _normalize_hf_dataset_id(item)
        if not ds_id:
            continue
        if ds_id not in seen:
            seen.add(ds_id)
            normalized.append(ds_id)
    return normalized


def build_explicit_hf_candidates(dataset_ids: list[str], task_type: str) -> list[HFCandidate]:
    candidates: list[HFCandidate] = []
    for idx, dataset_id in enumerate(dataset_ids):
        candidates.append(
            {
                "id": f"https://huggingface.co/datasets/{dataset_id}",
                "name": dataset_id,
                "num_examples": max(5000 - (idx * 200), 100),
                "license": "unknown",
                "task_categories": [task_type],
                "download_size": max(20_000_000 - (idx * 500_000), 100_000),
            }
        )
    return candidates


def fetch_hf_datasets(candidates: list[HFCandidate]) -> RawData:
    """
    Acquisition step for Mode B:
    fetch explicit HF datasets and return combined raw records.

    Set DATA_GENERATOR_OFFLINE=1 for deterministic local test runs.
    """
    records: list[dict] = []
    offline = os.getenv("DATA_GENERATOR_OFFLINE", "").strip().lower() in {"1", "true", "yes"}
    max_rows_per_dataset = _read_int_env("DATA_GENERATOR_MAX_ROWS_PER_DATASET", 80)
    max_total_records = _read_int_env("DATA_GENERATOR_MAX_TOTAL_RECORDS", 5000)

    if offline:
        for candidate in candidates:
            records.append(
                {
                    "source": candidate["name"],
                    "content": f"offline_placeholder: acquired metadata for {candidate['name']}",
                }
            )
        return {
            "records": records or [{"source": "none", "content": "offline_empty"}],
            "format_meta": {"modality": "text", "file_type": "hf_bundle", "encoding": "utf-8"},
        }

    # Fetch actual rows from Hugging Face datasets (no HTML fallback).
    for candidate in candidates:
        dataset_id = candidate["name"]
        try:
            dataset_records = _fetch_with_hf_datasets(dataset_id, max_rows_per_dataset)
            if dataset_records:
                records.extend(dataset_records)
            else:
                records.append(
                    {
                        "source": dataset_id,
                        "note": "fetch_empty",
                        "error": "dataset loaded but no usable rows were extracted",
                    }
                )
        except Exception as exc:
            records.append(
                {
                    "source": dataset_id,
                    "note": "fetch_failed",
                    "error": _short_error(exc),
                }
            )
        if len(records) >= max_total_records:
            break

    if not records:
        records = [{"source": "none", "content": "empty"}]
    return {
        "records": records,
        "format_meta": {"modality": "text", "file_type": "hf_bundle", "encoding": "utf-8"},
    }


def _split_tokens(value: str) -> list[str]:
    parts = re.split(r"[\n,; ]+", value)
    return [p.strip() for p in parts if p.strip()]


def _coerce_source_tokens(value: object) -> list[str]:
    tokens: list[str] = []
    if isinstance(value, str):
        tokens.extend(_split_tokens(value))
        return tokens
    if isinstance(value, list):
        for item in value:
            tokens.extend(_coerce_source_tokens(item))
        return tokens
    if isinstance(value, dict):
        # Accept flexible source object shapes.
        source_type = str(value.get("type", "")).strip().lower()
        if source_type and source_type not in {"hf_dataset", "huggingface_dataset"}:
            return tokens
        for key in ("id", "url", "dataset_id", "dataset_url", "value"):
            field = value.get(key)
            if isinstance(field, str) and field.strip():
                tokens.append(field.strip())
        return tokens
    return tokens


def _extract_hf_mentions(value: str) -> list[str]:
    mentions: list[str] = []
    for match in re.findall(r"huggingface\.co/datasets/([A-Za-z0-9._-]+/[A-Za-z0-9._-]+)", value):
        mentions.append(match)
    for match in re.findall(r"\b([A-Za-z0-9._-]+/[A-Za-z0-9._-]+)\b", value):
        mentions.append(match)
    return mentions


def _normalize_hf_dataset_id(value: str) -> str | None:
    token = value.strip()
    if not token:
        return None
    token = token.removeprefix("hf://")
    token = token.removeprefix("https://huggingface.co/datasets/")
    token = token.removeprefix("http://huggingface.co/datasets/")
    token = token.strip("/")
    if "/" not in token:
        return None
    left, right = token.split("/", 1)
    if not left or not right:
        return None
    if not re.fullmatch(r"[A-Za-z0-9._-]+", left):
        return None
    if not re.fullmatch(r"[A-Za-z0-9._-]+", right):
        return None
    return f"{left}/{right}"


def _fetch_with_hf_datasets(dataset_id: str, max_rows_per_dataset: int) -> list[dict]:
    from datasets import get_dataset_config_names, load_dataset

    records: list[dict] = []
    try:
        ds_obj = load_dataset(dataset_id)
    except Exception as first_error:
        # Some datasets require explicit config names (e.g., GLUE).
        configs = get_dataset_config_names(dataset_id)
        if not configs:
            raise first_error
        ds_obj = load_dataset(dataset_id, configs[0])

    split_items = ds_obj.items() if hasattr(ds_obj, "items") else [("train", ds_obj)]
    per_split_limit = max(1, max_rows_per_dataset // max(1, len(list(split_items))))
    split_items = ds_obj.items() if hasattr(ds_obj, "items") else [("train", ds_obj)]

    for split_name, split_ds in split_items:
        take_n = min(per_split_limit, len(split_ds))
        for row in split_ds.select(range(take_n)):
            text_val, label_val = _extract_input_and_label(row, getattr(split_ds, "features", None))
            if not text_val:
                continue
            rec = {
                "source": dataset_id,
                "split": str(split_name),
                "input": text_val[:1000],
                "label": label_val,
                "content": text_val[:500],
            }
            records.append(rec)
            if len(records) >= max_rows_per_dataset:
                break
        if len(records) >= max_rows_per_dataset:
            break
    return records


def _extract_input_and_label(row: dict, features: Any = None) -> tuple[str, str]:
    text_keys = ["text", "sentence", "question", "input", "prompt", "utterance", "content"]
    label_keys = ["label_text", "label", "intent", "target", "output", "answer"]

    text_val = ""
    for key in text_keys:
        if key in row and row[key] is not None:
            text_val = str(row[key]).strip()
            if text_val:
                break
    if not text_val:
        for key, value in row.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float)):
                text_val = str(value).strip()
                if text_val:
                    break

    label_val = ""
    for key in label_keys:
        if key in row and row[key] is not None:
            raw_label = row[key]
            if isinstance(raw_label, int) and features is not None and key in features:
                feature_obj = features[key]
                names = getattr(feature_obj, "names", None)
                if isinstance(names, list) and 0 <= raw_label < len(names):
                    label_val = str(names[raw_label]).strip()
                else:
                    label_val = str(raw_label).strip()
            else:
                label_val = str(raw_label).strip()
            if label_val:
                break

    return text_val, label_val


def _read_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
        return value if value > 0 else default
    except ValueError:
        return default


def _short_error(exc: Exception) -> str:
    text = str(exc).strip()
    return text[:300] if text else exc.__class__.__name__
