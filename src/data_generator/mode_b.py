from __future__ import annotations

import json
import os
import re
from urllib.request import urlopen

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
    fetch explicit HF dataset URLs and return combined raw records.

    Set DATA_GENERATOR_OFFLINE=1 for deterministic local test runs.
    """
    records: list[dict] = []
    offline = os.getenv("DATA_GENERATOR_OFFLINE", "").strip().lower() in {"1", "true", "yes"}

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

    for candidate in candidates:
        url = candidate["id"]
        try:
            with urlopen(url, timeout=20) as response:
                payload = response.read(1_200_000)
            text = payload.decode("utf-8", errors="replace")
        except Exception:
            text = json.dumps({"dataset": candidate["name"], "note": "fetch_failed"})
        for line in text.splitlines():
            clean = line.strip()
            if clean:
                records.append({"source": candidate["name"], "content": clean[:500]})
            if len(records) >= 5000:
                break
        if len(records) >= 5000:
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
