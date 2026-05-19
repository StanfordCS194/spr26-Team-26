"""Helpers for normalizing operator-provided dataset sources."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote, urlparse


_HF_DATASET_PART = re.compile(r"^[A-Za-z0-9._-]+$")
LOCAL_DATA_SUFFIXES = {
    ".csv",
    ".json",
    ".jsonl",
    ".parquet",
    ".txt",
    ".tsv",
}


def normalize_hf_dataset_source(value: str | None) -> str | None:
    """Return a canonical ``hf://org/dataset`` source, or ``None``."""
    if value is None:
        return None

    token = value.strip()
    if not token:
        return None

    if token.startswith(("https://", "http://")):
        parsed = urlparse(token)
        if parsed.netloc not in {"huggingface.co", "www.huggingface.co"}:
            return None
        parts = [unquote(part) for part in parsed.path.strip("/").split("/") if part]
        if len(parts) < 3 or parts[0] != "datasets":
            return None
        dataset_id = f"{parts[1]}/{parts[2]}"
        return _canonical_hf_dataset_id(dataset_id)

    if token.startswith("hf://"):
        token = token.removeprefix("hf://")
    elif "://" in token:
        return None

    return _canonical_hf_dataset_id(token)


def looks_like_local_data_path(value: str | None) -> bool:
    if value is None:
        return False
    token = value.strip()
    if not token:
        return False
    if token.startswith(("/", "./", "../", "~")):
        return True
    return Path(token).suffix.lower() in LOCAL_DATA_SUFFIXES


def _canonical_hf_dataset_id(value: str) -> str | None:
    token = value.strip().strip("/")
    if token.count("/") != 1:
        return None

    namespace, name = token.split("/", 1)
    if not namespace or not name:
        return None
    if not _HF_DATASET_PART.fullmatch(namespace):
        return None
    if not _HF_DATASET_PART.fullmatch(name):
        return None

    return f"hf://{namespace}/{name}"
