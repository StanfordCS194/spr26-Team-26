from __future__ import annotations

import csv
import json
from pathlib import Path

from src.types import DataFormat, RawData


def load_raw_data(data_path: str) -> RawData:
    """Acquisition step for Mode A: load heterogeneous local files."""
    path = Path(data_path)
    format_meta = detect_format(data_path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            records = list(csv.DictReader(handle))
    elif suffix in {".jsonl", ".ndjson"}:
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload if isinstance(payload, list) else [payload]
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
        records = [{"input": line.strip(), "output": "unknown"} for line in text.splitlines() if line.strip()]

    return {"records": records, "format_meta": format_meta}


def detect_format(data_path: str) -> DataFormat:
    suffix = Path(data_path).suffix.lower()
    if suffix in {".csv", ".tsv", ".json", ".jsonl", ".ndjson", ".parquet"}:
        modality = "tabular"
    elif suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        modality = "image"
    else:
        modality = "text"
    return {"modality": modality, "file_type": suffix.lstrip(".") or "unknown", "encoding": "utf-8"}
