from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.types import DataFormat, RawData


def load_raw_data(data_path: str) -> RawData:
    """Acquisition step for Mode A: load heterogeneous local files or directories."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data path does not exist: {data_path}")

    format_meta = detect_format(data_path)
    file_type = format_meta["file_type"]

    if path.is_dir():
        records = _load_directory_records(path, file_type)
    elif file_type == "csv":
        records = _attach_source_path(_read_delimited_file(path, delimiter=","), path)
    elif file_type == "tsv":
        records = _attach_source_path(_read_delimited_file(path, delimiter="\t"), path)
    elif file_type in {"jsonl", "ndjson"}:
        records = _attach_source_path(_read_jsonl_file(path), path)
    elif file_type == "json":
        records = _attach_source_path(_read_json_file(path), path)
    elif file_type == "parquet":
        records = _attach_source_path(_read_parquet_file(path), path)
    elif format_meta["modality"] == "image":
        records = [_image_record(path)]
    else:
        records = _attach_source_path(_read_text_file(path), path)

    return {
        "records": records,
        "format_meta": format_meta,
        "human_readable": build_local_human_readable_report(path, format_meta, records),
    }


def detect_format(data_path: str) -> DataFormat:
    path = Path(data_path)

    if path.is_dir():
        if _looks_like_image_dir(path):
            return {"modality": "image", "file_type": "image_dir", "encoding": "binary"}
        return {"modality": "tabular", "file_type": "directory", "encoding": "utf-8"}

    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv", ".json", ".jsonl", ".ndjson", ".parquet"}:
        modality = "tabular"
    elif suffix in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
        modality = "image"
    else:
        modality = "text"

    encoding = "binary" if modality == "image" else "utf-8"
    return {"modality": modality, "file_type": suffix.lstrip(".") or "unknown", "encoding": encoding}


def build_local_human_readable_report(
    path: Path,
    format_meta: DataFormat,
    records: list[Any],
    max_preview_records: int = 5,
    max_preview_chars: int = 300,
) -> str:
    lines: list[str] = []
    lines.append("Mode A Local Data Acquisition Report")
    lines.append("=" * 40)
    lines.append(f"Source path: {path}")
    lines.append(f"Path kind: {'directory' if path.is_dir() else 'file'}")
    lines.append(f"Detected modality: {format_meta.get('modality', 'unknown')}")
    lines.append(f"Detected file type: {format_meta.get('file_type', 'unknown')}")
    lines.append(f"Encoding: {format_meta.get('encoding', 'unknown')}")
    lines.append(f"Records loaded: {len(records)}")
    lines.append("")
    lines.append("Preview:")

    if not records:
        lines.append("(no records loaded)")
        return "\n".join(lines)

    for idx, record in enumerate(records[:max_preview_records], start=1):
        if isinstance(record, dict) and isinstance(record.get("records"), list):
            nested = record.get("records", [])
            lines.append(
                f"[{idx}] source file: {record.get('source_path')} | "
                f"type={record.get('file_type')} | modality={record.get('modality')} | "
                f"nested_records={len(nested)}"
            )
            if nested:
                preview = json.dumps(nested[0], ensure_ascii=True)[:max_preview_chars]
                lines.append(f"First nested record: {preview}")
            lines.append("-" * 40)
            continue

        preview = json.dumps(record, ensure_ascii=True)[:max_preview_chars]
        lines.append(f"[{idx}] {preview}")
        lines.append("-" * 40)

    return "\n".join(lines)


def _read_delimited_file(path: Path, delimiter: str) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def _read_jsonl_file(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _read_json_file(path: Path) -> list[Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else [payload]


def _read_parquet_file(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        raise RuntimeError("Reading parquet requires pyarrow to be installed.") from exc

    table = pq.read_table(path)
    rows = table.to_pylist()
    return [dict(row) if isinstance(row, dict) else {"value": row} for row in rows]


def _read_text_file(path: Path) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return [{"input": line.strip(), "output": "unknown"} for line in text.splitlines() if line.strip()]


def _load_directory_records(path: Path, file_type: str) -> list[dict[str, Any]]:
    if file_type == "image_dir":
        image_paths = sorted(p for p in path.rglob("*") if p.is_file() and _is_image_path(p))
        return [_image_record(image_path) for image_path in image_paths]

    records: list[dict[str, Any]] = []
    for child in sorted(p for p in path.iterdir() if p.is_file()):
        child_format = detect_format(str(child))
        child_record: dict[str, Any] = {
            "source_path": str(child),
            "file_type": child_format["file_type"],
            "modality": child_format["modality"],
        }

        if child_format["file_type"] == "csv":
            child_record["records"] = _attach_source_path(_read_delimited_file(child, delimiter=","), child)
        elif child_format["file_type"] == "tsv":
            child_record["records"] = _attach_source_path(_read_delimited_file(child, delimiter="\t"), child)
        elif child_format["file_type"] in {"jsonl", "ndjson"}:
            child_record["records"] = _attach_source_path(_read_jsonl_file(child), child)
        elif child_format["file_type"] == "json":
            child_record["records"] = _attach_source_path(_read_json_file(child), child)
        elif child_format["file_type"] == "parquet":
            child_record["records"] = _attach_source_path(_read_parquet_file(child), child)
        elif child_format["modality"] == "image":
            child_record["records"] = [_image_record(child)]
        else:
            child_record["records"] = _attach_source_path(_read_text_file(child), child)

        records.append(child_record)

    return records


def _looks_like_image_dir(path: Path) -> bool:
    files = [child for child in path.rglob("*") if child.is_file()]
    if not files:
        return False
    image_files = [child for child in files if _is_image_path(child)]
    return len(image_files) == len(files)


def _is_image_path(path: Path) -> bool:
    return path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".gif"}


def _image_record(path: Path) -> dict[str, str]:
    return {
        "source": "local_image",
        "path": str(path),
        "filename": path.name,
    }


def _attach_source_path(records: list[Any], path: Path) -> list[Any]:
    enriched: list[Any] = []
    for idx, record in enumerate(records, start=1):
        if isinstance(record, dict):
            updated = dict(record)
            updated.setdefault("source_path", str(path))
            updated.setdefault("row_index", idx)
            enriched.append(updated)
        else:
            enriched.append(record)
    return enriched
