from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.types import DataFormat, RawData


def load_raw_data(data_path: str) -> RawData:
    """
        take a local path and figure out what kind of data it points to, 
        load that data into a normalized structure, 
        and return metadata plus a readable summary.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data path does not exist: {data_path}")

    format_meta = detect_format(data_path) # figure out type of data (iamges, text, csv, etc..)
    file_type = format_meta["file_type"]

    if path.is_dir():
        # if the input is a folder, load all files in the directory 
        records = _load_directory_records(path, file_type)

    elif file_type == "csv":
        # read it as comma-separated tabular data
        records = _attach_source_path(_read_delimited_file(path, delimiter=","), path)

    elif file_type == "tsv":
        # read it as tab-separated data
        records = _attach_source_path(_read_delimited_file(path, delimiter="\t"), path)

    elif file_type in {"jsonl", "ndjson"}:
        # read it line by line as multiple JSON records
        records = _attach_source_path(_read_jsonl_file(path), path)

    elif file_type == "json":
        #read as standard json
        records = _attach_source_path(_read_json_file(path), path)

    elif file_type == "parquet":
        # read it as parquet table data.
        records = _attach_source_path(_read_parquet_file(path), path)

    elif format_meta["modality"] == "image":
        #don’t parse text from it, just create a simple dict with image info (its source, path, filname..)
        records = [_image_record(path)]
    else:
        records = _attach_source_path(_read_text_file(path), path)

    return {
        "records": records,
        "format_meta": format_meta,
        "human_readable": build_local_human_readable_report(path, format_meta, records),
    }


def detect_format(data_path: str) -> DataFormat:

    """ determins modality/format of data the system is dealing with (images/tabular/text..) 
        image data may need vision processing, while tabular data may need row normalization.
    """
    path = Path(data_path)

    if path.is_dir():  # if path is a folder, not a single file
       
        if _looks_like_image_dir(path): #is this folder entirely made of images?
            return {"modality": "image", "file_type": "image_dir", "encoding": "binary"}    # treat this as image dir

        return {"modality": "tabular", "file_type": "directory", "encoding": "utf-8"}       # format tells us to load its files one by one as structured/text data

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
    """
        builds a plain-text summary string describing what Mode A loaded from the local path.
        It does not change the data itself. It just creates a readable report for debugging, observability, and handoff inspection.
    """
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
    """ walks through every file under the directory recursively with rglob("*"), filters to actual files, 
        and then asks: “Are all of these files images?”

        `detect_format` uses it to classify the directory --> if all images, then classifies as image_dir
        if not, needs tabular style handling.
    """
    files = [child for child in path.rglob("*") if child.is_file()]
    if not files:
        return False
    image_files = [child for child in files if _is_image_path(child)]

    # If every file has an image extension, If even one file is not an image, it returns False.
    return len(image_files) == len(files)


def _is_image_path(path: Path) -> bool:
    """ it checks whether a file’s extension is one of the supported image types like .png, .jpg, or .gif """
    return path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".gif"}


def _image_record(path: Path) -> dict[str, str]:
    return {
        "source": "local_image",
        "path": str(path),
        "filename": path.name,
    }


def _attach_source_path(records: list[Any], path: Path) -> list[Any]:
    """
    [
    {"input": "hello", "output": "world"},
    {"input": "foo", "output": "bar"}
    ]

    then thsi turns it into
    [
    {
        "input": "hello",
        "output": "world",
        "source_path": ".../examples.csv",
        "row_index": 1,
    },
    {
        "input": "foo",
        "output": "bar",
        "source_path": ".../examples.csv",
        "row_index": 2,
    },
    ]
    """
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
