from __future__ import annotations

import csv
import json
import re
import zipfile
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

try:
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover
    pq = None

TEXT_SUFFIXES = {".txt", ".md", ".csv", ".tsv", ".json", ".jsonl", ".ndjson", ".html", ".htm", ".xml"}


def evaluate_artifact_relevance(
    file_path: Path,
    keyword_bank: list[str],
    label_space: list[str],
    min_score: float = 0.2,
) -> tuple[float, list[str], bool]:
    terms = _build_terms(keyword_bank, label_space)
    if not terms:
        return 0.0, [], False

    sample_text = _extract_text_sample(file_path)
    if not sample_text:
        return 0.0, [], False

    normalized_sample = _normalize_for_match(sample_text)
    matched = [term for term in terms if term in normalized_sample]

    informative_tokens = _build_informative_tokens(terms)
    token_hits = [token for token in informative_tokens if _contains_token(normalized_sample, token)]

    cue_bonus = _classification_cue_bonus(normalized_sample)

    denominator = max(min(len(terms), 12), 1)
    exact_score = min(len(set(matched)) / denominator, 1.0)
    token_score = min(len(set(token_hits)) / max(min(len(informative_tokens), 20), 1), 1.0) * 0.35
    score = min(exact_score + token_score + cue_bonus, 1.0)

    is_relevant = (
        score >= min_score
        or len(set(matched)) >= 2
        or (cue_bonus >= 0.2 and len(set(token_hits)) >= 1)
    )
    matched_terms = sorted(set(matched + [f"token:{token}" for token in token_hits[:20]]))
    return round(score, 4), matched_terms, is_relevant


def extract_text_preview(file_path: Path, max_chars: int = 12000) -> str:
    """Public helper for human-readable preview generation."""
    return _extract_text_sample(file_path=file_path, max_chars=max_chars)


def assess_artifact_reasonableness(file_path: Path) -> dict[str, Any]:
    """Lightweight sanity checks to ensure downloaded raw data looks usable."""
    if not file_path.exists():
        return {"is_reasonable": False, "reason": "file_missing", "details": ["File does not exist."]}
    if file_path.stat().st_size == 0:
        return {"is_reasonable": False, "reason": "empty_file", "details": ["Downloaded file is empty."]}

    suffix = file_path.suffix.lower()
    if suffix == ".zip":
        return _assess_zip(file_path)
    if suffix == ".csv" or suffix == ".tsv":
        return _assess_csv(file_path)
    if suffix in {".json", ".jsonl", ".ndjson"}:
        return _assess_json(file_path)
    if suffix == ".parquet":
        return _assess_parquet(file_path)
    if suffix in TEXT_SUFFIXES:
        return _assess_text(file_path)
    if suffix == ".pdf":
        return _assess_pdf(file_path)

    # Generic fallback for unknown binaries.
    preview = _extract_binary_strings(file_path, max_chars=2000)
    if len(preview.strip()) < 40:
        return {
            "is_reasonable": False,
            "reason": "low_signal_binary",
            "details": ["Could not extract enough readable content from binary file."],
        }
    return {"is_reasonable": True, "reason": "ok", "details": ["Binary file contains readable content snippets."]}


def _build_terms(keyword_bank: list[str], label_space: list[str]) -> list[str]:
    seen: set[str] = set()
    terms: list[str] = []
    for token in keyword_bank + label_space:
        cleaned = " ".join(token.replace("_", " ").replace("-", " ").split()).strip().lower()
        if len(cleaned) < 3:
            continue
        if cleaned not in seen:
            seen.add(cleaned)
            terms.append(cleaned)
    return terms


def _normalize_for_match(text: str) -> str:
    return " ".join(text.lower().replace("_", " ").replace("-", " ").split())


def _build_informative_tokens(terms: list[str]) -> list[str]:
    stop = {
        "data",
        "dataset",
        "label",
        "labels",
        "classification",
        "text",
        "language",
        "support",
        "customer",
        "issue",
        "issues",
        "normal",
        "other",
    }
    seen: set[str] = set()
    tokens: list[str] = []
    for term in terms:
        for token in term.split():
            tok = token.strip().lower()
            if len(tok) < 4 or tok in stop or tok.isdigit():
                continue
            if tok not in seen:
                seen.add(tok)
                tokens.append(tok)
    return tokens


def _contains_token(normalized_text: str, token: str) -> bool:
    return bool(re.search(rf"\b{re.escape(token)}\b", normalized_text))


def _classification_cue_bonus(normalized_sample: str) -> float:
    bonus = 0.0
    has_text_field = any(cue in normalized_sample for cue in ("text=", "utterance", "message", "sentence"))
    has_label_field = any(cue in normalized_sample for cue in ("label=", "label text", "intent", "category"))
    if has_text_field and has_label_field:
        bonus += 0.2
    if "columns=" in normalized_sample and ("label" in normalized_sample or "intent" in normalized_sample):
        bonus += 0.1
    return min(bonus, 0.3)


def _extract_text_sample(file_path: Path, max_chars: int = 12000) -> str:
    suffix = file_path.suffix.lower()
    if suffix == ".zip":
        return _extract_zip_preview(file_path, max_chars=max_chars)
    if suffix == ".pdf":
        return _extract_pdf_preview(file_path, max_chars=max_chars)
    if suffix == ".csv":
        return _extract_csv_preview(file_path, max_chars=max_chars)
    if suffix == ".json" or suffix == ".jsonl" or suffix == ".ndjson":
        return _extract_json_preview(file_path, max_chars=max_chars)
    if suffix == ".parquet":
        return _extract_parquet_preview(file_path, max_chars=max_chars)
    if suffix in TEXT_SUFFIXES:
        return _extract_text_file(file_path, max_chars=max_chars)
    return _extract_binary_strings(file_path, max_chars=max_chars)


def _extract_text_file(file_path: Path, max_chars: int) -> str:
    try:
        return file_path.read_text(encoding="utf-8", errors="ignore")[:max_chars]
    except Exception:
        return ""


def _extract_csv_preview(file_path: Path, max_chars: int) -> str:
    fragments: list[str] = []
    try:
        with file_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.reader(handle)
            for row in reader:
                fragments.append(" | ".join(cell[:120] for cell in row))
                if len("\n".join(fragments)) >= max_chars:
                    break
    except Exception:
        return _extract_text_file(file_path, max_chars=max_chars)
    return "\n".join(fragments)[:max_chars]


def _extract_json_preview(file_path: Path, max_chars: int) -> str:
    try:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

    stripped = text.strip()
    if not stripped:
        return ""

    if file_path.suffix.lower() in {".jsonl", ".ndjson"}:
        lines = stripped.splitlines()
        collected: list[str] = []
        total = 0
        for line in lines:
            if not line.strip():
                continue
            collected.append(line)
            total += len(line) + 1
            if total >= max_chars:
                break
        return "\n".join(collected)[:max_chars]

    try:
        parsed = json.loads(stripped)
    except Exception:
        return stripped[:max_chars]

    if isinstance(parsed, list):
        return json.dumps(parsed, ensure_ascii=False)[:max_chars]
    if isinstance(parsed, dict):
        return json.dumps(parsed, ensure_ascii=False)[:max_chars]
    return str(parsed)[:max_chars]


def _extract_zip_preview(file_path: Path, max_chars: int) -> str:
    parts: list[str] = []
    try:
        with zipfile.ZipFile(file_path, "r") as archive:
            names = archive.namelist()
            parts.extend(names)
            budget = max(max_chars - len("\n".join(parts)), 0)
            for member in names:
                if budget <= 0:
                    break
                try:
                    chunk_size = min(max(budget, 0), 20000)
                    data = archive.read(member)[:chunk_size]
                    if not _looks_data_like_member(member, data):
                        continue
                    parts.append(data.decode("utf-8", errors="ignore"))
                    budget = max(max_chars - len("\n".join(parts)), 0)
                except Exception:
                    continue
    except Exception:
        return ""
    return "\n".join(parts)[:max_chars]


def _extract_pdf_preview(file_path: Path, max_chars: int) -> str:
    try:
        data = file_path.read_bytes()
    except Exception:
        return ""

    # Lightweight fallback: extract printable strings from PDF bytes.
    strings = re.findall(rb"[A-Za-z][A-Za-z0-9 ,._:\-()]{4,}", data)
    joined = "\n".join(item.decode("latin1", errors="ignore") for item in strings[:600])
    return joined[:max_chars]


def _extract_binary_strings(file_path: Path, max_chars: int) -> str:
    try:
        data = file_path.read_bytes()
    except Exception:
        return ""
    strings = re.findall(rb"[A-Za-z][A-Za-z0-9 ,._:\-()]{4,}", data)
    joined = "\n".join(item.decode("latin1", errors="ignore") for item in strings[:500])
    return joined[:max_chars]


def _assess_zip(file_path: Path) -> dict[str, Any]:
    try:
        with zipfile.ZipFile(file_path, "r") as archive:
            bad_member = archive.testzip()
            if bad_member is not None:
                return {
                    "is_reasonable": False,
                    "reason": "zip_corrupt_member",
                    "details": [f"Corrupt ZIP member: {bad_member}"],
                }
            members = [name for name in archive.namelist() if not name.endswith("/")]
            if not members:
                return {"is_reasonable": False, "reason": "zip_empty", "details": ["ZIP has no files."]}

            data_like: list[str] = []
            for name in members:
                try:
                    sample = archive.read(name)[:4096]
                except Exception:
                    continue
                if _looks_data_like_member(name, sample):
                    data_like.append(name)
            if not data_like:
                return {
                    "is_reasonable": False,
                    "reason": "zip_no_data_files",
                    "details": ["ZIP has files but none look like data/text artifacts."],
                }

            sample_preview = ""
            for member in data_like[:3]:
                try:
                    sample_preview = archive.read(member)[:1800].decode("utf-8", errors="ignore")
                except Exception:
                    continue
                if sample_preview.strip():
                    break

            if len(sample_preview.strip()) < 20:
                return {
                    "is_reasonable": False,
                    "reason": "zip_low_signal",
                    "details": ["ZIP data files exist but extracted preview looked empty/low-signal."],
                }

            return {
                "is_reasonable": True,
                "reason": "ok",
                "details": [f"ZIP valid with {len(members)} files and {len(data_like)} data-like files."],
            }
    except Exception as exc:
        return {"is_reasonable": False, "reason": "zip_read_error", "details": [f"Could not read ZIP: {exc}"]}


def _assess_csv(file_path: Path) -> dict[str, Any]:
    delimiter = "\t" if file_path.suffix.lower() == ".tsv" else ","
    row_count = 0
    nonempty_rows = 0
    try:
        with file_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.reader(handle, delimiter=delimiter)
            for row in reader:
                row_count += 1
                if any(cell.strip() for cell in row):
                    nonempty_rows += 1
                if row_count >= 200:
                    break
    except Exception as exc:
        return {"is_reasonable": False, "reason": "csv_read_error", "details": [f"CSV read failed: {exc}"]}

    if nonempty_rows < 2:
        return {"is_reasonable": False, "reason": "csv_too_small", "details": ["CSV has fewer than 2 non-empty rows."]}
    return {"is_reasonable": True, "reason": "ok", "details": [f"CSV preview rows={row_count}, nonempty_rows={nonempty_rows}."]}


def _assess_json(file_path: Path) -> dict[str, Any]:
    try:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        return {"is_reasonable": False, "reason": "json_read_error", "details": [f"JSON read failed: {exc}"]}
    stripped = text.strip()
    if not stripped:
        return {"is_reasonable": False, "reason": "json_empty", "details": ["JSON file is empty."]}

    if file_path.suffix.lower() in {".jsonl", ".ndjson"}:
        lines = [line for line in stripped.splitlines() if line.strip()]
        if len(lines) < 2:
            return {"is_reasonable": False, "reason": "jsonl_too_small", "details": ["JSONL has fewer than 2 non-empty lines."]}
        return {"is_reasonable": True, "reason": "ok", "details": [f"JSONL non-empty lines={len(lines)}."]}

    try:
        parsed = json.loads(stripped)
    except Exception as exc:
        return {"is_reasonable": False, "reason": "json_parse_error", "details": [f"JSON parse failed: {exc}"]}

    if isinstance(parsed, list) and len(parsed) == 0:
        return {"is_reasonable": False, "reason": "json_empty_list", "details": ["JSON list is empty."]}
    if isinstance(parsed, dict) and len(parsed) == 0:
        return {"is_reasonable": False, "reason": "json_empty_object", "details": ["JSON object has no keys."]}
    return {"is_reasonable": True, "reason": "ok", "details": ["JSON parsed with non-empty content."]}


def _assess_parquet(file_path: Path) -> dict[str, Any]:
    if pd is None and pq is None:
        if file_path.stat().st_size < 200:
            return {"is_reasonable": False, "reason": "parquet_too_small", "details": ["Parquet file is too small."]}
        return {
            "is_reasonable": True,
            "reason": "ok",
            "details": ["Parquet accepted by size check (pandas/pyarrow unavailable)."],
        }

    if pd is not None:
        try:
            frame = pd.read_parquet(file_path)
        except Exception as exc:
            return {"is_reasonable": False, "reason": "parquet_read_error", "details": [f"Parquet read failed: {exc}"]}

        if frame.empty:
            return {"is_reasonable": False, "reason": "parquet_empty", "details": ["Parquet has zero rows."]}
        return {
            "is_reasonable": True,
            "reason": "ok",
            "details": [f"Parquet rows={len(frame)}, cols={len(frame.columns)}."],
        }

    try:
        table = pq.read_table(file_path)
    except Exception as exc:
        return {"is_reasonable": False, "reason": "parquet_read_error", "details": [f"Parquet read failed: {exc}"]}

    if table.num_rows == 0:
        return {"is_reasonable": False, "reason": "parquet_empty", "details": ["Parquet has zero rows."]}
    return {
        "is_reasonable": True,
        "reason": "ok",
        "details": [f"Parquet rows={table.num_rows}, cols={table.num_columns}."],
    }


def _assess_text(file_path: Path) -> dict[str, Any]:
    preview = _extract_text_file(file_path, max_chars=4000)
    cleaned = preview.strip()
    if len(cleaned) < 20:
        return {"is_reasonable": False, "reason": "text_too_short", "details": ["Text content too short to be useful."]}
    line_count = len([line for line in cleaned.splitlines() if line.strip()])
    if line_count < 2:
        return {"is_reasonable": False, "reason": "text_too_few_lines", "details": ["Text has fewer than 2 non-empty lines."]}
    return {"is_reasonable": True, "reason": "ok", "details": [f"Text preview chars={len(cleaned)}, lines={line_count}."]}


def _assess_pdf(file_path: Path) -> dict[str, Any]:
    preview = _extract_pdf_preview(file_path, max_chars=5000)
    if len(preview.strip()) < 60:
        return {"is_reasonable": False, "reason": "pdf_low_text_signal", "details": ["PDF preview had too little extractable text."]}
    return {"is_reasonable": True, "reason": "ok", "details": ["PDF has extractable text content."]}


def _looks_data_like_member(member_name: str, sample: bytes) -> bool:
    lowered = member_name.lower()
    if lowered.endswith((".csv", ".tsv", ".txt", ".md", ".json", ".jsonl", ".ndjson", ".xml")):
        return True
    # Some popular corpora use extension-less files (e.g., SMSSpamCollection, readme).
    if lowered.endswith(("readme", "smsspamcollection")):
        return True
    return _is_likely_text_bytes(sample)


def _is_likely_text_bytes(sample: bytes) -> bool:
    if not sample:
        return False
    if b"\x00" in sample:
        return False
    printable = 0
    for b in sample:
        if b in (9, 10, 13) or 32 <= b <= 126:
            printable += 1
    ratio = printable / max(len(sample), 1)
    return ratio >= 0.82


def _extract_parquet_preview(file_path: Path, max_chars: int) -> str:
    if pd is not None:
        try:
            frame = pd.read_parquet(file_path)
        except Exception as exc:
            return f"<failed to read parquet: {exc}>"

        if frame.empty:
            return "<parquet file has zero rows>"
        head = frame.head(30)
        preview = f"columns={list(head.columns)}\n" + head.to_csv(index=False)
        return preview[:max_chars]

    if pq is None:
        return f"<parquet file: {file_path.name}, pandas/pyarrow unavailable for preview>"

    try:
        table = pq.read_table(file_path).slice(0, 30)
    except Exception as exc:
        return f"<failed to read parquet: {exc}>"

    if table.num_rows == 0:
        return "<parquet file has zero rows>"

    columns = list(table.column_names)
    data = table.to_pydict()
    lines = [f"columns={columns}"]
    for row_idx in range(table.num_rows):
        row_parts: list[str] = []
        for col in columns:
            value = data.get(col, [None] * table.num_rows)[row_idx]
            row_parts.append(f"{col}={value}")
        lines.append(", ".join(row_parts))
        if len("\n".join(lines)) >= max_chars:
            break
    return "\n".join(lines)[:max_chars]
