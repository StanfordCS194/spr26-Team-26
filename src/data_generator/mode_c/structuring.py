"""Teacher-backed structuring for Mode C web-acquired sources."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.data_generator.mode_c.synthetic import (
    DEFAULT_TEACHER_MODEL,
    determine_data_schema,
    morph_to_standard,
    synthetic_quality_to_validation_report,
)
from src.types import DataSchema, OrchestrationConfig, RawData, ValidationReport


@dataclass(frozen=True)
class WebStructuringResult:
    schema: DataSchema
    raw_data: RawData
    validation_report: ValidationReport
    teacher_used: bool


def structure_web_sources_for_sft(
    config: OrchestrationConfig | Mapping[str, Any],
    pages: Sequence[Mapping[str, Any]],
    *,
    teacher_client: Any = None,
    teacher_model: str = DEFAULT_TEACHER_MODEL,
    max_records: int = 24,
) -> WebStructuringResult:
    """Turn raw web pages into trainable chat/SFT records when a teacher exists.

    Raw crawled pages are source material, not supervised data. This stage asks a
    teacher model to emit concrete input/output examples grounded in the source
    excerpts. If no teacher is available, the result is intentionally invalid so
    downstream curation will not train on targetless web records.
    """
    schema = determine_data_schema(
        config,
        teacher_client=teacher_client,
        teacher_model=teacher_model,
    )
    source_pages = [page for page in pages if _page_excerpt(page)]
    if not source_pages:
        return _unstructured_result(schema, pages, "No usable web page excerpts to structure.")

    client = teacher_client or _optional_anthropic_client()
    if client is None:
        return _unstructured_result(
            schema,
            pages,
            "Mode C web structuring requires a teacher client or ANTHROPIC_API_KEY.",
        )

    try:
        message = client.messages.create(
            model=teacher_model,
            max_tokens=max(1600, min(max_records, len(source_pages) * 3) * 320),
            temperature=0.4,
            system=(
                "You convert acquired public web source excerpts into supervised "
                "fine-tuning examples. Return only valid JSON arrays."
            ),
            messages=[
                {
                    "role": "user",
                    "content": _structuring_prompt(config, schema, source_pages, max_records),
                }
            ],
        )
        records = _coerce_teacher_records(_parse_json_array(_message_text(message)))
    except Exception as exc:
        if os.getenv("DATA_GENERATOR_WEB_STRUCTURING_STRICT") == "1":
            raise
        return _unstructured_result(schema, pages, f"Teacher web structuring failed: {exc}")

    raw = morph_to_standard(
        {
            "records": records,
            "format_meta": {
                "modality": "text",
                "file_type": "web_structured_chat_jsonl",
                "encoding": "utf-8",
                "schema": schema,
                "teacher_model": teacher_model,
                "teacher_used": True,
                "source_record_count": len(source_pages),
                "requested_records": max_records,
            },
        },
        schema,
    )
    quality = raw["format_meta"]["quality_report"]
    validation = synthetic_quality_to_validation_report(quality)
    return WebStructuringResult(
        schema=schema,
        raw_data=raw,
        validation_report=validation,
        teacher_used=True,
    )


def _unstructured_result(
    schema: DataSchema,
    pages: Sequence[Mapping[str, Any]],
    issue: str,
) -> WebStructuringResult:
    return WebStructuringResult(
        schema=schema,
        raw_data={
            "records": [],
            "format_meta": {
                "modality": "text",
                "file_type": "web_structured_chat_jsonl",
                "encoding": "utf-8",
                "schema": schema,
                "teacher_used": False,
                "source_record_count": len(pages),
                "quality_report": {
                    "passed": False,
                    "issues": [issue],
                    "total_records": 0,
                    "valid_records": 0,
                    "duplicate_records": 0,
                    "label_counts": {},
                },
            },
        },
        validation_report={
            "passed": False,
            "issues": [issue],
            "sample_accuracy_estimate": 0.0,
        },
        teacher_used=False,
    )


def _structuring_prompt(
    config: Mapping[str, Any],
    schema: DataSchema,
    pages: Sequence[Mapping[str, Any]],
    max_records: int,
) -> str:
    sources = []
    for index, page in enumerate(pages[:8], start=1):
        sources.append(
            {
                "source_id": f"source_{index}",
                "url": str(page.get("url") or ""),
                "title": str(page.get("title") or ""),
                "excerpt": _page_excerpt(page)[:1800],
            }
        )

    return f"""Structure acquired web sources into supervised fine-tuning data.

Task:
{_task_prompt(config)}

Target schema:
{json.dumps(schema, indent=2)}

Source excerpts:
{json.dumps(sources, indent=2)}

Rules:
- Return a JSON array of at most {max_records} objects.
- Each object must include "input", "output", and "messages".
- "messages" must contain a user message and an assistant message.
- The assistant target must satisfy the target schema exactly.
- Use only information supported by the provided source excerpts.
- Include "source_url" when a source URL supports the example.
- If the sources are not useful for the task, return [].

Example object:
{{
  "input": "task input grounded in a source excerpt",
  "output": "schema-compatible answer or label",
  "messages": [
    {{"role": "user", "content": "task input grounded in a source excerpt"}},
    {{"role": "assistant", "content": "schema-compatible answer or label"}}
  ],
  "source_url": "https://example.com/source"
}}"""


def _coerce_teacher_records(records: Sequence[Any]) -> list[dict[str, Any]]:
    coerced: list[dict[str, Any]] = []
    for item in records:
        if not isinstance(item, Mapping):
            continue
        record = dict(item)
        source_url = str(record.get("source_url") or record.get("url") or "").strip()
        metadata = record.get("metadata") if isinstance(record.get("metadata"), Mapping) else {}
        metadata = dict(metadata)
        if source_url:
            metadata["source_url"] = source_url
            record["url"] = source_url
        record["source"] = "mode_c_web_structuring"
        record["synthetic"] = True
        record["metadata"] = metadata
        coerced.append(record)
    return coerced


def _optional_anthropic_client() -> Any | None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
    except Exception:
        return None
    return anthropic.Anthropic()


def _parse_json_array(text: str) -> list[Any]:
    parsed = json.loads(_strip_fences(text))
    if not isinstance(parsed, list):
        raise ValueError("teacher response must be a JSON array")
    return parsed


def _strip_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = [
        line for line in stripped.splitlines()
        if not line.strip().startswith("```")
    ]
    return "\n".join(lines).strip()


def _message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence):
        parts: list[str] = []
        for item in content:
            text = getattr(item, "text", None)
            if text is not None:
                parts.append(str(text))
        return "\n".join(parts)
    return str(content)


def _page_excerpt(page: Mapping[str, Any]) -> str:
    for key in ("content", "snippet", "title"):
        value = page.get(key)
        if value is not None and str(value).strip():
            return " ".join(str(value).split())
    return ""


def _task_prompt(config: Mapping[str, Any]) -> str:
    return " ".join(str(config.get("prompt") or "the requested ML task").split())
