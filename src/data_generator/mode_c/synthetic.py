"""Synthetic chat/SFT data generation for DataGen Mode C."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence, TypedDict

from src.data_generator.mode_c.offline import mode_c_offline
from src.types import DataSchema, OrchestrationConfig, RawData, ValidationReport

DEFAULT_TEACHER_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_SYNTHETIC_EXAMPLES = 60
DEFAULT_BATCH_SIZE = 12
MIN_SYNTHETIC_EXAMPLES = 8


class SyntheticBatchPlan(TypedDict):
    name: str
    goal: str
    count: int
    difficulty: str
    label_hint: str


class SyntheticQualityReport(TypedDict):
    passed: bool
    issues: list[str]
    total_records: int
    valid_records: int
    duplicate_records: int
    label_counts: dict[str, int]


@dataclass(frozen=True)
class SyntheticGenerationResult:
    schema: DataSchema
    raw_data: RawData
    validation_report: ValidationReport
    quality_report: SyntheticQualityReport


def determine_data_schema(
    config: OrchestrationConfig | Mapping[str, Any],
    *,
    teacher_client: Any = None,
    teacher_model: str = DEFAULT_TEACHER_MODEL,
) -> DataSchema:
    """Infer the input/output schema for synthetic generation.

    Uses a Claude-style teacher client when one is supplied or when
    ``ANTHROPIC_API_KEY`` is available. Falls back to deterministic schema
    inference so local tests and demos remain runnable.
    """
    if _should_use_teacher(teacher_client):
        try:
            client = teacher_client or _anthropic_client()
            message = client.messages.create(
                model=teacher_model,
                max_tokens=900,
                temperature=0,
                system=(
                    "You design supervised fine-tuning datasets. Return only "
                    "valid JSON with no markdown fences."
                ),
                messages=[{"role": "user", "content": _schema_prompt(config)}],
            )
            parsed = _parse_json_object(_message_text(message))
            return _coerce_schema(parsed, config)
        except Exception:
            if os.getenv("DATA_GENERATOR_SYNTHETIC_STRICT") == "1":
                raise

    return infer_schema_without_teacher(config)


def infer_schema_without_teacher(config: OrchestrationConfig | Mapping[str, Any]) -> DataSchema:
    """Deterministic schema inference for tests and offline demos."""
    prompt = _task_prompt(config)
    task_type = _task_type(config)
    data_format = _data_format(config)

    if "classification" in task_type:
        output_format = "one of: relevant, not_relevant"
        output_description = (
            "A short class label indicating whether the input matches the task."
        )
        example_output = "relevant"
    elif task_type in {"seq2seq", "summarization", "translation"}:
        output_format = "natural language answer"
        output_description = "A concise target response for the requested transformation."
        example_output = f"A concise response for: {prompt}"
    else:
        output_format = "task-appropriate natural language answer"
        output_description = "The expected answer or label for the task."
        example_output = f"Answer related to: {prompt}"

    return {
        "input_format": data_format or "chat instruction or text input",
        "output_format": output_format,
        "input_description": f"User input for the task: {prompt}",
        "output_description": output_description,
        "example_pair": {
            "input": f"Example input about {prompt}",
            "output": example_output,
        },
    }


def plan_synthetic_generation(
    config: OrchestrationConfig | Mapping[str, Any],
    schema: DataSchema,
    n_examples: int,
) -> list[SyntheticBatchPlan]:
    """Create a coverage plan for the synthetic examples."""
    total = max(MIN_SYNTHETIC_EXAMPLES, int(n_examples))
    buckets = [
        (
            "core_positive",
            "Canonical in-distribution examples that should satisfy the task.",
            "easy",
            "relevant",
        ),
        (
            "core_negative",
            "Clearly unrelated examples that should be rejected or labeled negative.",
            "easy",
            "not_relevant",
        ),
        (
            "boundary_cases",
            "Ambiguous or partial matches that force the model to attend to details.",
            "medium",
            "mixed",
        ),
        (
            "format_variants",
            "Inputs with varied phrasing, length, and surface form.",
            "medium",
            "mixed",
        ),
        (
            "hard_edge_cases",
            "Adversarial, noisy, or underspecified examples that are still answerable.",
            "hard",
            "mixed",
        ),
    ]
    base = total // len(buckets)
    remainder = total % len(buckets)
    plan: list[SyntheticBatchPlan] = []
    for index, (name, goal, difficulty, label_hint) in enumerate(buckets):
        count = base + (1 if index < remainder else 0)
        if count <= 0:
            continue
        plan.append(
            {
                "name": name,
                "goal": goal,
                "count": count,
                "difficulty": difficulty,
                "label_hint": label_hint,
            }
        )
    return plan


def generate_synthetic_data(
    schema: DataSchema,
    n_examples: int,
    *,
    config: OrchestrationConfig | Mapping[str, Any] | None = None,
    teacher_client: Any = None,
    teacher_model: str = DEFAULT_TEACHER_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> RawData:
    """Generate synthetic chat/SFT examples according to a schema."""
    target = max(MIN_SYNTHETIC_EXAMPLES, int(n_examples))
    plan = plan_synthetic_generation(config or {}, schema, target)
    records: list[dict[str, Any]] = []
    teacher_used = False
    teacher_available = _should_use_teacher(teacher_client)
    client = None
    if teacher_available:
        try:
            client = teacher_client or _anthropic_client()
        except Exception:
            if os.getenv("DATA_GENERATOR_SYNTHETIC_STRICT") == "1":
                raise

    for bucket in plan:
        remaining = bucket["count"]
        while remaining > 0:
            count = min(max(1, int(batch_size)), remaining)
            batch_records: list[dict[str, Any]]
            if client is not None:
                try:
                    batch_records = _generate_batch_with_teacher(
                        schema,
                        bucket,
                        count,
                        config=config or {},
                        client=client,
                        teacher_model=teacher_model,
                    )
                    teacher_used = teacher_used or bool(batch_records)
                except Exception:
                    if os.getenv("DATA_GENERATOR_SYNTHETIC_STRICT") == "1":
                        raise
                    batch_records = []
            else:
                batch_records = []

            if len(batch_records) < count:
                batch_records.extend(
                    _deterministic_records(
                        schema,
                        bucket,
                        count - len(batch_records),
                        config=config or {},
                        offset=len(records),
                    )
                )
            records.extend(batch_records)
            remaining -= count

    standardized = _standardize_records(records, schema, target)
    quality = validate_synthetic_records(standardized, schema, min_examples=target)
    return {
        "records": standardized,
        "format_meta": {
            "modality": "text",
            "file_type": "synthetic_chat_jsonl",
            "encoding": "utf-8",
            "schema": schema,
            "generation_plan": plan,
            "teacher_model": teacher_model,
            "teacher_available": teacher_available,
            "teacher_used": teacher_used,
            "quality_report": quality,
        },
    }


def validate_synthetic_records(
    records: Sequence[Mapping[str, Any]],
    schema: DataSchema,
    *,
    min_examples: int = MIN_SYNTHETIC_EXAMPLES,
) -> SyntheticQualityReport:
    """Validate synthetic records before they are handed to curation/training."""
    issues: list[str] = []
    valid = 0
    duplicate_count = 0
    seen: set[str] = set()
    label_counts: dict[str, int] = {}
    allowed_labels = set(_schema_labels(schema))

    for index, record in enumerate(records):
        try:
            normalized = _coerce_generated_record(record, schema)
        except ValueError as exc:
            issues.append(f"row {index}: {exc}")
            continue
        key = _record_key(normalized)
        if key in seen:
            duplicate_count += 1
            issues.append(f"row {index}: duplicate input/output pair")
            continue
        seen.add(key)
        label = _assistant_content(normalized)
        if allowed_labels and label not in allowed_labels:
            issues.append(
                f"row {index}: label {label!r} is not in schema labels "
                f"{sorted(allowed_labels)!r}"
            )
            continue
        valid += 1
        label_counts[label] = label_counts.get(label, 0) + 1

    if valid < min_examples:
        issues.append(f"Only {valid} valid records; expected at least {min_examples}")
    if len(label_counts) < 2 and "one of:" in schema.get("output_format", ""):
        issues.append("Classification-style data should include at least two labels")

    return {
        "passed": not issues,
        "issues": issues,
        "total_records": len(records),
        "valid_records": valid,
        "duplicate_records": duplicate_count,
        "label_counts": label_counts,
    }


def synthetic_quality_to_validation_report(
    quality: SyntheticQualityReport,
) -> ValidationReport:
    """Map Mode C quality details into the shared ValidationReport shape."""
    if quality["total_records"] == 0:
        estimate = 0.0
    else:
        estimate = round(quality["valid_records"] / quality["total_records"], 3)
    return {
        "passed": quality["passed"],
        "issues": quality["issues"],
        "sample_accuracy_estimate": estimate,
    }


def morph_to_standard(raw: RawData, schema: DataSchema) -> RawData:
    """Normalize generated/scraped data into chat/SFT records."""
    standardized = _standardize_records(raw.get("records", []), schema, len(raw.get("records", [])))
    quality = validate_synthetic_records(
        standardized,
        schema,
        min_examples=min(MIN_SYNTHETIC_EXAMPLES, max(1, len(standardized))),
    )
    format_meta = dict(raw.get("format_meta", {}))
    format_meta["quality_report"] = quality
    return {"records": standardized, "format_meta": format_meta}


def build_mode_c_dataset(
    config: OrchestrationConfig | Mapping[str, Any],
    *,
    teacher_client: Any = None,
    teacher_model: str = DEFAULT_TEACHER_MODEL,
    n_examples: int | None = None,
) -> SyntheticGenerationResult:
    """Run the full Mode C schema -> generate -> validate pipeline."""
    schema = determine_data_schema(
        config,
        teacher_client=teacher_client,
        teacher_model=teacher_model,
    )
    target = n_examples if n_examples is not None else _target_examples(config)
    raw = generate_synthetic_data(
        schema,
        target,
        config=config,
        teacher_client=teacher_client,
        teacher_model=teacher_model,
    )
    standardized = morph_to_standard(raw, schema)
    quality = standardized["format_meta"]["quality_report"]
    validation = synthetic_quality_to_validation_report(quality)
    return SyntheticGenerationResult(
        schema=schema,
        raw_data=standardized,
        validation_report=validation,
        quality_report=quality,
    )


def scrape_web(query: str, schema: DataSchema, max_examples: int = 500) -> RawData:
    """Scrape/adapt web sources into the same chat/SFT contract.

    If the Mode C web-acquisition modules from PR #27 are present, this uses
    them as the source collector and adapts extracted pages into trainable
    records. Otherwise it returns deterministic excerpt-style records so tests
    and demos can run without network credentials.
    """
    acquired = _scrape_with_mode_c_web_pipeline(query, schema, max_examples)
    if acquired is not None:
        return acquired

    config = {
        "prompt": query,
        "training_procedure": {
            "task_type": "custom",
            "data_format": schema["input_format"],
            "training_type": "SFT",
            "base_model": None,
            "hyperparameters": {"synthetic_examples": max_examples},
            "notes": "synthetic fallback for unavailable web acquisition",
        },
    }
    raw = generate_synthetic_data(
        schema,
        min(max_examples, DEFAULT_SYNTHETIC_EXAMPLES),
        config=config,
    )
    raw["format_meta"]["web_acquisition_used"] = False
    raw["format_meta"]["web_acquisition_fallback"] = "deterministic_synthetic"
    return raw


def _scrape_with_mode_c_web_pipeline(
    query: str,
    schema: DataSchema,
    max_examples: int,
) -> RawData | None:
    if mode_c_offline():
        return None

    try:
        from src.data_generator.mode_c.crawler import crawl_and_extract_pages
        from src.data_generator.mode_c.mock_llm import mock_plan_web_acquisition
        from src.data_generator.mode_c.search import search_web_sources
    except Exception:
        return None

    config = {
        "data": False,
        "prompt": query,
        "compute_budget": 0.0,
        "training_procedure": {
            "task_type": "custom",
            "data_format": schema["input_format"],
            "training_type": "SFT",
            "base_model": None,
            "hyperparameters": {"synthetic_examples": max_examples},
            "notes": "web acquisition adapter",
        },
    }
    try:
        web_plan = mock_plan_web_acquisition(config)
        web_plan["max_pages"] = min(
            int(web_plan.get("max_pages", max_examples)),
            max(1, max_examples),
        )
        search_results = search_web_sources(web_plan)
        pages = crawl_and_extract_pages(search_results, web_plan)
    except Exception:
        if os.getenv("DATA_GENERATOR_SYNTHETIC_STRICT") == "1":
            raise
        return None

    records = _web_pages_to_sft_records(pages, schema, query, max_examples)
    if not records:
        return None
    quality = validate_synthetic_records(
        records,
        schema,
        min_examples=min(MIN_SYNTHETIC_EXAMPLES, len(records)),
    )
    return {
        "records": records,
        "format_meta": {
            "modality": "text",
            "file_type": "web_acquired_chat_jsonl",
            "encoding": "utf-8",
            "schema": schema,
            "web_acquisition_used": True,
            "web_plan": web_plan,
            "num_search_results": len(search_results),
            "num_pages_crawled": len(pages),
            "quality_report": quality,
        },
    }


def _web_pages_to_sft_records(
    pages: Sequence[Mapping[str, Any]],
    schema: DataSchema,
    query: str,
    max_examples: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    labels = _schema_labels(schema)
    positive_label = labels[0] if labels else ""
    for page in pages:
        if len(records) >= max_examples:
            break
        content = _first_text(page, ("content", "snippet", "title"))
        if not content:
            continue
        excerpt = content[:1800]
        title = _first_text(page, ("title", "url", "domain")) or "web source"
        url = _first_text(page, ("url", "source_path", "path"))
        input_text = (
            f"Task: {query}\n"
            f"Source: {title}\n"
            f"URL: {url or 'unknown'}\n"
            f"Excerpt: {excerpt}"
        )
        if "one of:" in schema.get("output_format", "") and positive_label:
            output_text = positive_label
        else:
            output_text = (
                "Use this source as relevant evidence for the requested task; "
                f"it discusses: {excerpt[:280]}"
            )
        records.append(
            {
                "source": "mode_c_web_acquisition",
                "synthetic": False,
                "task_type": "custom",
                "bucket": "web_acquired",
                "difficulty": "medium",
                "input": input_text,
                "output": output_text,
                "messages": [
                    {"role": "user", "content": input_text},
                    {"role": "assistant", "content": output_text},
                ],
                "url": url,
                "metadata": page.get("metadata", {}),
            }
        )
    return _standardize_records(records, schema, max_examples)


def _generate_batch_with_teacher(
    schema: DataSchema,
    bucket: SyntheticBatchPlan,
    count: int,
    *,
    config: Mapping[str, Any],
    client: Any,
    teacher_model: str,
) -> list[dict[str, Any]]:
    message = client.messages.create(
        model=teacher_model,
        max_tokens=max(1200, count * 220),
        temperature=0.7,
        system=(
            "You are a careful synthetic data generation agent. Return only "
            "valid JSON arrays with no markdown fences."
        ),
        messages=[
            {
                "role": "user",
                "content": _batch_prompt(schema, bucket, count, config),
            }
        ],
    )
    payload = _parse_json_array(_message_text(message))
    records = []
    for item in payload:
        if isinstance(item, Mapping):
            try:
                records.append(
                    _coerce_generated_record(
                        {
                            **item,
                            "source": "mode_c_teacher",
                            "bucket": bucket["name"],
                            "difficulty": bucket["difficulty"],
                        },
                        schema,
                    )
                )
            except ValueError:
                continue
    return records


def _deterministic_records(
    schema: DataSchema,
    bucket: SyntheticBatchPlan,
    count: int,
    *,
    config: Mapping[str, Any],
    offset: int = 0,
) -> list[dict[str, Any]]:
    prompt = _task_prompt(config)
    task_type = _task_type(config)
    labels = _schema_labels(schema)
    if not labels:
        labels = ["relevant", "not_relevant"]

    records: list[dict[str, Any]] = []
    templates = _bucket_templates(bucket["name"])
    for index in range(count):
        template = templates[(offset + index) % len(templates)]
        label = _label_for_bucket(bucket, labels, offset + index)
        scenario_id = offset + index + 1
        input_text = (
            f"Scenario {scenario_id}: "
            f"{template.format(task=prompt, index=scenario_id)}"
        )
        output_text = _output_for_label(schema, label, input_text, prompt)
        records.append(
            {
                "source": "mode_c_deterministic_teacher",
                "synthetic": True,
                "task_type": task_type,
                "bucket": bucket["name"],
                "difficulty": bucket["difficulty"],
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"{schema['input_description']}\n\n"
                            f"Input: {input_text}"
                        ),
                    },
                    {"role": "assistant", "content": output_text},
                ],
                "input": input_text,
                "output": output_text,
            }
        )
    return records


def _standardize_records(
    records: Sequence[Mapping[str, Any]],
    schema: DataSchema,
    target: int,
) -> list[dict[str, Any]]:
    standardized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        try:
            normalized = _coerce_generated_record(record, schema)
        except ValueError:
            continue
        key = _record_key(normalized)
        if key in seen:
            continue
        seen.add(key)
        normalized["id"] = normalized.get("id") or f"mode-c-synthetic-{len(standardized):05d}"
        standardized.append(normalized)
        if len(standardized) >= target:
            break
    return standardized


def _coerce_generated_record(record: Mapping[str, Any], schema: DataSchema) -> dict[str, Any]:
    messages = record.get("messages")
    if isinstance(messages, list) and messages:
        normalized_messages = _normalize_messages(messages)
        if not any(message["role"] == "user" for message in normalized_messages):
            raise ValueError("messages must include a user message")
        if not any(message["role"] == "assistant" for message in normalized_messages):
            raise ValueError("messages must include an assistant message")
    else:
        input_text = _first_text(record, ("input", "prompt", "question", "content", "text"))
        output_text = _first_text(record, ("output", "answer", "label", "target", "label_text"))
        if not input_text:
            raise ValueError("missing input text")
        if not output_text:
            raise ValueError("missing output text")
        normalized_messages = [
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": output_text},
        ]

    input_text = _first_text(record, ("input", "prompt", "question", "content", "text"))
    if not input_text:
        input_text = next(
            message["content"] for message in normalized_messages if message["role"] == "user"
        )
    output_text = _first_text(record, ("output", "answer", "label", "target", "label_text"))
    if not output_text:
        output_text = _assistant_content({"messages": normalized_messages})

    return {
        "id": str(record.get("id", "")).strip() or None,
        "source": str(record.get("source", "mode_c_synthetic")),
        "synthetic": True,
        "task_type": str(record.get("task_type", "")) or "custom",
        "bucket": str(record.get("bucket", "")) or "unbucketed",
        "difficulty": str(record.get("difficulty", "")) or "medium",
        "input": input_text,
        "output": output_text,
        "messages": normalized_messages,
        "schema": {
            "input_format": schema["input_format"],
            "output_format": schema["output_format"],
        },
    }


def _normalize_messages(messages: Sequence[Any]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise ValueError("message entries must be objects")
        role = str(message.get("role", "")).strip()
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"unsupported message role {role!r}")
        if content is None or str(content).strip() == "":
            raise ValueError("message content cannot be empty")
        normalized.append({"role": role, "content": str(content).strip()})
    return normalized


def _schema_prompt(config: Mapping[str, Any]) -> str:
    return f"""Infer a supervised fine-tuning data schema for this ML task.

Orchestration config:
{json.dumps(config, indent=2)}

Return this JSON object:
{{
  "input_format": "short description of the user/input field",
  "output_format": "short description of the assistant/target field",
  "input_description": "what a useful input example should contain",
  "output_description": "what the target answer should contain",
  "example_pair": {{"input": "example input", "output": "example output"}}
}}"""


def _batch_prompt(
    schema: DataSchema,
    bucket: SyntheticBatchPlan,
    count: int,
    config: Mapping[str, Any],
) -> str:
    return f"""Generate {count} supervised fine-tuning examples.

Task prompt:
{_task_prompt(config)}

Schema:
{json.dumps(schema, indent=2)}

Coverage bucket:
{json.dumps(bucket, indent=2)}

Rules:
- Return a JSON array of exactly {count} objects.
- Each object must have "input", "output", "messages", "difficulty", and "tags".
- "messages" must contain one user message and one assistant message.
- Keep examples diverse and avoid duplicates.
- Do not include unsafe private data, real personal information, or markdown fences.

Example object:
{{
  "input": "task input text",
  "output": "target answer or label",
  "messages": [
    {{"role": "user", "content": "task input text"}},
    {{"role": "assistant", "content": "target answer or label"}}
  ],
  "difficulty": "{bucket['difficulty']}",
  "tags": ["{bucket['name']}"]
}}"""


def _parse_json_object(text: str) -> dict[str, Any]:
    parsed = json.loads(_strip_fences(text))
    if not isinstance(parsed, dict):
        raise ValueError("teacher response must be a JSON object")
    return parsed


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


def _coerce_schema(parsed: Mapping[str, Any], config: Mapping[str, Any]) -> DataSchema:
    fallback = infer_schema_without_teacher(config)
    example_pair = parsed.get("example_pair")
    if not isinstance(example_pair, Mapping):
        example_pair = fallback["example_pair"]
    return {
        "input_format": _field_or_fallback(parsed, "input_format", fallback),
        "output_format": _field_or_fallback(parsed, "output_format", fallback),
        "input_description": _field_or_fallback(parsed, "input_description", fallback),
        "output_description": _field_or_fallback(parsed, "output_description", fallback),
        "example_pair": {
            "input": str(example_pair.get("input") or fallback["example_pair"]["input"]),
            "output": str(example_pair.get("output") or fallback["example_pair"]["output"]),
        },
    }


def _field_or_fallback(
    parsed: Mapping[str, Any],
    key: Literal["input_format", "output_format", "input_description", "output_description"],
    fallback: DataSchema,
) -> str:
    value = parsed.get(key)
    if value is None or str(value).strip() == "":
        return fallback[key]
    return str(value).strip()


def _target_examples(config: Mapping[str, Any]) -> int:
    env_value = os.getenv("DATA_GENERATOR_SYNTHETIC_EXAMPLES")
    if env_value:
        try:
            return max(MIN_SYNTHETIC_EXAMPLES, int(env_value))
        except ValueError:
            pass
    procedure = config.get("training_procedure", {})
    if isinstance(procedure, Mapping):
        hyperparams = procedure.get("hyperparameters", {})
        if isinstance(hyperparams, Mapping):
            for key in ("synthetic_examples", "n_examples", "num_examples"):
                value = hyperparams.get(key)
                if value is not None:
                    try:
                        return max(MIN_SYNTHETIC_EXAMPLES, int(value))
                    except (TypeError, ValueError):
                        pass
    return DEFAULT_SYNTHETIC_EXAMPLES


def _should_use_teacher(teacher_client: Any) -> bool:
    if mode_c_offline():
        return False
    return teacher_client is not None or bool(os.getenv("ANTHROPIC_API_KEY"))


def _anthropic_client() -> Any:
    import anthropic

    return anthropic.Anthropic()


def _task_prompt(config: Mapping[str, Any] | None) -> str:
    if not config:
        return "generic ML task"
    return " ".join(str(config.get("prompt", "generic ML task")).split()) or "generic ML task"


def _task_type(config: Mapping[str, Any] | None) -> str:
    if not config:
        return "custom"
    procedure = config.get("training_procedure", {})
    if isinstance(procedure, Mapping):
        return str(procedure.get("task_type") or "custom")
    return "custom"


def _data_format(config: Mapping[str, Any]) -> str:
    procedure = config.get("training_procedure", {})
    if isinstance(procedure, Mapping):
        return str(procedure.get("data_format") or "chat JSONL")
    return "chat JSONL"


def _schema_labels(schema: DataSchema) -> list[str]:
    output_format = schema.get("output_format", "")
    if "one of:" not in output_format:
        return []
    label_text = output_format.split("one of:", 1)[1]
    labels = [part.strip(" .") for part in re.split(r"[,|/]", label_text)]
    return [label for label in labels if label]


def _bucket_templates(bucket_name: str) -> list[str]:
    templates_by_bucket = {
        "core_positive": [
            "A user request directly asks for {task} and includes all required context.",
            "Example {index}: a clean, in-domain input for {task}.",
            "A concise support-style message that clearly matches {task}.",
        ],
        "core_negative": [
            "An unrelated note about travel plans with no connection to {task}.",
            "Example {index}: a casual message about lunch, not the ML task.",
            "A generic weather update that should not match {task}.",
        ],
        "boundary_cases": [
            "A partial request that mentions {task} but omits one important detail.",
            "Example {index}: an ambiguous input that could be related to {task}.",
            "A noisy sentence containing both relevant and irrelevant details for {task}.",
        ],
        "format_variants": [
            "JSON-ish text: {{'request': '{task}', 'priority': 'unknown'}}.",
            "Bullet list input for {task}: first clue; second clue; missing target.",
            "Long-form paragraph describing a scenario where {task} is needed.",
        ],
        "hard_edge_cases": [
            "A contradictory input that asks for {task} but supplies mismatched evidence.",
            "Example {index}: an adversarially phrased request around {task}.",
            "An underspecified input where the safest answer should acknowledge uncertainty.",
        ],
    }
    return templates_by_bucket.get(bucket_name, templates_by_bucket["boundary_cases"])


def _label_for_bucket(
    bucket: SyntheticBatchPlan,
    labels: Sequence[str],
    index: int,
) -> str:
    hint = bucket["label_hint"]
    if hint in labels:
        return hint
    if hint == "not_relevant" and "not_relevant" in labels:
        return "not_relevant"
    if hint == "relevant" and "relevant" in labels:
        return "relevant"
    return labels[index % len(labels)]


def _output_for_label(
    schema: DataSchema,
    label: str,
    input_text: str,
    task: str,
) -> str:
    if "one of:" in schema.get("output_format", ""):
        return label
    return (
        f"For the task '{task}', respond to this input with a concise, correct "
        f"answer: {input_text}"
    )


def _first_text(record: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _assistant_content(record: Mapping[str, Any]) -> str:
    messages = record.get("messages", [])
    if isinstance(messages, Sequence):
        for message in reversed(messages):
            if isinstance(message, Mapping) and message.get("role") == "assistant":
                content = message.get("content")
                if content is not None:
                    return str(content).strip()
    return _first_text(record, ("output", "answer", "label", "target", "label_text"))


def _record_key(record: Mapping[str, Any]) -> str:
    return json.dumps(
        [
            _first_text(record, ("input", "prompt", "question", "content", "text")).lower(),
            _assistant_content(record).lower(),
        ],
        sort_keys=True,
    )
