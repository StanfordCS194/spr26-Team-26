from __future__ import annotations

from typing import Any


def planner_output_to_acquisition_spec(payload: dict[str, Any]) -> dict[str, Any]:
    task_spec = payload.get("task_spec", {})
    model_spec = payload.get("model_spec", {})
    input_contract = model_spec.get("input_contract", {})
    data_acq = payload.get("data_acquisition_spec", {})

    explicit_sources_raw = payload.get("explicit_sources", [])
    explicit_sources: list[str] = []
    for item in explicit_sources_raw:
        if isinstance(item, str) and item.strip():
            explicit_sources.append(item.strip())
        elif isinstance(item, dict):
            source_url = str(item.get("source_url", "")).strip()
            if source_url:
                explicit_sources.append(source_url)

    queries: list[str] = []
    queries.extend(data_acq.get("broad_queries", []))
    queries.extend(data_acq.get("narrow_queries", []))
    queries.extend(data_acq.get("label_space_terms_to_search", []))

    seen_queries: set[str] = set()
    normalized_queries: list[str] = []
    for query in queries:
        normalized = " ".join(str(query).split()).strip()
        if normalized and normalized.lower() not in seen_queries:
            seen_queries.add(normalized.lower())
            normalized_queries.append(normalized)

    label_space = [str(x) for x in task_spec.get("label_space", [])]
    required_fields = [str(x) for x in input_contract.get("required_fields", [])]

    task_name = str(task_spec.get("task_name") or payload.get("pipeline_name") or "unnamed_task")
    task_type = str(task_spec.get("task_type") or "other")

    example_unit = str(
        task_spec.get("prediction_unit")
        or model_spec.get("train_example_schema", {}).get("example_unit")
        or "one example"
    )

    output_fields = ["label"]
    if task_type == "text_to_text":
        output_fields = ["response"]

    return {
        "task_name": task_name,
        "task_type": task_type,
        "target_schema": {
            "example_unit": example_unit,
            "input_fields": required_fields or ["message_text"],
            "output_fields": output_fields,
            "label_space": label_space,
        },
        "data_requirements": {
            "preferred_sources": [str(x) for x in data_acq.get("preferred_sources", [])],
            "query_keywords": normalized_queries,
            "min_examples": int(data_acq.get("min_raw_examples_target") or 0),
            "languages": [str(x) for x in data_acq.get("languages", [])],
        },
        "constraints": {
            "allow_scraping": bool(data_acq.get("allow_scraping", True)),
            "allow_api_sources": bool(data_acq.get("allow_api_sources", True)),
            "allow_synthetic": bool(data_acq.get("allow_synthetic", False)),
        },
        "explicit_sources": explicit_sources,
    }
