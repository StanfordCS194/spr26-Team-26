from __future__ import annotations

from typing import Any, Mapping

from src.types import RawData


def acquire_web_data(query: str, config: Mapping[str, Any] | None = None) -> RawData:
    """Generate a small synthetic chat/SFT dataset for Mode C.

    The MVP spec makes web scraping a fallback and synthetic generation the
    primary no-data path. This deterministic teacher gives the downstream
    Tinker SFT runner valid user/assistant examples without requiring a live
    LLM call during tests or demos.
    """
    clean_query = " ".join(str(query or "generic task").split())
    task_type = _task_type(config)
    records = _classification_records(clean_query, task_type)
    records.extend(_instruction_records(clean_query, task_type))
    return {
        "records": records,
        "format_meta": {
            "modality": "text",
            "file_type": "synthetic_chat_jsonl",
            "encoding": "utf-8",
        },
    }


def _task_type(config: Mapping[str, Any] | None) -> str:
    if not config:
        return "custom"
    procedure = config.get("training_procedure", {})
    if isinstance(procedure, Mapping):
        return str(procedure.get("task_type") or "custom")
    return "custom"


def _classification_records(query: str, task_type: str) -> list[dict[str, Any]]:
    labels = ("relevant", "not_relevant")
    snippets = [
        (f"This text directly asks for help with {query}.", labels[0]),
        (f"A user wants examples and evaluation criteria for {query}.", labels[0]),
        (f"The request describes inputs, outputs, and constraints for {query}.", labels[0]),
        ("This text is about an unrelated calendar reminder.", labels[1]),
        ("The message discusses lunch plans and does not mention the task.", labels[1]),
        ("A generic weather note with no training objective.", labels[1]),
    ]
    records: list[dict[str, Any]] = []
    for index, (text, label) in enumerate(snippets):
        records.append(
            {
                "source": "mode_c_synthetic",
                "task_type": task_type,
                "synthetic": True,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Classify the text for the requested ML task. "
                            f"Task: {query}\nText: {text}\nLabels: relevant, not_relevant"
                        ),
                    },
                    {"role": "assistant", "content": label},
                ],
                "input": text,
                "output": label,
                "id": f"mode-c-classification-{index}",
            }
        )
    return records


def _instruction_records(query: str, task_type: str) -> list[dict[str, Any]]:
    examples = [
        (
            f"Write a concise training example for this task: {query}",
            f"Input: an example related to {query}\nOutput: relevant",
        ),
        (
            f"Name one likely quality check for data about {query}.",
            "Check that each record has a clear input and a target answer.",
        ),
        (
            f"Summarize the target behavior for a model trained on {query}.",
            "The model should follow the task instruction and produce the expected label or answer.",
        ),
        (
            f"Create a negative example for {query}.",
            "Input: unrelated personal scheduling text\nOutput: not_relevant",
        ),
    ]
    records: list[dict[str, Any]] = []
    for index, (prompt, answer) in enumerate(examples):
        records.append(
            {
                "source": "mode_c_synthetic",
                "task_type": task_type,
                "synthetic": True,
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer},
                ],
                "id": f"mode-c-instruction-{index}",
            }
        )
    return records
