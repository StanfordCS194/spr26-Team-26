from __future__ import annotations

import json
import os
from typing import Any

try:
    from langchain_ollama import ChatOllama
except Exception:  # pragma: no cover
    ChatOllama = None


PLANNER_PROMPT_TEMPLATE = """
You are a pipeline planning agent.
Output a single JSON object matching the expected rich planning format:
- task_spec
- model_spec
- data_acquisition_spec
- explicit_sources
- any supporting planning sections

Return only valid JSON.

User request:
{user_request}
""".strip()


def generate_planner_output_with_llm(user_request: str) -> dict[str, Any]:
    if ChatOllama is None:
        raise RuntimeError("langchain_ollama is not available in this environment.")

    model = os.environ.get("RETRIEVAL_PLANNER_MODEL", "qwen2.5:7b-instruct")
    api_base = os.environ.get("RETRIEVAL_PLANNER_API_BASE", "http://localhost:11434")
    llm = ChatOllama(model=model, base_url=api_base, temperature=0.0)

    prompt = PLANNER_PROMPT_TEMPLATE.format(user_request=user_request)
    response = llm.invoke(prompt)
    content = getattr(response, "content", response)
    if not isinstance(content, str):
        content = str(content)

    try:
        return json.loads(content)
    except Exception as exc:
        raise RuntimeError(f"Planner LLM returned non-JSON output: {exc}") from exc
