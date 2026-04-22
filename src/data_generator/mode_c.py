from __future__ import annotations

from src.types import RawData


def acquire_web_data(query: str) -> RawData:
    """
    Mode C backbone placeholder.
    Real internet retrieval/relevance logic will be implemented later.
    """
    return {
        "records": [
            {
                "source": "mode_c_placeholder",
                "content": f"TODO: retrieve relevant web data for query: {query}",
            }
        ],
        "format_meta": {"modality": "text", "file_type": "web_placeholder", "encoding": "utf-8"},
    }
