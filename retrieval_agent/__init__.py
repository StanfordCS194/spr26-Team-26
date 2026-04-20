"""Spec-driven retrieval agent for external raw data acquisition planning."""

__all__ = ["RetrievalAgent"]


def __getattr__(name: str):
    if name == "RetrievalAgent":
        from .agent import RetrievalAgent

        return RetrievalAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
