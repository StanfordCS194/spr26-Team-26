"""Run-scoped runtime context for mutable output paths."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

_OUTPUT_ROOT: ContextVar[Path | None] = ContextVar("output_root", default=None)


def get_output_root() -> Path | None:
    """Return the current run-scoped output root, if one is active."""
    return _OUTPUT_ROOT.get()


def resolve_output_path(default: str | Path, *run_parts: str) -> Path:
    """Resolve a path under the active run root, otherwise return ``default``."""
    root = get_output_root()
    if root is None:
        return Path(default)
    return root.joinpath(*run_parts)


@contextmanager
def output_root(path: str | Path) -> Iterator[Path]:
    """Temporarily route mutable outputs under ``path`` for this context."""
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    token = _OUTPUT_ROOT.set(root)
    try:
        yield root
    finally:
        _OUTPUT_ROOT.reset(token)
