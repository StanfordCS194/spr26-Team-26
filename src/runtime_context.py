"""Run-scoped runtime context for mutable output paths."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from threading import Event
from typing import MutableSet

_OUTPUT_ROOT: ContextVar[Path | None] = ContextVar("output_root", default=None)
_CANCEL_EVENT: ContextVar[Event | None] = ContextVar("cancel_event", default=None)
_ACTIVE_TINKER_JOBS: ContextVar[MutableSet[str] | None] = ContextVar(
    "active_tinker_jobs",
    default=None,
)


class RunCancelled(RuntimeError):
    """Raised when a run has been cancelled by its caller."""


def get_output_root() -> Path | None:
    """Return the current run-scoped output root, if one is active."""
    return _OUTPUT_ROOT.get()


def cancellation_requested() -> bool:
    """Return True when the active run has been asked to stop."""
    event = _CANCEL_EVENT.get()
    return bool(event and event.is_set())


def raise_if_cancelled() -> None:
    """Abort cooperatively when the active run has been cancelled."""
    if cancellation_requested():
        raise RunCancelled("Run cancelled by user")


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


@contextmanager
def cancellation_context(
    cancel_event: Event | None,
    active_tinker_jobs: MutableSet[str] | None = None,
) -> Iterator[None]:
    """Make a cancellation signal and active Tinker registry visible in this context."""
    event_token = _CANCEL_EVENT.set(cancel_event)
    jobs_token = _ACTIVE_TINKER_JOBS.set(active_tinker_jobs)
    try:
        yield
    finally:
        _ACTIVE_TINKER_JOBS.reset(jobs_token)
        _CANCEL_EVENT.reset(event_token)


@contextmanager
def active_tinker_job(job_id: str) -> Iterator[None]:
    """Register a Tinker job while it is running so callers can cancel it."""
    jobs = _ACTIVE_TINKER_JOBS.get()
    if jobs is not None:
        jobs.add(job_id)
    try:
        yield
    finally:
        if jobs is not None:
            jobs.discard(job_id)
