"""
Tinker API Wrapper
Owner: Sid Potti

Thin HTTP client around Tinker's job submission and billing REST APIs.
Raises TinkerAPIError on non-2xx. Retries up to 3× with exponential backoff.

Auth: set TINKER_API_KEY and TINKER_API_BASE environment variables.
"""

from __future__ import annotations

from src.types import JobConfig, JobSummary


class TinkerAPIError(Exception):
    def __init__(self, status_code: int, message: str):
        super().__init__(f"Tinker API error {status_code}: {message}")
        self.status_code = status_code


def submit_job(script_path: str, job_config: JobConfig) -> str:
    """Submits a training script to Tinker for execution on a GPU instance. Returns the Tinker job ID."""
    raise NotImplementedError


def get_job_status(job_id: str) -> str:
    """Fetches the current status of a Tinker job (JobStatus string)."""
    raise NotImplementedError


def get_cumulative_spend(job_id: str) -> float:
    """Returns cumulative USD spend for a job from the Tinker billing API."""
    raise NotImplementedError


def cancel_job(job_id: str) -> None:
    """Immediately cancels and terminates a Tinker job, releasing the GPU instance."""
    raise NotImplementedError


def get_job_logs(job_id: str, tail: int = 100) -> list[str]:
    """Fetches the last N lines of stdout/stderr from a running or completed Tinker job."""
    raise NotImplementedError


def list_jobs(limit: int = 20) -> list[JobSummary]:
    """Lists recent Tinker jobs for the current account, ordered by submission time descending."""
    raise NotImplementedError
