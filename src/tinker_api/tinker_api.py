"""
Tinker API Wrapper
Owner: Sid Potti

Thin HTTP client around Tinker's job submission and billing REST APIs.
Raises TinkerAPIError on non-2xx. Retries up to 3× with exponential backoff.

Auth: set TINKER_API_KEY and TINKER_API_BASE environment variables.
"""

from __future__ import annotations

import os
import time
import requests

from src.types import JobConfig, JobSummary, JobStatus

TINKER_API_BASE = os.environ.get("TINKER_API_BASE", "https://api.tinker.ai/v1")
TINKER_API_KEY  = os.environ.get("TINKER_API_KEY", "")
_MAX_RETRIES = 3


class TinkerAPIError(Exception):
    def __init__(self, status_code: int, message: str):
        super().__init__(f"Tinker API error {status_code}: {message}")
        self.status_code = status_code


def _headers() -> dict:
    return {"Authorization": f"Bearer {TINKER_API_KEY}", "Content-Type": "application/json"}


def _request(method: str, path: str, **kwargs) -> dict:
    """Makes an HTTP request with retry/backoff. Raises TinkerAPIError on non-2xx."""
    url = f"{TINKER_API_BASE}{path}"
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.request(method, url, headers=_headers(), timeout=30, **kwargs)
            if resp.status_code < 300:
                return resp.json() if resp.content else {}
            if resp.status_code >= 500 and attempt < _MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
                continue
            raise TinkerAPIError(resp.status_code, resp.text)
        except requests.RequestException as e:
            if attempt == _MAX_RETRIES - 1:
                raise TinkerAPIError(0, str(e))
            time.sleep(2 ** attempt)
    raise TinkerAPIError(0, "Max retries exceeded")


def submit_job(script_path: str, job_config: JobConfig) -> str:
    """Submits a training script to Tinker for execution. Returns the Tinker job ID."""
    with open(script_path, "r") as f:
        script_content = f.read()
    data = _request("POST", "/jobs", json={
        "script": script_content,
        "gpu_type": job_config["gpu_type"],
        "num_gpus": job_config["num_gpus"],
        "timeout_min": job_config["timeout_min"],
        "env_vars": job_config.get("env_vars", {}),
        "output_dir": job_config.get("output_dir", "outputs/"),
    })
    return data["job_id"]


def get_job_status(job_id: str) -> str:
    """Fetches the current status of a Tinker job (JobStatus string)."""
    data = _request("GET", f"/jobs/{job_id}/status")
    return data["status"]


def get_cumulative_spend(job_id: str) -> float:
    """Returns cumulative USD spend for a job from the Tinker billing API."""
    data = _request("GET", f"/billing/{job_id}/spend")
    return float(data["spend_usd"])


def cancel_job(job_id: str) -> None:
    """Immediately cancels and terminates a Tinker job, releasing the GPU instance."""
    _request("POST", f"/jobs/{job_id}/cancel")


def get_job_logs(job_id: str, tail: int = 100) -> list[str]:
    """Fetches the last N lines of stdout/stderr from a running or completed Tinker job."""
    data = _request("GET", f"/jobs/{job_id}/logs", params={"tail": tail})
    return data.get("lines", [])


def list_jobs(limit: int = 20) -> list[JobSummary]:
    """Lists recent Tinker jobs ordered by submission time descending."""
    data = _request("GET", "/jobs", params={"limit": limit})
    return data.get("jobs", [])
