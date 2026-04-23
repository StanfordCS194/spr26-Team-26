"""Tests for Tinker API Wrapper (owner: Sid Potti)"""

import pytest
from src.tinker_api.tinker_api import (
    submit_job,
    get_job_status,
    get_cumulative_spend,
    cancel_job,
    get_job_logs,
    list_jobs,
    TinkerAPIError,
)


def test_submit_job_returns_job_id():
    raise NotImplementedError


def test_get_job_status_returns_valid_status():
    raise NotImplementedError


def test_get_cumulative_spend_returns_float():
    raise NotImplementedError


def test_cancel_job_succeeds():
    raise NotImplementedError


def test_tinker_api_error_raised_on_non_2xx():
    raise NotImplementedError
