"""Tests for Tinker API Wrapper (owner: Sid Potti)"""

import pytest
from unittest.mock import MagicMock, patch
from src.tinker_api.tinker_api import (
    TinkerAPIError,
    cancel_job,
    get_cumulative_spend,
    get_job_status,
    list_jobs,
    submit_job,
)
from src.types import JobConfig, JobStatus


def _mock_response(status_code: int, json_data: dict) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.content = b"ok"
    resp.text = str(json_data)
    return resp


def _job_config() -> JobConfig:
    return JobConfig(gpu_type="A100", num_gpus=1, timeout_min=30,
                     env_vars={}, output_dir="outputs/")


def test_submit_job_returns_job_id(tmp_path):
    script = tmp_path / "train.py"
    script.write_text("print('training')")
    with patch("requests.request", return_value=_mock_response(200, {"job_id": "abc-123"})):
        job_id = submit_job(str(script), _job_config())
    assert job_id == "abc-123"


def test_get_job_status_returns_valid_status():
    with patch("requests.request", return_value=_mock_response(200, {"status": JobStatus.RUNNING})):
        status = get_job_status("abc-123")
    assert status == JobStatus.RUNNING


def test_get_cumulative_spend_returns_float():
    with patch("requests.request", return_value=_mock_response(200, {"spend_usd": "12.50"})):
        spend = get_cumulative_spend("abc-123")
    assert isinstance(spend, float)
    assert spend == 12.50


def test_cancel_job_succeeds():
    with patch("requests.request", return_value=_mock_response(200, {})):
        cancel_job("abc-123")  # should not raise


def test_tinker_api_error_raised_on_non_2xx():
    err_resp = MagicMock()
    err_resp.status_code = 404
    err_resp.text = "not found"
    err_resp.content = b"not found"
    with patch("requests.request", return_value=err_resp):
        with pytest.raises(TinkerAPIError) as exc_info:
            get_job_status("bad-id")
    assert exc_info.value.status_code == 404
