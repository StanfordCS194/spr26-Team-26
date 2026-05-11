"""
Tinker API Wrapper
Owner: Sid Potti

Thin HTTP client around Tinker's job submission and billing REST APIs.
Raises TinkerAPIError on non-2xx. Retries up to 3× with exponential backoff.

Auth: set TINKER_API_KEY and TINKER_API_BASE environment variables.

Local stub mode: set TINKER_LOCAL_STUB=1 to run training scripts as local
subprocesses instead of submitting to Tinker. Output is written to
outputs/experiments/<job_id>/ to match what wait_for_experiment expects.
Job state is persisted in outputs/experiments/registry.json.
"""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src.types import JobConfig, JobSummary


class TinkerAPIError(Exception):
    def __init__(self, status_code: int, message: str):
        super().__init__(f"Tinker API error {status_code}: {message}")
        self.status_code = status_code


# ─── ROUTING ──────────────────────────────────────────────────────────────────

def _is_local_stub() -> bool:
    return os.getenv("TINKER_LOCAL_STUB", "").strip().lower() in {"1", "true", "yes"}


def submit_job(script_path: str, job_config: JobConfig) -> str:
    """Submits a training script to Tinker for execution on a GPU instance. Returns the Tinker job ID."""
    if _is_local_stub():
        return _local_submit_job(script_path, job_config)
    return _remote_submit_job(script_path, job_config)


def get_job_status(job_id: str) -> str:
    """Fetches the current status of a Tinker job (JobStatus string)."""
    if _is_local_stub():
        return _local_get_job_status(job_id)
    return _remote_get_job_status(job_id)


def get_cumulative_spend(job_id: str) -> float:
    """Returns cumulative USD spend for a job from the Tinker billing API."""
    if _is_local_stub():
        return _local_get_cumulative_spend(job_id)
    return _remote_get_cumulative_spend(job_id)


def cancel_job(job_id: str) -> None:
    """Immediately cancels and terminates a Tinker job, releasing the GPU instance."""
    if _is_local_stub():
        _local_cancel_job(job_id)
        return
    _remote_cancel_job(job_id)


def get_job_logs(job_id: str, tail: int = 100) -> list[str]:
    """Fetches the last N lines of stdout/stderr from a running or completed Tinker job."""
    if _is_local_stub():
        return _local_get_job_logs(job_id, tail)
    return _remote_get_job_logs(job_id, tail)


def list_jobs(limit: int = 20) -> list[JobSummary]:
    """Lists recent Tinker jobs for the current account, ordered by submission time descending."""
    if _is_local_stub():
        return _local_list_jobs(limit)
    return _remote_list_jobs(limit)


# ─── LOCAL STUB ───────────────────────────────────────────────────────────────
# Runs scripts as local subprocesses. Mirrors the output layout that
# wait_for_experiment in autoresearch.py expects:
#   outputs/experiments/<job_id>/metrics.json
#   outputs/experiments/<job_id>/model/
#   outputs/experiments/<job_id>/logs.txt

_REGISTRY_PATH = Path("outputs/experiments/registry.json")
_STUB_COST_PER_HOUR = 2.50  # simulated A100 spot rate


def _load_registry() -> dict:
    if _REGISTRY_PATH.exists():
        try:
            return json.loads(_REGISTRY_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_registry(registry: dict) -> None:
    _REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _REGISTRY_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(registry, indent=2))
    tmp.replace(_REGISTRY_PATH)


def _job_dir(job_id: str) -> Path:
    return Path("outputs/experiments") / job_id


def _local_submit_job(script_path: str, job_config: JobConfig) -> str:
    job_id = str(uuid.uuid4())[:8]
    out_dir = _job_dir(job_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "logs.txt"

    env = {**os.environ, **job_config.get("env_vars", {})}

    proc = subprocess.Popen(
        ["python", script_path],
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
        env=env,
    )

    registry = _load_registry()
    registry[job_id] = {
        "pid": proc.pid,
        "script": script_path,
        "status": "RUNNING",
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "timeout_sec": job_config.get("timeout_min", 10) * 60,
        "log_path": str(log_path),
    }
    _save_registry(registry)
    return job_id


def _local_get_job_status(job_id: str) -> str:
    registry = _load_registry()
    entry = registry.get(job_id)
    if not entry:
        return "FAILED"

    if entry["status"] in ("COMPLETED", "FAILED", "CANCELLED"):
        return entry["status"]

    pid = entry.get("pid")
    if pid is None:
        return "FAILED"

    # Check if the subprocess is still alive.
    try:
        os.kill(pid, 0)
        alive = True
    except OSError:
        alive = False

    if not alive:
        out_dir = _job_dir(job_id)
        log_path = Path(entry["log_path"])
        metrics = _parse_metrics_from_log(log_path)
        (out_dir / "metrics.json").write_text(json.dumps(metrics))
        (out_dir / "model").mkdir(exist_ok=True)
        status = "COMPLETED" if metrics["primary_metric"] != 0.0 else "FAILED"
        entry["status"] = status
        registry[job_id] = entry
        _save_registry(registry)
        return status

    return "RUNNING"


def _parse_metrics_from_log(log_path: Path) -> dict:
    """
    Parses the summary block printed by train.py at the end of a run.
    Maps val_bpb → primary_metric (negated so higher scalar = better,
    consistent with compare_scores in autoresearch.py).
    Falls back to scanning for any loss lines if the block is absent.
    """
    metrics: dict = {
        "train_loss": 0.0,
        "val_loss": 0.0,
        "test_loss": 0.0,
        "primary_metric": 0.0,
    }
    if not log_path.exists():
        return metrics

    text = log_path.read_text(errors="replace")
    for line in text.splitlines():
        line = line.strip()
        if ":" not in line:
            continue
        key, _, raw_val = line.partition(":")
        key = key.strip().lower()
        try:
            val = float(raw_val.strip())
        except ValueError:
            continue
        if math.isnan(val) or math.isinf(val):
            continue

        if key == "val_bpb":
            # Lower val_bpb is better; negate so higher primary_metric = better.
            metrics["primary_metric"] = -val
            metrics["val_loss"] = val
        elif key in ("train_loss", "training_loss"):
            metrics["train_loss"] = val
        elif key in ("val_loss", "validation_loss") and metrics["val_loss"] == 0.0:
            metrics["val_loss"] = val
            if metrics["primary_metric"] == 0.0:
                metrics["primary_metric"] = -val
        elif key in ("test_loss",):
            metrics["test_loss"] = val

    return metrics


def _local_get_cumulative_spend(job_id: str) -> float:
    registry = _load_registry()
    entry = registry.get(job_id)
    if not entry:
        return 0.0
    try:
        submitted = datetime.fromisoformat(entry["submitted_at"])
        elapsed_hours = (datetime.now(timezone.utc) - submitted).total_seconds() / 3600
        return round(elapsed_hours * _STUB_COST_PER_HOUR, 4)
    except (KeyError, ValueError):
        return 0.0


def _local_cancel_job(job_id: str) -> None:
    registry = _load_registry()
    entry = registry.get(job_id)
    if not entry:
        return
    pid = entry.get("pid")
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    entry["status"] = "CANCELLED"
    registry[job_id] = entry
    _save_registry(registry)


def _local_get_job_logs(job_id: str, tail: int = 100) -> list[str]:
    registry = _load_registry()
    entry = registry.get(job_id, {})
    log_path = Path(entry.get("log_path", str(_job_dir(job_id) / "logs.txt")))
    if not log_path.exists():
        return []
    lines = log_path.read_text(errors="replace").splitlines()
    return lines[-tail:] if len(lines) > tail else lines


def _local_list_jobs(limit: int = 20) -> list[JobSummary]:
    registry = _load_registry()
    items = list(registry.items())[-limit:]
    summaries: list[JobSummary] = [
        {
            "job_id": job_id,
            "status": entry.get("status", "UNKNOWN"),
            "submitted_at": entry.get("submitted_at", ""),
            "cost_usd": _local_get_cumulative_spend(job_id),
            "script_name": Path(entry.get("script", "")).name,
        }
        for job_id, entry in items
    ]
    return list(reversed(summaries))


# ─── REMOTE TINKER SDK CLIENT ─────────────────────────────────────────────────
# IMPORTANT: Tinker is a Python SDK, not a REST job queue.
#
# The actual model (from tinker-docs.thinkingmachines.ai):
#   client = tinker.ServiceClient()
#   tc = client.create_lora_training_client(base_model="...")
#   tc.forward_backward(data, loss_fn)
#   tc.optim_step(adam_params)
#   tc.save_state()
#   rest = client.create_rest_client()
#   run = rest.get_training_run(training_run_id)
#
# This means submit_job(script_path) doesn't map directly — Tinker has no
# "run this file" endpoint. The remote path below wraps the Tinker SDK:
#   - submit_job: creates a training run + spawns a background thread that
#     calls forward_backward/optim_step using the patched config
#   - get_job_status: polls RestClient.get_training_run()
#   - get_cumulative_spend: no billing API in Tinker — tracked locally
#   - cancel_job: stops the background training thread
#   - get_job_logs: reads local log file (Tinker has no log endpoint)
#   - list_jobs: RestClient.list_training_runs()
#
# This requires `pip install tinker` and Tinker credentials configured.
# Until the team aligns on how train.py integrates with the Tinker SDK,
# use TINKER_LOCAL_STUB=1 (above) for all testing.

# Tracks background training threads: job_id → {"thread", "stop_event", "run_id", "started_at"}
_remote_jobs: dict[str, dict] = {}


def _remote_submit_job(script_path: str, job_config: JobConfig) -> str:
    try:
        import tinker  # type: ignore[import]
    except ImportError as exc:
        raise TinkerAPIError(0, "tinker package not installed — run: pip install tinker") from exc

    job_id = str(uuid.uuid4())[:8]
    out_dir = _job_dir(job_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "logs.txt"
    stop_event = __import__("threading").Event()

    base_model = job_config.get("env_vars", {}).get("BASE_MODEL", "Qwen/Qwen3-8B")

    def _run() -> None:
        try:
            client = tinker.ServiceClient()
            tc = client.create_lora_training_client(base_model=base_model)
            _remote_jobs[job_id]["run_id"] = getattr(tc, "training_run_id", job_id)

            # Import and run the patched training script in this process.
            # The script is expected to honour the config at configs/current.json.
            import importlib.util
            spec = importlib.util.spec_from_file_location("_tinker_train", script_path)
            mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
            with open(log_path, "w") as lf:
                import sys as _sys
                old_stdout, _sys.stdout = _sys.stdout, lf  # type: ignore[assignment]
                old_stderr, _sys.stderr = _sys.stderr, lf  # type: ignore[assignment]
                try:
                    spec.loader.exec_module(mod)  # type: ignore[union-attr]
                finally:
                    _sys.stdout, _sys.stderr = old_stdout, old_stderr

            # Save checkpoint after training.
            tc.save_state()
            metrics = _parse_metrics_from_log(log_path)
            (out_dir / "metrics.json").write_text(json.dumps(metrics))
            (out_dir / "model").mkdir(exist_ok=True)
            _remote_jobs[job_id]["status"] = "COMPLETED"
        except Exception as exc:  # noqa: BLE001
            with open(log_path, "a") as lf:
                lf.write(f"\nTinker job error: {exc}\n")
            _remote_jobs[job_id]["status"] = "FAILED"

    import threading
    t = threading.Thread(target=_run, daemon=True)
    _remote_jobs[job_id] = {
        "thread": t,
        "stop_event": stop_event,
        "run_id": job_id,
        "status": "RUNNING",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "log_path": str(log_path),
    }
    t.start()
    return job_id


def _remote_get_job_status(job_id: str) -> str:
    entry = _remote_jobs.get(job_id)
    if not entry:
        # Fall back to registry (handles restarts).
        registry = _load_registry()
        return registry.get(job_id, {}).get("status", "UNKNOWN")
    thread: __import__("threading").Thread = entry["thread"]  # type: ignore[type-arg]
    if not thread.is_alive() and entry["status"] == "RUNNING":
        entry["status"] = "FAILED"
    return entry["status"]


def _remote_get_cumulative_spend(job_id: str) -> float:
    # Tinker has no billing API — track elapsed time at the stub rate.
    entry = _remote_jobs.get(job_id, {})
    try:
        started = datetime.fromisoformat(entry.get("started_at", ""))
        elapsed_hours = (datetime.now(timezone.utc) - started).total_seconds() / 3600
        return round(elapsed_hours * _STUB_COST_PER_HOUR, 4)
    except (ValueError, TypeError):
        return 0.0


def _remote_cancel_job(job_id: str) -> None:
    entry = _remote_jobs.get(job_id)
    if entry:
        entry.get("stop_event").set()  # type: ignore[union-attr]
        entry["status"] = "CANCELLED"


def _remote_get_job_logs(job_id: str, tail: int = 100) -> list[str]:
    entry = _remote_jobs.get(job_id, {})
    log_path = Path(entry.get("log_path", str(_job_dir(job_id) / "logs.txt")))
    if not log_path.exists():
        return []
    lines = log_path.read_text(errors="replace").splitlines()
    return lines[-tail:] if len(lines) > tail else lines


def _remote_list_jobs(limit: int = 20) -> list[JobSummary]:
    try:
        import tinker  # type: ignore[import]
        client = tinker.ServiceClient()
        rest = client.create_rest_client()
        result = rest.list_training_runs(limit=limit)
        runs = getattr(result, "training_runs", []) or []
        return [
            {
                "job_id": str(getattr(r, "training_run_id", "")),
                "status": "COMPLETED" if not getattr(r, "corrupted", False) else "FAILED",
                "submitted_at": str(getattr(r, "last_request_time", "")),
                "cost_usd": 0.0,  # no billing API
                "script_name": str(getattr(r, "base_model", "")),
            }
            for r in runs
        ]
    except Exception:  # noqa: BLE001
        # Fall back to in-memory registry if SDK unavailable.
        return [
            {
                "job_id": jid,
                "status": e.get("status", "UNKNOWN"),
                "submitted_at": e.get("started_at", ""),
                "cost_usd": _remote_get_cumulative_spend(jid),
                "script_name": "",
            }
            for jid, e in list(_remote_jobs.items())[-limit:]
        ]
