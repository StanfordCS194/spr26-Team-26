from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.autoresearch.config import TrainingConfig
from src.tinker_api.sft_runner import (
    DEFAULT_LIVE_SMOKE_STEPS,
    DEFAULT_TINKER_MODEL,
    run_tinker_sft_experiment,
)


def test_live_tinker_chat_sft_smoke(tmp_path):
    if os.getenv("RUN_LIVE_TINKER") != "1":
        pytest.skip("set RUN_LIVE_TINKER=1 to run the live Tinker smoke test")
    if not os.getenv("TINKER_API_KEY"):
        pytest.skip("TINKER_API_KEY is required for the live Tinker smoke test")

    model = os.getenv("TINKER_SMOKE_MODEL", DEFAULT_TINKER_MODEL)
    steps = int(os.getenv("TINKER_SMOKE_STEPS", str(DEFAULT_LIVE_SMOKE_STEPS)))
    data_path = tmp_path / "train.jsonl"
    rows = [
        {"input": "Reply with the word alpha.", "output": "alpha"},
        {"input": "Reply with the word beta.", "output": "beta"},
        {"input": "Reply with the word gamma.", "output": "gamma"},
        {"input": "Reply with the word delta.", "output": "delta"},
        {"input": "Reply with the word epsilon.", "output": "epsilon"},
    ]
    data_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    result = run_tinker_sft_experiment(
        TrainingConfig(
            model_name=model,
            learning_rate=1e-4,
            batch_size=1,
            num_epochs=1,
            max_seq_length=512,
            lora_rank=8,
        ),
        str(data_path),
        run_id="live-tinker-smoke",
        max_steps=steps,
        output_dir=str(tmp_path / "experiments"),
    )

    run_dir = tmp_path / "experiments" / "live-tinker-smoke"
    assert result["status"] == "COMPLETED"
    assert result["model_path"] == str(run_dir)
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "metrics.jsonl").exists()
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "sample.json").exists()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["completed_steps"] == steps
    assert manifest["checkpoints"]
