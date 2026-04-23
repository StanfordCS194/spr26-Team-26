"""
AutoResearch training configuration.

TrainingConfig is the shared hyperparameter representation used by the
Decision Engine (which produces the baseline) and the AutoResearch Loop
(which iterates on it). apply_patch always returns a new instance so the
original is never mutated in-place.

BOUNDS is the single source of truth for what values are legal. Both
ProposalStrategy (algorithmic sampling) and apply_patch (validation) read
from it, so adding a new knob means editing exactly one place.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Optional


# ─── SAFETY BOUNDS ────────────────────────────────────────────────────────────
# Each entry declares how a parameter may be changed.
# "scale" is used by sampling strategies: "log" for params that span decades
# (lr, seq_length), "linear" otherwise.
# "candidates" means only discrete values are allowed.

BOUNDS: dict[str, dict[str, Any]] = {
    "learning_rate":  {"min": 1e-6,  "max": 1e-2,  "scale": "log"},
    "batch_size":     {"min": 4,     "max": 8192,   "scale": "linear", "type": int},
    "weight_decay":   {"min": 0.0,   "max": 0.1,    "scale": "linear"},
    "num_epochs":     {"min": 1,     "max": 100,    "scale": "linear", "type": int},
    "max_seq_length": {"min": 128,   "max": 4096,   "scale": "log",    "type": int},
    "lora_rank":      {"candidates": [4, 8, 16, 32, 64, 128],           "type": int},
    "lora_alpha":     {"candidates": [8, 16, 32, 64, 128, 256],         "type": int},
    "optimizer":      {"candidates": ["adamw", "adam", "sgd", "lion"]},
    "warmup_steps":   {"min": 0,     "max": 2000,   "scale": "linear", "type": int},
    "dropout":        {"min": 0.0,   "max": 0.5,    "scale": "linear"},
}


# ─── TRAINING CONFIG ──────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    """
    Hyperparameter snapshot for one training run.

    Consumed by Decision Engine to produce a baseline, then mutated by the
    AutoResearch Loop via apply_patch(). Immutable in practice: every patch
    returns a new instance.
    """
    model_name: str
    learning_rate: float = 3e-4
    batch_size: int = 16
    weight_decay: float = 0.01
    num_epochs: int = 3
    max_seq_length: int = 512
    lora_rank: Optional[int] = 16
    lora_alpha: Optional[int] = 32
    optimizer: str = "adamw"
    warmup_steps: int = 100
    dropout: float = 0.1

    # ── Serialisation ──────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingConfig":
        valid_keys = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid_keys})

    @classmethod
    def load(cls, path: Path | str) -> "TrainingConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    # ── Patching ───────────────────────────────────────────────────────────

    def apply_patch(self, patch: dict[str, Any]) -> "TrainingConfig":
        """
        Returns a new TrainingConfig with patch values applied and validated.

        Raises ValueError if any patched value is out of bounds or wrong type.
        Validates against BOUNDS — unknown keys are rejected to prevent silent
        drift from the spec.
        """
        updated = self.to_dict()
        for key, value in patch.items():
            if key == "model_name":
                updated[key] = str(value)
                continue
            if key not in BOUNDS:
                raise ValueError(f"Unknown hyperparameter: {key!r}")
            spec = BOUNDS[key]
            # Type coercion
            if "type" in spec:
                value = spec["type"](value)
            # Discrete candidates
            if "candidates" in spec and value not in spec["candidates"]:
                raise ValueError(
                    f"{key}={value!r} not in allowed candidates {spec['candidates']}"
                )
            # NaN is not a valid hyperparameter value — reject before bound checks
            # because NaN comparisons always return False, letting it silently pass.
            if isinstance(value, float) and (value != value):  # NaN check
                raise ValueError(f"{key}: NaN is not a valid hyperparameter value")
            # Continuous bounds
            if "min" in spec and value < spec["min"]:
                raise ValueError(f"{key}={value} is below minimum {spec['min']}")
            if "max" in spec and value > spec["max"]:
                raise ValueError(f"{key}={value} is above maximum {spec['max']}")
            updated[key] = value
        return TrainingConfig.from_dict(updated)
