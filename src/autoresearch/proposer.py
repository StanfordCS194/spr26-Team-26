"""
Proposal strategies for the AutoResearch PROPOSE phase.

RandomSearchProposalStrategy samples one parameter uniformly from its full
range — useful early on when there's no signal yet.

LocalPerturbationProposalStrategy applies a small ±perturbation around the
current config, biasing away from parameters that recently caused reverts.

Both are pure algorithmic (no API calls), deterministic given a seed, and
used by the CLI scaffold. The full LangGraph graph calls Claude directly
via propose_hypothesis() in autoresearch.py.
"""

from __future__ import annotations

import abc
import json
import math
import random
from dataclasses import dataclass, field
from typing import Any

from src.autoresearch.config import BOUNDS, TrainingConfig
from src.types import IterationRecord


# ─── PROPOSAL DATACLASS ───────────────────────────────────────────────────────

@dataclass
class Proposal:
    """One proposed config change from a ProposalStrategy."""
    hypothesis: str
    patch: dict[str, Any]
    target: str = "training_config"
    metadata: dict[str, Any] = field(default_factory=dict)


# ─── SEARCH SPACE ─────────────────────────────────────────────────────────────

class SearchSpace:
    """Which parameters are tunable and how to sample or perturb them. Reads from BOUNDS in config.py."""

    # Parameters the proposer should never touch
    NON_TUNABLE: frozenset[str] = frozenset({"model_name"})

    @classmethod
    def tunable_params(cls) -> list[str]:
        return [k for k in BOUNDS if k not in cls.NON_TUNABLE]

    @classmethod
    def sample_value(cls, param: str, rng: random.Random) -> Any:
        """Sample a value for param from its full allowed range."""
        spec = BOUNDS[param]
        if "candidates" in spec:
            return rng.choice(spec["candidates"])
        lo, hi = spec["min"], spec["max"]
        if spec.get("scale") == "log":
            val = math.exp(rng.uniform(math.log(lo), math.log(hi)))
        else:
            val = rng.uniform(lo, hi)
        if spec.get("type") == int:
            val = int(round(val))
        return val

    @classmethod
    def perturb_value(cls, param: str, current: Any, factor: float, rng: random.Random) -> Any:
        """Multiplicative ±factor perturbation, clamped to bounds. Falls back to full-range sample if current is None."""
        spec = BOUNDS[param]
        if "candidates" in spec:
            candidates = spec["candidates"]
            if current is None:
                return rng.choice(candidates)
            try:
                idx = candidates.index(current)
            except ValueError:
                idx = 0
            delta = rng.choice([-1, 0, 1])
            new_idx = max(0, min(len(candidates) - 1, idx + delta))
            return candidates[new_idx]
        # Multiplicative perturbation doesn't work on zero or None — sample fresh instead.
        if current is None or current == 0:
            return cls.sample_value(param, rng)
        multiplier = 1.0 + rng.uniform(-factor, factor)
        val = current * multiplier
        lo = spec.get("min", val)
        hi = spec.get("max", val)
        val = max(lo, min(hi, val))
        if spec.get("type") == int:
            val = int(round(val))
        return val


# ─── ABSTRACT BASE ────────────────────────────────────────────────────────────

class ProposalStrategy(abc.ABC):
    """Base class for proposal strategies. propose() must return a Proposal changing exactly one hyperparameter."""

    @abc.abstractmethod
    def propose(
        self,
        config: TrainingConfig,
        history: list[IterationRecord],
    ) -> Proposal:
        ...


# ─── RANDOM SEARCH ────────────────────────────────────────────────────────────

class RandomSearchProposalStrategy(ProposalStrategy):
    """Samples one parameter uniformly from its full range, ignoring history. Good for early exploration."""

    def __init__(self, seed: int | None = None) -> None:
        # Advance the seed each call so a fixed initial seed still varies across iterations.
        self._next_seed = seed
        self._initial_seed = seed

    def propose(self, config: TrainingConfig, history: list[IterationRecord]) -> Proposal:
        if self._next_seed is not None:
            seed = self._next_seed
            self._next_seed = (self._next_seed + 1) % (2**32)
        else:
            seed = random.randint(0, 2**32 - 1)
        rng = random.Random(seed)
        param = rng.choice(SearchSpace.tunable_params())
        old_val = getattr(config, param, None)
        new_val = SearchSpace.sample_value(param, rng)
        hypothesis = (
            f"Change {param} from {old_val} to {new_val} via random search "
            f"to explore a different region of the hyperparameter space."
        )
        return Proposal(
            hypothesis=hypothesis,
            patch={param: new_val},
            metadata={"strategy": "random_search", "seed": seed, "param": param},
        )


# ─── LOCAL PERTURBATION ───────────────────────────────────────────────────────

class LocalPerturbationProposalStrategy(ProposalStrategy):
    """
    Perturbs one parameter by ±perturbation_factor around its current value.
    Parameters that recently caused REVERTED entries are deprioritised so the
    loop avoids re-testing directions that already failed.
    """

    def __init__(
        self,
        perturbation_factor: float = 0.2,
        seed: int | None = None,
        history_window: int = 5,
    ) -> None:
        self._factor = perturbation_factor
        self._next_seed = seed
        self._window = history_window

    def propose(self, config: TrainingConfig, history: list[IterationRecord]) -> Proposal:
        if self._next_seed is not None:
            seed = self._next_seed
            self._next_seed = (self._next_seed + 1) % (2**32)
        else:
            seed = random.randint(0, 2**32 - 1)
        rng = random.Random(seed)
        candidates = SearchSpace.tunable_params()

        # The diary patch field is a diff string, not JSON — parse "+ key: value" lines.
        regressed: set[str] = set()
        for entry in history[-self._window:]:
            if entry.get("decision") == "REVERTED":
                for line in entry.get("patch", "").splitlines():
                    if line.startswith("+ "):
                        key = line[2:].partition(":")[0].strip()
                        if key:
                            regressed.add(key)

        preferred = [p for p in candidates if p not in regressed] or candidates
        param = rng.choice(preferred)
        old_val = getattr(config, param, None)
        new_val = SearchSpace.perturb_value(param, old_val, self._factor, rng)

        direction = "increase" if (
            isinstance(new_val, (int, float)) and isinstance(old_val, (int, float))
            and new_val > old_val
        ) else "decrease"
        hypothesis = (
            f"{direction.capitalize()} {param} from {old_val} to {new_val} "
            f"(local ±{int(self._factor * 100)}% perturbation) "
            f"to refine the current best configuration."
        )
        return Proposal(
            hypothesis=hypothesis,
            patch={param: new_val},
            metadata={
                "strategy": "local_perturbation",
                "seed": seed,
                "param": param,
                "factor": self._factor,
                "regressed_params": list(regressed),
            },
        )
