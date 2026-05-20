"""
Proposal strategies for the PROPOSE phase of the AutoResearch Loop.

Two concrete strategies are provided:

  RandomSearchProposalStrategy
    Samples one tunable parameter uniformly (or log-uniformly) within its
    safe range, independent of history. Useful early in the search when
    there is little prior information.

  LocalPerturbationProposalStrategy
    Applies a small ±perturbation around the current best config. Biases
    away from parameters that recently caused REVERTED entries, so the loop
    spends less time re-testing known-bad directions.

Both strategies are pure algorithmic — no Claude API calls — so they are
cheap, fast, and deterministic given a seed. The LangGraph propose_node
uses Claude (propose_hypothesis in autoresearch.py) for richer reasoning;
these strategies are used by the CLI scaffold (AutoResearchLoop in loop.py)
and can also be composed with the Claude path.
"""

from __future__ import annotations

import abc
import json
import math
import random
from dataclasses import dataclass, field
from typing import Any

from src.autoresearch.config import BOUNDS, TrainingConfig
from src.tinker_api.sft_runner import SUPPORTED_TINKER_TUNABLES
from src.types import IterationRecord


# ─── PROPOSAL DATACLASS ───────────────────────────────────────────────────────

@dataclass
class Proposal:
    """
    Atomic config-level proposal from a ProposalStrategy.

    hypothesis: natural-language description of the change and its rationale.
    patch:      dict with exactly one key — the parameter being changed.
    target:     what is being patched (always "training_config" for now;
                "training_script" will be used once train.py patching is wired in).
    metadata:   strategy name, seed, and any strategy-specific context for
                diary logging and reproducibility.
    """
    hypothesis: str
    patch: dict[str, Any]
    target: str = "training_config"
    metadata: dict[str, Any] = field(default_factory=dict)


# ─── SEARCH SPACE ─────────────────────────────────────────────────────────────

class SearchSpace:
    """
    Declares which parameters are tunable and how to sample or perturb them.

    Reads directly from BOUNDS in config.py so adding a new hyperparameter
    there automatically makes it available for exploration here.
    """

    # Parameters the proposer should never touch
    NON_TUNABLE: frozenset[str] = frozenset({"model_name"})

    @classmethod
    def tunable_params(cls, backend: str | None = None) -> list[str]:
        params = [k for k in BOUNDS if k not in cls.NON_TUNABLE]
        if backend == "tinker_sft":
            return [k for k in params if k in SUPPORTED_TINKER_TUNABLES]
        return params

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
    def sample_value_except(cls, param: str, current: Any, rng: random.Random) -> Any:
        """Sample a valid value that differs from current when the space allows it."""
        spec = BOUNDS[param]
        if "candidates" in spec:
            candidates = [value for value in spec["candidates"] if value != current]
            return rng.choice(candidates or spec["candidates"])

        for _ in range(16):
            val = cls.sample_value(param, rng)
            if val != current:
                return val

        lo = spec.get("min", current)
        hi = spec.get("max", current)
        if spec.get("type") == int:
            current_int = int(current)
            if current_int < hi:
                return current_int + 1
            if current_int > lo:
                return current_int - 1
        if current != lo:
            return lo
        return hi

    @classmethod
    def perturb_value(cls, param: str, current: Any, factor: float, rng: random.Random) -> Any:
        """
        Apply a multiplicative ±factor perturbation around current, clamped to bounds.

        For discrete candidates, moves one step up or down the ordered list.
        If current is None (e.g. lora_rank=None meaning LoRA is disabled),
        falls back to sampling a fresh value from the full range.
        """
        spec = BOUNDS[param]
        if "candidates" in spec:
            candidates = spec["candidates"]
            if current is None:
                return rng.choice(candidates)
            try:
                idx = candidates.index(current)
            except ValueError:
                return cls.sample_value_except(param, current, rng)
            neighbor_indices = [i for i in (idx - 1, idx + 1) if 0 <= i < len(candidates)]
            if not neighbor_indices:
                return current
            new_idx = rng.choice(neighbor_indices)
            return candidates[new_idx]
        # current=None or current=0 with a multiplicative scheme would get stuck;
        # fall back to sampling from the full range in those cases.
        if current is None or current == 0:
            return cls.sample_value_except(param, current, rng)
        lo = spec.get("min", current)
        hi = spec.get("max", current)
        if spec.get("type") == int:
            current_int = int(current)
            direction = rng.choice([-1, 1])
            if current_int <= lo:
                direction = 1
            elif current_int >= hi:
                direction = -1
            max_step = max(1, int(round(abs(current_int) * factor)))
            step = rng.randint(1, max_step) * direction
            val = max(lo, min(hi, current_int + step))
            if val == current_int:
                return cls.sample_value_except(param, current_int, rng)
            return int(val)
        direction = rng.choice([-1.0, 1.0])
        magnitude = rng.uniform(max(factor * 0.1, 1e-12), factor)
        multiplier = 1.0 + direction * magnitude
        val = current * multiplier
        val = max(lo, min(hi, val))
        if val == current:
            return cls.sample_value_except(param, current, rng)
        return val


# ─── ABSTRACT BASE ────────────────────────────────────────────────────────────

class ProposalStrategy(abc.ABC):
    """
    Abstract base for all proposal strategies.

    Each concrete strategy must implement propose(), which takes the current
    best TrainingConfig and the recent diary history and returns an atomic
    Proposal changing exactly one hyperparameter.
    """

    @abc.abstractmethod
    def propose(
        self,
        config: TrainingConfig,
        history: list[IterationRecord],
    ) -> Proposal:
        ...


# ─── RANDOM SEARCH ────────────────────────────────────────────────────────────

class RandomSearchProposalStrategy(ProposalStrategy):
    """
    Samples one tunable parameter uniformly (or log-uniformly) within its
    allowed range, independent of history.

    Best for early exploration when there is little prior signal. The seed
    stored in Proposal.metadata makes any iteration reproducible.
    """

    def __init__(self, seed: int | None = None, backend: str | None = None) -> None:
        # _next_seed advances on each call so a fixed initial seed still
        # produces a distinct proposal every iteration.
        self._next_seed = seed
        self._initial_seed = seed
        self._backend = backend

    def propose(self, config: TrainingConfig, history: list[IterationRecord]) -> Proposal:
        if self._next_seed is not None:
            seed = self._next_seed
            self._next_seed = (self._next_seed + 1) % (2**32)
        else:
            seed = random.randint(0, 2**32 - 1)
        rng = random.Random(seed)
        param = rng.choice(SearchSpace.tunable_params(self._backend))
        old_val = getattr(config, param, None)
        new_val = SearchSpace.sample_value_except(param, old_val, rng)
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
    Applies a small ±perturbation around the current best config.

    Uses recent diary history to bias the choice of parameter: parameters
    that recently caused REVERTED entries are deprioritised, so the strategy
    naturally avoids re-testing known-bad directions.

    perturbation_factor: relative magnitude of the perturbation, e.g.
        0.2 means ±20% of the current value.
    history_window: how many recent diary entries to consider for bias.
    """

    def __init__(
        self,
        perturbation_factor: float = 0.2,
        seed: int | None = None,
        history_window: int = 5,
        backend: str | None = None,
    ) -> None:
        self._factor = perturbation_factor
        self._next_seed = seed
        self._window = history_window
        self._backend = backend

    def propose(self, config: TrainingConfig, history: list[IterationRecord]) -> Proposal:
        if self._next_seed is not None:
            seed = self._next_seed
            self._next_seed = (self._next_seed + 1) % (2**32)
        else:
            seed = random.randint(0, 2**32 - 1)
        rng = random.Random(seed)
        candidates = SearchSpace.tunable_params(self._backend)

        # Collect params that appeared in recently REVERTED patches.
        # The diary "patch" field is a diff string (format_patch_as_diff output),
        # not JSON — extract changed keys by parsing "+ key: value" lines.
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
