"""
Tinker SDK Wrapper
Owner: Sid Potti

Thin wrapper around Thinking Machines Lab's Tinker training SDK.
Exposes helpers for creating clients, running training steps, saving
checkpoints, and tracking token-based billing.

Auth: set TINKER_API_KEY env var — read automatically by tinker.ServiceClient.
Pricing: $0.40 / million tokens processed.

Reference: https://docs.tinkerlabs.ai/sdk
"""

from __future__ import annotations

import os
from typing import Any

try:
    import tinker                        # pip install tinker
    from tinker import types as tinker_types
    _TINKER_AVAILABLE = True
except ImportError:                      # graceful degradation in test / CI envs
    tinker = None          # type: ignore[assignment]
    tinker_types = None    # type: ignore[assignment]
    _TINKER_AVAILABLE = False

# ── Pricing constant ───────────────────────────────────────────────────────────
_COST_PER_MILLION_TOKENS: float = 0.40   # USD

# ── Module-level state (per-process) ──────────────────────────────────────────
# job_id → cumulative tokens seen in run_training_step calls
_token_ledger: dict[str, int] = {}

# job_id → True when cancel_job() has been called
_cancel_signals: dict[str, bool] = {}


# ── Error class ────────────────────────────────────────────────────────────────

class TinkerAPIError(Exception):
    """Raised when a Tinker SDK call fails or the package is not installed."""

    def __init__(self, message: str, status_code: int = 0):
        super().__init__(f"Tinker error: {message}")
        self.status_code = status_code


def _require_tinker() -> None:
    if not _TINKER_AVAILABLE:
        raise TinkerAPIError(
            "tinker package not installed — run: pip install tinker",
            status_code=0,
        )


def _live_tinker_block_reason() -> str | None:
    if os.getenv("NO_SPEND", "").strip() == "1":
        return "NO_SPEND=1"
    if os.getenv("TINKER_BACKEND", "").strip().lower() == "dry_run":
        return "TINKER_BACKEND=dry_run"
    return None


def _require_live_tinker_allowed(operation: str) -> None:
    reason = _live_tinker_block_reason()
    if reason:
        raise TinkerAPIError(
            f"{operation} is blocked because live Tinker API calls are disabled by {reason}",
            status_code=0,
        )


# ── Client creation ────────────────────────────────────────────────────────────

def create_service_client() -> Any:
    """Creates and returns an authenticated Tinker ServiceClient.

    Reads TINKER_API_KEY from the environment automatically.
    """
    _require_live_tinker_allowed("create_service_client")
    _require_tinker()
    try:
        return tinker.ServiceClient()
    except Exception as exc:
        raise TinkerAPIError(str(exc)) from exc


def create_lora_training_client(
    service_client: Any,
    base_model: str,
    rank: int = 32,
) -> Any:
    """Creates a LoRA training client for the given HuggingFace base model.

    Returns a client with forward_backward / optim_step / save_weights methods.
    """
    _require_live_tinker_allowed("create_lora_training_client")
    _require_tinker()
    try:
        return service_client.create_lora_training_client(
            base_model=base_model,
            rank=rank,
        )
    except Exception as exc:
        raise TinkerAPIError(str(exc)) from exc


# ── Tokenisation helpers ───────────────────────────────────────────────────────

def get_tokenizer(training_client: Any) -> Any:
    """Returns the tokeniser for the training client's base model."""
    _require_live_tinker_allowed("get_tokenizer")
    return training_client.get_tokenizer()


def make_datum(
    input_tokens: list[int],
    target_tokens: list[int] | None = None,
    weights: list[float] | None = None,
) -> Any:
    """Wraps token lists in a tinker.types.Datum for the training loop.

    If target_tokens is None a standard causal-LM shift is applied:
    input  = input_tokens[:-1]
    target = input_tokens[1:]
    """
    _require_live_tinker_allowed("make_datum")
    _require_tinker()
    if target_tokens is None:
        target_tokens = input_tokens[1:]
        input_tokens = input_tokens[:-1]
    if weights is None:
        weights = [1.0] * len(target_tokens)
    return tinker_types.Datum(
        model_input=tinker_types.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={"weights": weights, "target_tokens": target_tokens},
    )


# ── Training loop primitives ───────────────────────────────────────────────────

def run_training_step(
    training_client: Any,
    batch: list[Any],
    learning_rate: float = 2e-5,
    job_id: str | None = None,
) -> dict:
    """Runs one forward/backward pass + AdamW optimizer step on a batch of Datums.

    Optionally records token usage to the ledger under *job_id*.
    Returns ``{"tokens_processed": int, "loss": float | None}``.
    """
    _require_live_tinker_allowed("run_training_step")
    _require_tinker()
    try:
        fwdbwd_result = training_client.forward_backward(
            batch, loss_fn="cross_entropy"
        ).result()
        training_client.optim_step(
            tinker_types.AdamParams(learning_rate=learning_rate)
        ).result()
    except Exception as exc:
        raise TinkerAPIError(str(exc)) from exc

    n_tokens = sum(len(d.model_input.tokens) for d in batch)
    if job_id:
        record_tokens(job_id, n_tokens)

    loss = getattr(fwdbwd_result, "loss", None)
    return {
        "tokens_processed": n_tokens,
        "loss": float(loss) if loss is not None else None,
    }


def run_training_loop(
    training_client: Any,
    batches: list[list[Any]],
    learning_rate: float = 2e-5,
    job_id: str | None = None,
    checkpoint_name: str = "final",
) -> dict:
    """Runs a full training loop over *batches*.

    Checks for cancellation between steps (via :func:`cancel_job`).
    Saves a final checkpoint when done or cancelled.
    Returns ``{"steps": int, "tokens": int, "cancelled": bool}``.
    """
    steps = 0
    total_tokens = 0
    cancelled = False

    for batch in batches:
        if job_id and is_cancelled(job_id):
            cancelled = True
            break
        result = run_training_step(
            training_client, batch,
            learning_rate=learning_rate,
            job_id=job_id,
        )
        steps += 1
        total_tokens += result["tokens_processed"]

    try:
        save_weights(training_client, checkpoint_name)
    except TinkerAPIError:
        pass  # best-effort checkpoint on error

    return {"steps": steps, "tokens": total_tokens, "cancelled": cancelled}


# ── Checkpointing & inference ──────────────────────────────────────────────────

def save_checkpoint(training_client: Any, name: str) -> Any:
    """Saves weights and returns a SamplingClient for inference against this checkpoint."""
    _require_live_tinker_allowed("save_checkpoint")
    _require_tinker()
    try:
        return training_client.save_weights_and_get_sampling_client(name=name)
    except Exception as exc:
        raise TinkerAPIError(str(exc)) from exc


def save_weights(training_client: Any, name: str) -> None:
    """Saves weights without returning a sampling client (fire-and-forget)."""
    _require_live_tinker_allowed("save_weights")
    _require_tinker()
    try:
        training_client.save_weights_for_sampler(name=name)
    except Exception as exc:
        raise TinkerAPIError(str(exc)) from exc


def sample(
    sampling_client: Any,
    prompt_tokens: list[int],
    num_samples: int = 1,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
) -> list[list[int]]:
    """Runs inference using a saved checkpoint's SamplingClient.

    Returns a list of token-id lists (one per sample).
    """
    _require_live_tinker_allowed("sample")
    _require_tinker()
    try:
        model_input = tinker_types.ModelInput.from_ints(prompt_tokens)
        sampling_params = tinker_types.SamplingParams(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        result = sampling_client.sample(
            prompt=model_input,
            num_samples=num_samples,
            sampling_params=sampling_params,
        )
        return [list(r.tokens) for r in result.samples]
    except Exception as exc:
        raise TinkerAPIError(str(exc)) from exc


# ── Cost tracking ──────────────────────────────────────────────────────────────

def record_tokens(job_id: str, n_tokens: int) -> None:
    """Accumulates *n_tokens* in the ledger for *job_id*.

    Called automatically by :func:`run_training_step` when job_id is provided.
    """
    _token_ledger[job_id] = _token_ledger.get(job_id, 0) + n_tokens


def get_cumulative_spend(job_id: str) -> float:
    """Returns estimated cumulative USD spend for *job_id* based on recorded tokens.

    Uses Tinker's public rate of $0.40 per million tokens.
    """
    tokens = _token_ledger.get(job_id, 0)
    return round(tokens / 1_000_000 * _COST_PER_MILLION_TOKENS, 6)


# ── Job lifecycle helpers (cost_manager compatibility) ────────────────────────

def cancel_job(job_id: str) -> None:
    """Signals the training loop for *job_id* to stop at the next step boundary.

    Because Tinker training runs in-process (no remote job queue), cancellation
    is implemented as a module-level flag that :func:`run_training_loop` checks
    between batches.
    """
    _cancel_signals[job_id] = True


def is_cancelled(job_id: str) -> bool:
    """Returns True if :func:`cancel_job` has been called for *job_id*."""
    return _cancel_signals.get(job_id, False)
