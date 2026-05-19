"""Tests for Tinker SDK Wrapper (owner: Sid Potti)

All tests mock the `tinker` package so the suite runs without the SDK
installed (CI / dev environments).
"""

import sys
import types as builtin_types
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers — build a minimal fake tinker package
# ---------------------------------------------------------------------------

def _make_tinker_mock():
    """Returns (tinker_mod, tinker_types_mod) MagicMocks that behave like the real SDK."""
    tinker_mod = MagicMock(name="tinker")
    types_mod = MagicMock(name="tinker.types")

    # Simulate Datum / ModelInput / AdamParams construction
    types_mod.Datum = MagicMock(side_effect=lambda **kw: MagicMock(**kw))
    types_mod.ModelInput.from_ints = MagicMock(
        side_effect=lambda tokens: MagicMock(tokens=tokens)
    )
    types_mod.AdamParams = MagicMock(side_effect=lambda **kw: MagicMock(**kw))
    types_mod.SamplingParams = MagicMock(side_effect=lambda **kw: MagicMock(**kw))

    tinker_mod.types = types_mod
    return tinker_mod, types_mod


def _install_tinker_mock(tinker_mod, types_mod):
    sys.modules["tinker"] = tinker_mod
    sys.modules["tinker.types"] = types_mod


def _remove_tinker_mock():
    sys.modules.pop("tinker", None)
    sys.modules.pop("tinker.types", None)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def fresh_tinker_state(monkeypatch):
    """Each test gets a clean tinker mock and a fresh import of the wrapper module."""
    monkeypatch.delenv("NO_SPEND", raising=False)
    monkeypatch.delenv("TINKER_BACKEND", raising=False)

    tinker_mod, types_mod = _make_tinker_mock()
    _install_tinker_mock(tinker_mod, types_mod)

    # Force re-import so the module picks up the mocked tinker
    sys.modules.pop("src.tinker_api.tinker_api", None)

    yield tinker_mod, types_mod

    _remove_tinker_mock()
    sys.modules.pop("src.tinker_api.tinker_api", None)


def _api():
    """Lazily import the module under test (after mocks are installed)."""
    import importlib
    return importlib.import_module("src.tinker_api.tinker_api")


# ---------------------------------------------------------------------------
# create_service_client
# ---------------------------------------------------------------------------

def test_create_service_client_returns_client(fresh_tinker_state):
    tinker_mod, _ = fresh_tinker_state
    api = _api()

    client = api.create_service_client()
    tinker_mod.ServiceClient.assert_called_once()
    assert client is tinker_mod.ServiceClient.return_value


def test_create_service_client_wraps_sdk_error(fresh_tinker_state):
    tinker_mod, _ = fresh_tinker_state
    tinker_mod.ServiceClient.side_effect = RuntimeError("auth failed")
    api = _api()

    with pytest.raises(api.TinkerAPIError, match="auth failed"):
        api.create_service_client()


# ---------------------------------------------------------------------------
# create_lora_training_client
# ---------------------------------------------------------------------------

def test_create_lora_training_client_passes_args(fresh_tinker_state):
    tinker_mod, _ = fresh_tinker_state
    service_client = tinker_mod.ServiceClient.return_value
    api = _api()

    training_client = api.create_lora_training_client(
        service_client, base_model="meta-llama/Llama-3.2-1B", rank=16
    )
    service_client.create_lora_training_client.assert_called_once_with(
        base_model="meta-llama/Llama-3.2-1B", rank=16
    )
    assert training_client is service_client.create_lora_training_client.return_value


# ---------------------------------------------------------------------------
# make_datum
# ---------------------------------------------------------------------------

def test_make_datum_causal_lm_shift(fresh_tinker_state):
    _, types_mod = fresh_tinker_state
    api = _api()

    tokens = [1, 2, 3, 4, 5]
    datum = api.make_datum(tokens)

    # ModelInput should be called with tokens[:-1]
    types_mod.ModelInput.from_ints.assert_called_once_with([1, 2, 3, 4])
    # Datum should receive loss_fn_inputs with target=tokens[1:]
    call_kwargs = types_mod.Datum.call_args.kwargs
    assert call_kwargs["loss_fn_inputs"]["target_tokens"] == [2, 3, 4, 5]
    assert len(call_kwargs["loss_fn_inputs"]["weights"]) == 4


def test_make_datum_explicit_targets(fresh_tinker_state):
    _, types_mod = fresh_tinker_state
    api = _api()

    api.make_datum([10, 20], target_tokens=[20, 30], weights=[0.5, 0.5])
    types_mod.ModelInput.from_ints.assert_called_once_with([10, 20])
    call_kwargs = types_mod.Datum.call_args.kwargs
    assert call_kwargs["loss_fn_inputs"]["target_tokens"] == [20, 30]
    assert call_kwargs["loss_fn_inputs"]["weights"] == [0.5, 0.5]


# ---------------------------------------------------------------------------
# run_training_step
# ---------------------------------------------------------------------------

def _make_training_client(types_mod):
    """Returns a mock training_client whose forward_backward/optim_step chain works."""
    tc = MagicMock()
    fwdbwd_result = MagicMock(loss=0.42)
    tc.forward_backward.return_value.result.return_value = fwdbwd_result
    tc.optim_step.return_value.result.return_value = MagicMock()
    return tc


def _make_batch(types_mod, n_batches=2, tokens_each=10):
    """Returns a list of Datum-like mocks with .model_input.tokens attributes."""
    batch = []
    for _ in range(n_batches):
        datum = MagicMock()
        datum.model_input.tokens = list(range(tokens_each))
        batch.append(datum)
    return batch


def test_run_training_step_calls_sdk_methods(fresh_tinker_state):
    _, types_mod = fresh_tinker_state
    api = _api()
    tc = _make_training_client(types_mod)
    batch = _make_batch(types_mod)

    result = api.run_training_step(tc, batch, learning_rate=1e-4)

    tc.forward_backward.assert_called_once_with(batch, loss_fn="cross_entropy")
    tc.optim_step.assert_called_once()
    assert result["tokens_processed"] == 20   # 2 datums × 10 tokens
    assert result["loss"] == pytest.approx(0.42)


def test_run_training_step_records_tokens_with_job_id(fresh_tinker_state):
    _, types_mod = fresh_tinker_state
    api = _api()
    tc = _make_training_client(types_mod)
    batch = _make_batch(types_mod, n_batches=1, tokens_each=5)

    api.run_training_step(tc, batch, job_id="job-xyz")
    assert api._token_ledger.get("job-xyz", 0) == 5


def test_run_training_step_wraps_sdk_error(fresh_tinker_state):
    _, types_mod = fresh_tinker_state
    api = _api()
    tc = MagicMock()
    tc.forward_backward.side_effect = RuntimeError("gpu oom")

    with pytest.raises(api.TinkerAPIError, match="gpu oom"):
        api.run_training_step(tc, [])


# ---------------------------------------------------------------------------
# Token ledger / cost tracking
# ---------------------------------------------------------------------------

def test_record_tokens_accumulates(fresh_tinker_state):
    api = _api()
    api.record_tokens("job-1", 500_000)
    api.record_tokens("job-1", 500_000)
    assert api._token_ledger["job-1"] == 1_000_000


def test_get_cumulative_spend_zero_for_unknown_job(fresh_tinker_state):
    api = _api()
    assert api.get_cumulative_spend("unknown-job") == 0.0


def test_get_cumulative_spend_one_million_tokens(fresh_tinker_state):
    api = _api()
    api.record_tokens("job-2", 1_000_000)
    spend = api.get_cumulative_spend("job-2")
    assert spend == pytest.approx(0.40, rel=1e-4)


def test_get_cumulative_spend_returns_float(fresh_tinker_state):
    api = _api()
    api.record_tokens("job-3", 250_000)
    assert isinstance(api.get_cumulative_spend("job-3"), float)


# ---------------------------------------------------------------------------
# cancel_job / is_cancelled
# ---------------------------------------------------------------------------

def test_cancel_job_sets_signal(fresh_tinker_state):
    api = _api()
    assert not api.is_cancelled("job-a")
    api.cancel_job("job-a")
    assert api.is_cancelled("job-a")


def test_cancel_job_does_not_affect_other_jobs(fresh_tinker_state):
    api = _api()
    api.cancel_job("job-b")
    assert not api.is_cancelled("job-c")


# ---------------------------------------------------------------------------
# run_training_loop
# ---------------------------------------------------------------------------

def test_run_training_loop_runs_all_steps(fresh_tinker_state):
    _, types_mod = fresh_tinker_state
    api = _api()
    tc = _make_training_client(types_mod)
    batches = [_make_batch(types_mod, n_batches=1, tokens_each=4) for _ in range(3)]

    result = api.run_training_loop(tc, batches, job_id="loop-job")
    assert result["steps"] == 3
    assert result["tokens"] == 12
    assert result["cancelled"] is False


def test_run_training_loop_stops_on_cancel(fresh_tinker_state):
    _, types_mod = fresh_tinker_state
    api = _api()
    tc = _make_training_client(types_mod)

    # Cancel after the first step by side-effecting via a counter
    call_count = {"n": 0}
    original_step = api.run_training_step

    def _step_with_cancel(tc_, batch, **kw):
        call_count["n"] += 1
        r = original_step(tc_, batch, **kw)
        if call_count["n"] == 1:
            api.cancel_job("cancel-loop")
        return r

    batches = [_make_batch(types_mod) for _ in range(5)]

    with patch.object(api, "run_training_step", side_effect=_step_with_cancel):
        result = api.run_training_loop(tc, batches, job_id="cancel-loop")

    assert result["steps"] == 1
    assert result["cancelled"] is True


# ---------------------------------------------------------------------------
# save_checkpoint / save_weights
# ---------------------------------------------------------------------------

def test_save_checkpoint_returns_sampling_client(fresh_tinker_state):
    api = _api()
    tc = MagicMock()
    sc = api.save_checkpoint(tc, name="my-ckpt")
    tc.save_weights_and_get_sampling_client.assert_called_once_with(name="my-ckpt")
    assert sc is tc.save_weights_and_get_sampling_client.return_value


def test_save_weights_calls_save_for_sampler(fresh_tinker_state):
    api = _api()
    tc = MagicMock()
    api.save_weights(tc, name="v1")
    tc.save_weights_for_sampler.assert_called_once_with(name="v1")


# ---------------------------------------------------------------------------
# NO_SPEND / dry-run guards
# ---------------------------------------------------------------------------

def _assert_live_tinker_call_blocked(api, call, reason):
    with pytest.raises(api.TinkerAPIError) as excinfo:
        call()
    message = str(excinfo.value)
    assert "live Tinker API calls are disabled" in message
    assert reason in message


@pytest.mark.parametrize(
    ("env_name", "env_value", "reason"),
    [
        ("NO_SPEND", "1", "NO_SPEND=1"),
        ("NO_SPEND", "true", "NO_SPEND=1"),
        ("NO_SPEND", "on", "NO_SPEND=1"),
        ("TINKER_BACKEND", "dry_run", "TINKER_BACKEND=dry_run"),
    ],
)
def test_no_spend_guards_block_sdk_helpers_before_construction_or_use(
    fresh_tinker_state, monkeypatch, env_name, env_value, reason
):
    tinker_mod, types_mod = fresh_tinker_state
    api = _api()
    monkeypatch.setenv(env_name, env_value)

    _assert_live_tinker_call_blocked(api, api.create_service_client, reason)
    tinker_mod.ServiceClient.assert_not_called()

    service_client = MagicMock()
    _assert_live_tinker_call_blocked(
        api,
        lambda: api.create_lora_training_client(
            service_client, base_model="meta-llama/Llama-3.2-1B", rank=16
        ),
        reason,
    )
    service_client.create_lora_training_client.assert_not_called()

    training_client = _make_training_client(types_mod)
    _assert_live_tinker_call_blocked(
        api, lambda: api.get_tokenizer(training_client), reason
    )
    training_client.get_tokenizer.assert_not_called()

    _assert_live_tinker_call_blocked(api, lambda: api.make_datum([1, 2, 3]), reason)
    types_mod.ModelInput.from_ints.assert_not_called()
    types_mod.Datum.assert_not_called()

    batch = _make_batch(types_mod, n_batches=1, tokens_each=3)
    _assert_live_tinker_call_blocked(
        api,
        lambda: api.run_training_step(
            training_client, batch, learning_rate=1e-4, job_id="blocked-job"
        ),
        reason,
    )
    training_client.forward_backward.assert_not_called()
    training_client.optim_step.assert_not_called()
    types_mod.AdamParams.assert_not_called()
    assert "blocked-job" not in api._token_ledger

    _assert_live_tinker_call_blocked(
        api, lambda: api.save_checkpoint(training_client, name="ckpt"), reason
    )
    training_client.save_weights_and_get_sampling_client.assert_not_called()

    _assert_live_tinker_call_blocked(
        api, lambda: api.save_weights(training_client, name="weights"), reason
    )
    training_client.save_weights_for_sampler.assert_not_called()

    sampling_client = MagicMock()
    _assert_live_tinker_call_blocked(
        api, lambda: api.sample(sampling_client, prompt_tokens=[1, 2, 3]), reason
    )
    sampling_client.sample.assert_not_called()
    types_mod.ModelInput.from_ints.assert_not_called()
    types_mod.SamplingParams.assert_not_called()


@pytest.mark.parametrize(
    ("env_name", "env_value"),
    [
        ("NO_SPEND", "1"),
        ("NO_SPEND", "true"),
        ("TINKER_BACKEND", "dry_run"),
    ],
)
def test_no_spend_guards_allow_ledger_and_cancel_helpers(
    fresh_tinker_state, monkeypatch, env_name, env_value
):
    api = _api()
    monkeypatch.setenv(env_name, env_value)

    api.record_tokens("guarded-job", 250_000)
    api.record_tokens("guarded-job", 250_000)
    assert api.get_cumulative_spend("guarded-job") == pytest.approx(0.20)

    api.cancel_job("guarded-job")
    assert api.is_cancelled("guarded-job")


# ---------------------------------------------------------------------------
# TinkerAPIError when tinker not available
# ---------------------------------------------------------------------------

def test_tinker_api_error_when_not_installed(fresh_tinker_state):
    """If tinker import failed, every SDK call should raise TinkerAPIError."""
    # Temporarily hide tinker from the mock state
    sys.modules.pop("tinker", None)
    sys.modules.pop("tinker.types", None)
    sys.modules.pop("src.tinker_api.tinker_api", None)

    # Re-import with _TINKER_AVAILABLE = False
    import importlib
    import unittest.mock as mock

    with mock.patch.dict(sys.modules, {"tinker": None, "tinker.types": None}):
        sys.modules.pop("src.tinker_api.tinker_api", None)
        api = importlib.import_module("src.tinker_api.tinker_api")
        # _TINKER_AVAILABLE should be False because import was None
        # Force the flag for this test
        api._TINKER_AVAILABLE = False

        with pytest.raises(api.TinkerAPIError):
            api.create_service_client()
        with pytest.raises(api.TinkerAPIError):
            api.make_datum([1, 2, 3])
