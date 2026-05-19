import importlib.util
import sys
import types
from pathlib import Path

import pytest


def _fake_torch_module():
    torch = types.ModuleType("torch")

    def no_grad():
        def decorator(func):
            return func

        return decorator

    torch.no_grad = no_grad
    return torch


def _fail_requests_get(*args, **kwargs):
    raise AssertionError("requests.get called")


@pytest.fixture
def prepare_module(monkeypatch):
    fake_requests = types.ModuleType("requests")
    fake_requests.RequestException = RuntimeError
    fake_requests.get = lambda *args, **kwargs: None

    fake_pyarrow = types.ModuleType("pyarrow")
    fake_parquet = types.ModuleType("pyarrow.parquet")
    fake_pyarrow.parquet = fake_parquet

    monkeypatch.setitem(sys.modules, "requests", fake_requests)
    monkeypatch.setitem(sys.modules, "pyarrow", fake_pyarrow)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", fake_parquet)
    monkeypatch.setitem(sys.modules, "rustbpe", types.ModuleType("rustbpe"))
    monkeypatch.setitem(sys.modules, "tiktoken", types.ModuleType("tiktoken"))
    monkeypatch.setitem(sys.modules, "torch", _fake_torch_module())

    module_path = Path(__file__).resolve().parents[1] / "autoresearch" / "prepare.py"
    module_name = "legacy_autoresearch_prepare_for_tests"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("env_var", ["NO_SPEND", "DATA_GENERATOR_OFFLINE"])
def test_download_data_offline_guard_fails_when_required_shards_missing(
    tmp_path, monkeypatch, prepare_module, env_var
):
    monkeypatch.setenv(env_var, "1")
    monkeypatch.setattr(prepare_module, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(prepare_module.requests, "get", _fail_requests_get)

    with pytest.raises(RuntimeError) as excinfo:
        prepare_module.download_data(num_shards=1, download_workers=1)

    message = str(excinfo.value)
    assert f"{env_var}=1" in message
    assert "Data download disabled" in message
    assert "missing required cached shards" in message
    assert "shard_00000.parquet" in message
    assert "shard_06542.parquet" in message


@pytest.mark.parametrize("env_var", ["NO_SPEND", "DATA_GENERATOR_OFFLINE"])
def test_download_data_offline_guard_allows_fully_cached_shards(
    tmp_path, monkeypatch, capsys, prepare_module, env_var
):
    monkeypatch.setenv(env_var, "1")
    monkeypatch.setattr(prepare_module, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(prepare_module.requests, "get", _fail_requests_get)

    for index in (0, 1, prepare_module.VAL_SHARD):
        (tmp_path / prepare_module._shard_filename(index)).write_bytes(b"cached")

    prepare_module.download_data(num_shards=2, download_workers=1)

    assert "all 3 shards already downloaded" in capsys.readouterr().out
