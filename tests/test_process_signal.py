from unittest.mock import MagicMock

import numpy as np
import pytest
import xarray as xr

from shot_tabular.backends import registry
from shot_tabular.main import TimeSettings, process_signal


def make_dataset(tmin: float = 0.0, tmax: float = 5.0, n: int = 200) -> xr.Dataset:
    time = np.linspace(tmin, tmax, n)
    da = xr.DataArray(np.sin(time), coords={"time": time}, dims=["time"])
    return xr.Dataset({"data": da})


@pytest.fixture
def mock_backend(monkeypatch):
    """Patch registry.get to return a controllable mock backend."""
    backend = MagicMock()
    monkeypatch.setattr(registry, "get", lambda _name: backend)
    return backend


@pytest.fixture
def ts():
    return TimeSettings(tmin=0.0, tmax=2.0, dt=0.5, method="linear")


# --- success path ---


def test_success_writes_parquet(tmp_path, mock_backend, ts):
    mock_backend.get.return_value = make_dataset()
    shot, msg = process_signal(100, {"ip": "AMC_IP"}, "uda", tmp_path, ts)
    assert (tmp_path / "100.parquet").exists()
    assert msg == ""


def test_success_returns_correct_shot_number(tmp_path, mock_backend, ts):
    mock_backend.get.return_value = make_dataset()
    shot, _ = process_signal(42, {"ip": "AMC_IP"}, "uda", tmp_path, ts)
    assert shot == 42


def test_parquet_has_expected_columns(tmp_path, mock_backend, ts):
    import pyarrow.parquet as pq

    mock_backend.get.return_value = make_dataset()
    process_signal(1, {"ip": "AMC_IP", "ne": "ANE_NE"}, "uda", tmp_path, ts)
    table = pq.read_table(tmp_path / "1.parquet")
    assert set(table.column_names) == {"shot", "time", "ip", "ne"}


def test_parquet_shot_column_has_correct_value(tmp_path, mock_backend, ts):
    import pyarrow.parquet as pq

    mock_backend.get.return_value = make_dataset()
    process_signal(99, {"ip": "AMC_IP"}, "uda", tmp_path, ts)
    table = pq.read_table(tmp_path / "99.parquet")
    assert all(v == 99 for v in table.column("shot").to_pylist())


def test_tmax_from_settings_limits_time_range(tmp_path, mock_backend):
    import pyarrow.parquet as pq

    mock_backend.get.return_value = make_dataset(tmax=10.0)
    ts = TimeSettings(tmin=0.0, tmax=2.0, dt=0.5, method="nearest")
    process_signal(1, {"ip": "IP"}, "uda", tmp_path, ts)
    table = pq.read_table(tmp_path / "1.parquet")
    times = table.column("time").to_pylist()
    assert max(times) < 3.0


def test_tmax_none_uses_signal_max_time(tmp_path, mock_backend):
    import pyarrow.parquet as pq

    mock_backend.get.return_value = make_dataset(tmax=4.0, n=400)
    ts = TimeSettings(tmin=0.0, tmax=None, dt=1.0, method="nearest")
    process_signal(2, {"ip": "IP"}, "uda", tmp_path, ts)
    table = pq.read_table(tmp_path / "2.parquet")
    times = table.column("time").to_pylist()
    assert max(times) >= 3.0


# --- error paths ---


def test_backend_exception_returns_error_no_parquet(tmp_path, mock_backend, ts):
    mock_backend.get.side_effect = RuntimeError("connection refused")
    shot, msg = process_signal(5, {"ip": "AMC_IP"}, "uda", tmp_path, ts)
    assert shot == 5
    assert "connection refused" in msg
    assert not (tmp_path / "5.parquet").exists()


def test_dataset_missing_data_variable_returns_error(tmp_path, mock_backend, ts):
    time = np.linspace(0, 5, 100)
    mock_backend.get.return_value = xr.Dataset(
        {"other": xr.DataArray(np.zeros(100), coords={"time": time}, dims=["time"])}
    )
    shot, msg = process_signal(6, {"ip": "AMC_IP"}, "uda", tmp_path, ts)
    assert shot == 6
    assert "no valid signals" in msg.lower()
    assert not (tmp_path / "6.parquet").exists()


def test_all_signals_fail_no_parquet_written(tmp_path, mock_backend, ts):
    mock_backend.get.side_effect = RuntimeError("unreachable")
    process_signal(7, {"ip": "IP", "ne": "NE"}, "uda", tmp_path, ts)
    assert not (tmp_path / "7.parquet").exists()


# --- partial success ---


def test_partial_success_writes_parquet_with_nan_column(tmp_path, ts):
    import pyarrow.parquet as pq

    ds = make_dataset()
    mock = MagicMock()
    mock.get.side_effect = [ds, RuntimeError("signal unavailable")]

    with pytest.MonkeyPatch.context() as m:
        m.setattr(registry, "get", lambda _name: mock)
        shot, msg = process_signal(8, {"ip": "IP", "ne": "NE"}, "uda", tmp_path, ts)

    assert (tmp_path / "8.parquet").exists()
    assert "partial" in msg.lower() or "warning" in msg.lower()

    df = pq.read_table(tmp_path / "8.parquet").to_pandas()
    assert df["ne"].isna().all()
    assert not df["ip"].isna().all()
