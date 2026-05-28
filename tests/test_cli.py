import argparse
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from shot_tabular.main import StoreKeyValue, main

# --- StoreKeyValue ---


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        dest="signals",
        action=StoreKeyValue,
        nargs="+",
        metavar=("KEY", "VALUE"),
    )
    return parser


def test_store_key_value_alias_and_signal():
    args = _make_parser().parse_args(["-s", "ip", "AMC_IP"])
    assert args.signals == {"ip": "AMC_IP"}


def test_store_key_value_signal_only_uses_signal_as_alias():
    args = _make_parser().parse_args(["-s", "ip"])
    assert args.signals == {"ip": "ip"}


def test_store_key_value_multiple_flags_accumulate():
    args = _make_parser().parse_args(["-s", "ip", "AMC_IP", "-s", "ne", "ANE_NE"])
    assert args.signals == {"ip": "AMC_IP", "ne": "ANE_NE"}


# --- main() shot-source tests ---


def _run_main(argv: list[str]) -> list[int]:
    """Run main() with patched argv; returns the shots passed to the pool."""
    captured: dict = {}

    def fake_imap(fn, shots):
        captured["shots"] = list(shots)
        return iter([])

    mock_pool = MagicMock()
    mock_pool.__enter__ = MagicMock(return_value=mock_pool)
    mock_pool.__exit__ = MagicMock(return_value=False)
    mock_pool.imap_unordered.side_effect = fake_imap

    with patch("sys.argv", ["st", *argv]):
        with patch("shot_tabular.main.mp.Pool", return_value=mock_pool):
            with patch("shot_tabular.main.gather_results", return_value=(0, 0, 0)):
                main()

    return captured.get("shots", [])


def test_shots_from_explicit_list(tmp_path):
    shots = _run_main(["--shots", "10", "20", "30", "-s", "ip", "-o", str(tmp_path)])
    assert shots == [10, 20, 30]


def test_shots_from_range(tmp_path):
    shots = _run_main(
        ["--shot-min", "5", "--shot-max", "8", "-s", "ip", "-o", str(tmp_path)]
    )
    assert shots == [5, 6, 7, 8]


def test_shots_from_csv(tmp_path):
    csv_path = tmp_path / "shots.csv"
    pd.DataFrame({"shot": [100, 200, 300]}).to_csv(csv_path, index=False)
    shots = _run_main(["--shot-file", str(csv_path), "-s", "ip", "-o", str(tmp_path)])
    assert shots == [100, 200, 300]


def test_shots_from_parquet(tmp_path):
    parquet_path = tmp_path / "shots.parquet"
    df = pd.DataFrame({"shot": [10, 20, 30]})
    df.index = pd.Index([10, 20, 30])
    df.to_parquet(parquet_path)
    shots = _run_main(
        ["--shot-file", str(parquet_path), "-s", "ip", "-o", str(tmp_path)]
    )
    assert shots == [10, 20, 30]


def test_invalid_shot_file_format_exits(tmp_path):
    bad_file = tmp_path / "shots.txt"
    bad_file.write_text("100\n200\n")
    with patch("sys.argv", ["st", "--shot-file", str(bad_file), "-s", "ip"]):
        with pytest.raises(SystemExit):
            main()


def test_no_shots_specified_exits():
    with patch("sys.argv", ["st", "-s", "ip"]):
        with pytest.raises(SystemExit):
            main()


def test_output_folder_is_created(tmp_path):
    out = tmp_path / "nested" / "output"
    _run_main(["--shots", "1", "-s", "ip", "-o", str(out)])
    assert out.exists()
