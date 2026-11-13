"""Tests for the signal-lab command.

These replace ``download_and_save_prices`` inside the CLI module, so they test only
the CLI: argument parsing, what gets passed through, messages and exit codes.
"""

from pathlib import Path

import pandas as pd
import pytest

from signal_lab import cli
from signal_lab.data import DEFAULT_OUTPUT, PriceDataError

PRICES = pd.DataFrame(
    {
        "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-02"]),
        "ticker": ["AAPL", "AAPL", "MSFT"],
        "adj_close": [10.0, 11.0, 20.0],
        "volume": [100.0, 110.0, 200.0],
    }
)


@pytest.fixture
def fake_download(monkeypatch: pytest.MonkeyPatch):
    """Replace the CLI's download function, and record the arguments it receives."""

    def install(result: pd.DataFrame | Exception = PRICES) -> dict:
        calls: dict = {}

        def fake(tickers, start, end, output):
            calls.update(tickers=tickers, start=start, end=end, output=output)
            if isinstance(result, Exception):
                raise result
            return result

        # cli.py imports the function by name, so patch it where cli looks it up.
        monkeypatch.setattr(cli, "download_and_save_prices", fake)
        return calls

    return install


def test_download_passes_arguments_and_reports_success(fake_download, capsys):
    calls = fake_download()

    exit_code = cli.main(
        ["download", "--tickers", "AAPL", "MSFT", "--start", "2024-01-02", "--end", "2024-01-03"]
    )

    assert exit_code == 0
    assert calls == {
        "tickers": ["AAPL", "MSFT"],
        "start": "2024-01-02",
        "end": "2024-01-03",
        "output": DEFAULT_OUTPUT,
    }
    assert capsys.readouterr().out.strip() == (
        f"Saved 3 rows for 2 tickers to {DEFAULT_OUTPUT}"
    )


def test_download_uses_the_output_option(fake_download):
    calls = fake_download()

    cli.main(
        ["download", "--tickers", "AAPL", "--start", "2024-01-02", "--end", "2024-01-03",
         "--output", "elsewhere/prices.parquet"]
    )

    assert calls["output"] == Path("elsewhere/prices.parquet")


def test_data_error_exits_with_1_and_a_message(fake_download, capsys):
    fake_download(PriceDataError("No price data for tickers: ['BAD']"))

    exit_code = cli.main(
        ["download", "--tickers", "BAD", "--start", "2024-01-02", "--end", "2024-01-03"]
    )

    assert exit_code == 1
    assert capsys.readouterr().err.strip() == "Data error: No price data for tickers: ['BAD']"


def test_start_after_end_exits_with_1_and_a_message(tmp_path, capsys):
    # Uses the real function: the date check happens before any network call.
    exit_code = cli.main(
        ["download", "--tickers", "AAPL", "--start", "2024-01-05", "--end", "2024-01-04",
         "--output", str(tmp_path / "prices.parquet")]
    )

    assert exit_code == 1
    assert capsys.readouterr().err.startswith("Invalid input: start (2024-01-05) is after end")
    assert not (tmp_path / "prices.parquet").exists()


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["download"],
        ["download", "--start", "2024-01-02", "--end", "2024-01-03"],
        ["download", "--tickers", "AAPL", "--end", "2024-01-03"],
    ],
)
def test_missing_arguments_exit_with_usage_error(argv, capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(argv)

    assert exit_info.value.code == 2
    assert "usage: signal-lab" in capsys.readouterr().err
