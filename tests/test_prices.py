"""Tests for the price pipeline.

Most tests mock ``yf.download`` so they run offline. The ``network`` test checks the
same output against the real API, so the mock can't drift from it unnoticed.
"""

from pathlib import Path

import pandas as pd
import pytest

from signal_lab.data import (
    PriceDataError,
    download_and_save_prices,
    to_wide,
    validate_prices,
)

NAN = float("nan")
DATES = pd.date_range("2024-01-02", periods=3, freq="B", name="Date")
CLOSE = {"AAPL": [10.0, 11.0, 12.0], "MSFT": [20.0, 21.0, 22.0]}
VOLUME = {"AAPL": [100, 110, 120], "MSFT": [200, 210, 220]}


def make_raw(close: dict[str, list[float]], volume: dict[str, list[float]]) -> pd.DataFrame:
    """Build a frame shaped like ``yf.download`` output, with (Price, Ticker) columns."""
    columns = {("Open", ticker): values for ticker, values in close.items()}
    columns |= {("Close", ticker): values for ticker, values in close.items()}
    columns |= {("Volume", ticker): values for ticker, values in volume.items()}
    raw = pd.DataFrame(columns, index=DATES)
    raw.columns = pd.MultiIndex.from_tuples(raw.columns, names=["Price", "Ticker"])
    return raw.sort_index(axis=1)


@pytest.fixture
def mock_download(monkeypatch: pytest.MonkeyPatch):
    """Make ``yf.download`` return a given frame, and record the call's arguments."""

    def install(raw: pd.DataFrame) -> dict:
        calls: dict = {}

        def fake_download(tickers, **kwargs):
            calls.update(kwargs, tickers=tickers)
            return raw

        monkeypatch.setattr("signal_lab.data.prices.yf.download", fake_download)
        return calls

    return install


def assert_valid_output(path: Path, tickers: list[str]) -> pd.DataFrame:
    """The output has no duplicate dates for any ticker and no missing tickers."""
    saved = pd.read_parquet(path)
    assert not saved.duplicated(subset=["date", "ticker"]).any()
    assert set(saved["ticker"]) == set(tickers)
    return saved


def test_output_has_no_duplicate_dates_or_missing_tickers(mock_download, tmp_path):
    mock_download(make_raw(CLOSE, VOLUME))
    output = tmp_path / "data" / "prices.parquet"

    download_and_save_prices(["AAPL", "MSFT"], "2024-01-02", "2024-01-04", output)

    saved = assert_valid_output(output, ["AAPL", "MSFT"])
    assert len(saved) == 6


def test_output_keeps_each_tickers_own_values(mock_download, tmp_path):
    mock_download(make_raw(CLOSE, VOLUME))
    output = tmp_path / "prices.parquet"

    download_and_save_prices(["AAPL", "MSFT"], "2024-01-02", "2024-01-04", output)

    saved = pd.read_parquet(output)
    assert list(saved.columns) == ["date", "ticker", "adj_close", "volume"]
    for ticker in CLOSE:
        rows = saved[saved["ticker"] == ticker]
        assert rows["adj_close"].tolist() == CLOSE[ticker]
        assert rows["volume"].tolist() == VOLUME[ticker]


def test_ticker_with_no_data_raises_and_writes_no_file(mock_download, tmp_path):
    close = CLOSE | {"BAD": [NAN, NAN, NAN]}
    volume = VOLUME | {"BAD": [NAN, NAN, NAN]}
    mock_download(make_raw(close, volume))
    output = tmp_path / "prices.parquet"

    with pytest.raises(PriceDataError, match="BAD"):
        download_and_save_prices(["AAPL", "MSFT", "BAD"], "2024-01-02", "2024-01-04", output)

    assert not output.exists()


def test_failed_download_keeps_the_existing_file(mock_download, tmp_path):
    mock_download(pd.DataFrame())
    output = tmp_path / "prices.parquet"
    output.write_bytes(b"previous good file")

    with pytest.raises(PriceDataError, match="No price data"):
        download_and_save_prices(["AAPL"], "2024-01-02", "2024-01-04", output)

    assert output.read_bytes() == b"previous good file"


def test_interrupted_write_keeps_the_existing_file(mock_download, tmp_path, monkeypatch):
    mock_download(make_raw(CLOSE, VOLUME))
    output = tmp_path / "prices.parquet"
    output.write_bytes(b"previous good file")

    def failing_to_parquet(self, path, **kwargs):
        Path(path).write_bytes(b"partial")  # simulate a write cut off halfway
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", failing_to_parquet)

    with pytest.raises(OSError, match="disk full"):
        download_and_save_prices(["AAPL", "MSFT"], "2024-01-02", "2024-01-04", output)

    assert output.read_bytes() == b"previous good file"
    assert list(tmp_path.iterdir()) == [output]  # no temporary file left behind


def test_successful_write_replaces_the_existing_file(mock_download, tmp_path):
    mock_download(make_raw(CLOSE, VOLUME))
    output = tmp_path / "prices.parquet"
    output.write_bytes(b"old file")

    download_and_save_prices(["AAPL", "MSFT"], "2024-01-02", "2024-01-04", output)

    assert len(pd.read_parquet(output)) == 6
    assert list(tmp_path.iterdir()) == [output]


def test_validate_rejects_duplicate_rows():
    row = {"date": DATES[0], "ticker": "AAPL", "adj_close": 10.0, "volume": 100.0}
    prices = pd.DataFrame([row, row])

    with pytest.raises(PriceDataError, match="AAPL"):
        validate_prices(prices, ["AAPL"])


def test_end_date_is_inclusive(mock_download, tmp_path):
    calls = mock_download(make_raw(CLOSE, VOLUME))

    download_and_save_prices(
        ["AAPL", "MSFT"], "2024-01-02", "2024-01-04", tmp_path / "prices.parquet"
    )

    assert calls["start"] == "2024-01-02"
    assert calls["end"] == "2024-01-05"
    assert calls["auto_adjust"] is True


def test_missing_volume_keeps_the_row(mock_download, tmp_path):
    volume = VOLUME | {"AAPL": [100, NAN, 120]}
    mock_download(make_raw(CLOSE, volume))
    output = tmp_path / "prices.parquet"

    download_and_save_prices(["AAPL", "MSFT"], "2024-01-02", "2024-01-04", output)

    aapl = pd.read_parquet(output).query("ticker == 'AAPL'")
    assert len(aapl) == 3
    assert pd.isna(aapl["volume"].iloc[1])


def test_missing_price_drops_only_that_row(mock_download, tmp_path):
    close = CLOSE | {"MSFT": [NAN, 21.0, 22.0]}  # e.g. listed after the first date
    mock_download(make_raw(close, VOLUME))
    output = tmp_path / "prices.parquet"

    download_and_save_prices(["AAPL", "MSFT"], "2024-01-02", "2024-01-04", output)

    saved = pd.read_parquet(output)
    assert len(saved[saved["ticker"] == "AAPL"]) == 3
    assert saved.loc[saved["ticker"] == "MSFT", "adj_close"].tolist() == [21.0, 22.0]


def test_tickers_are_normalized(mock_download, tmp_path):
    calls = mock_download(make_raw(CLOSE, VOLUME))

    download_and_save_prices(
        ["aapl", "AAPL", " msft "], "2024-01-02", "2024-01-04", tmp_path / "prices.parquet"
    )

    assert calls["tickers"] == ["AAPL", "MSFT"]


@pytest.mark.parametrize(
    ("tickers", "start", "end"),
    [
        ([], "2024-01-02", "2024-01-04"),
        ("AAPL", "2024-01-02", "2024-01-04"),
        (["AAPL"], "2024-01-05", "2024-01-04"),
    ],
)
def test_invalid_input_raises(tickers, start, end, tmp_path):
    with pytest.raises(ValueError):
        download_and_save_prices(tickers, start, end, tmp_path / "prices.parquet")


def test_to_wide_returns_one_column_per_ticker(mock_download, tmp_path):
    mock_download(make_raw(CLOSE, VOLUME))
    prices = download_and_save_prices(
        ["AAPL", "MSFT"], "2024-01-02", "2024-01-04", tmp_path / "prices.parquet"
    )

    wide = to_wide(prices, "volume")

    assert list(wide.columns) == ["AAPL", "MSFT"]
    assert wide["MSFT"].tolist() == VOLUME["MSFT"]
    assert len(wide) == 3


@pytest.mark.network
def test_real_download_has_no_duplicate_dates_or_missing_tickers(tmp_path):
    tickers = ["AAPL", "MSFT"]
    output = tmp_path / "prices.parquet"

    download_and_save_prices(tickers, "2024-01-02", "2024-01-31", output)

    saved = assert_valid_output(output, tickers)
    assert list(saved.columns) == ["date", "ticker", "adj_close", "volume"]
    assert saved["date"].min() == pd.Timestamp("2024-01-02")
    assert saved["date"].max() == pd.Timestamp("2024-01-31")  # end date is inclusive
    assert saved["adj_close"].notna().all()
    assert saved["volume"].notna().all()
