"""Download daily adjusted prices and volume from yfinance and store them as parquet.

Prices are in long format, one row per ticker per day:

    date, ticker, adj_close, volume

Use ``to_wide`` to get one column per ticker for a single field.
"""

from collections.abc import Iterable
from datetime import date
from pathlib import Path
from uuid import uuid4

import pandas as pd
import yfinance as yf

DEFAULT_OUTPUT = Path("data/prices.parquet")
COLUMNS = ["date", "ticker", "adj_close", "volume"]


class PriceDataError(ValueError):
    """The downloaded prices failed validation."""


def _normalize_tickers(tickers: Iterable[str]) -> list[str]:
    """Uppercase and de-duplicate tickers, keeping their order."""
    if isinstance(tickers, str):
        raise ValueError(f"tickers must be a list of symbols, got the string {tickers!r}")
    normalized = list(dict.fromkeys(t.strip().upper() for t in tickers if t.strip()))
    if not normalized:
        raise ValueError("tickers must contain at least one symbol")
    return normalized


def _to_long(raw: pd.DataFrame) -> pd.DataFrame:
    """Reshape yfinance's (Price, Ticker) columns into the long schema."""
    if raw.empty:
        return pd.DataFrame(columns=COLUMNS)

    stacked = raw[["Close", "Volume"]].stack(level=1, future_stack=True)
    stacked.index.names = ["date", "ticker"]
    prices = stacked.reset_index().rename(columns={"Close": "adj_close", "Volume": "volume"})

    # A missing price means the ticker didn't trade that day. A missing volume alone
    # keeps the row, because the price is still usable.
    prices = prices.dropna(subset=["adj_close"]).astype({"volume": "float64"})
    prices = prices[COLUMNS].rename_axis(columns=None)
    return prices.sort_values(["ticker", "date"]).reset_index(drop=True)


def download_prices(
    tickers: Iterable[str], start: str | date, end: str | date
) -> pd.DataFrame:
    """Download daily prices for ``tickers`` from ``start`` to ``end``, both inclusive.

    The result is not validated. yfinance logs failed tickers rather than raising, so
    a failed ticker is simply absent here; ``validate_prices`` detects that.
    """
    symbols = _normalize_tickers(tickers)
    first, last = pd.Timestamp(start), pd.Timestamp(end)
    if first > last:
        raise ValueError(f"start ({first.date()}) is after end ({last.date()})")

    raw = yf.download(
        symbols,
        start=first.strftime("%Y-%m-%d"),
        # yfinance treats end as exclusive.
        end=(last + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=True,  # "Close" is then the adjusted close.
        actions=False,
        progress=False,
        multi_level_index=True,  # Same column shape for one ticker or many.
    )
    return _to_long(raw)


def validate_prices(prices: pd.DataFrame, tickers: Iterable[str]) -> None:
    """Raise ``PriceDataError`` if rows are empty, duplicated or missing a ticker.

    A date appears once per ticker in long format, so "no duplicate dates" means no
    ticker has the same date twice.
    """
    symbols = _normalize_tickers(tickers)

    missing_columns = [column for column in COLUMNS if column not in prices.columns]
    if missing_columns:
        raise PriceDataError(f"Missing columns: {missing_columns}")
    if prices.empty:
        raise PriceDataError(f"No price data returned for: {symbols}")

    duplicates = prices[prices.duplicated(subset=["date", "ticker"], keep=False)]
    if not duplicates.empty:
        raise PriceDataError(
            f"{len(duplicates)} rows share a (date, ticker) pair, "
            f"for tickers: {sorted(duplicates['ticker'].unique())}"
        )

    missing_tickers = sorted(set(symbols) - set(prices["ticker"]))
    if missing_tickers:
        raise PriceDataError(f"No price data for tickers: {missing_tickers}")


def save_prices(prices: pd.DataFrame, output: str | Path = DEFAULT_OUTPUT) -> Path:
    """Write prices to a parquet file, creating parent directories as needed.

    The file is written to a temporary name in the same directory and then renamed
    over the target. The rename is atomic, so an interrupted write never leaves a
    partial file and the previous file survives until the new one is complete.
    """
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    # A unique name so concurrent runs don't collide. Same directory so the rename
    # stays on one filesystem, which is what makes it atomic.
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        prices.to_parquet(temporary, index=False)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def download_and_save_prices(
    tickers: Iterable[str],
    start: str | date,
    end: str | date,
    output: str | Path = DEFAULT_OUTPUT,
) -> pd.DataFrame:
    """Download, validate and save prices. Bad data never overwrites an existing file."""
    symbols = _normalize_tickers(tickers)
    prices = download_prices(symbols, start, end)
    validate_prices(prices, symbols)
    save_prices(prices, output)
    return prices


def to_wide(prices: pd.DataFrame, field: str = "adj_close") -> pd.DataFrame:
    """Pivot one field to dates as rows and tickers as columns."""
    return prices.pivot(index="date", columns="ticker", values=field)
