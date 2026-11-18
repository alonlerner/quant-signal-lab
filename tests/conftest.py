"""Shared test helpers."""

import pandas as pd


def make_prices(
    month_end_closes: dict[str, list[float]],
    start: str = "2020-01",
    volume: float = 1000.0,
) -> pd.DataFrame:
    """Build long-format prices (``date, ticker, adj_close, volume``) from month-end closes.

    Each list gives one ticker's close on the last business day of consecutive months,
    beginning at ``start``. Other business days move in a straight line from the
    previous month-end close to this one, so every day has a distinct price. Use NaN
    for a month in which the ticker has no prices at all, for example before it listed.
    """
    rows = []
    for ticker, closes in month_end_closes.items():
        previous = float("nan")
        for month, close in zip(pd.period_range(start, periods=len(closes), freq="M"), closes):
            if pd.isna(close):
                previous = float("nan")
                continue
            days = pd.bdate_range(month.start_time, month.end_time.normalize())
            first = close if pd.isna(previous) else previous
            for day, price in zip(days, pd.Series(range(1, len(days) + 1)) / len(days)):
                rows.append((day, ticker, first + (close - first) * price, volume))
            previous = close
    prices = pd.DataFrame(rows, columns=["date", "ticker", "adj_close", "volume"])
    return prices.sort_values(["ticker", "date"]).reset_index(drop=True)


def month_end_date(prices: pd.DataFrame, month: str) -> pd.Timestamp:
    """The last trading day in ``prices`` for the calendar month, e.g. ``"2020-12"``."""
    dates = prices["date"]
    return dates[dates.dt.to_period("M") == pd.Period(month)].max()
