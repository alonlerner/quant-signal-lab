"""12-1 momentum: the return from 12 months ago to 1 month ago, skipping the latest month."""

import pandas as pd

from signal_lab.data import to_wide
from signal_lab.strategies.base import Strategy


class Momentum(Strategy):
    """Classic cross-sectional momentum (Jegadeesh-Titman, Fama-French UMD).

    For a date in calendar month ``M`` the signal is::

        close(end of M - skip_months) / close(end of M - lookback_months) - 1

    where "close at the end of a month" is that month's last available price. With
    the defaults (12, 1) the most recent month is skipped, because returns over the
    last month tend to reverse instead of continue.

    ``skip_months`` must be at least 1. That way the signal only depends on completed
    months: it is the same on every day of month ``M`` and never uses a price from
    month ``M`` itself, so it is safe to read at any date, in particular on the last
    trading day of a month.

    A ticker gets NaN until it has a price in both reference months.
    """

    def __init__(self, lookback_months: int = 12, skip_months: int = 1) -> None:
        # skip_months=0 would use the current month's close, which on any day before the
        # month's last trading day is a future price.
        if skip_months < 1:
            raise ValueError(f"skip_months must be >= 1, got {skip_months}")
        if lookback_months <= skip_months:
            raise ValueError(
                f"lookback_months ({lookback_months}) must be greater than "
                f"skip_months ({skip_months})"
            )
        self.lookback_months = lookback_months
        self.skip_months = skip_months

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        closes = to_wide(prices, "adj_close").sort_index()
        months = closes.index.to_period("M")

        # Last available price of each month (NaN gaps are skipped), on a complete
        # monthly grid so that shifting by n rows means shifting by n calendar months.
        month_end = closes.groupby(months).last()
        month_end = month_end.reindex(pd.period_range(month_end.index[0], month_end.index[-1]))

        signal = month_end.shift(self.skip_months) / month_end.shift(self.lookback_months) - 1

        # Every day takes the value of its own month.
        daily = signal.reindex(months)
        daily.index = closes.index
        return daily
