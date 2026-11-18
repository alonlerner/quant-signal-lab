"""Tests for the 12-1 momentum signal, using small synthetic prices."""

import pandas as pd
import pytest
from conftest import make_prices, month_end_date

from signal_lab.strategies.momentum import Momentum

NAN = float("nan")

# 14 months starting 2020-01: index 0 = Jan 2020, index 13 = Feb 2021.
A = [100.0, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126]
B = [50.0, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37]


@pytest.fixture
def prices() -> pd.DataFrame:
    return make_prices({"A": A, "B": B})


def signal_at(signals: pd.DataFrame, prices: pd.DataFrame, month: str) -> pd.Series:
    return signals.loc[month_end_date(prices, month)]


def test_signal_is_return_from_12_months_ago_to_1_month_ago(prices):
    signals = Momentum().generate_signals(prices)

    # Feb 2021 (index 13): close of Jan 2021 (index 12) over close of Feb 2020 (index 1).
    at_feb = signal_at(signals, prices, "2021-02")
    assert at_feb["A"] == pytest.approx(124 / 102 - 1)
    assert at_feb["B"] == pytest.approx(38 / 49 - 1)

    # Jan 2021 (index 12): close of Dec 2020 (index 11) over close of Jan 2020 (index 0).
    at_jan = signal_at(signals, prices, "2021-01")
    assert at_jan["A"] == pytest.approx(122 / 100 - 1)


def test_signal_is_nan_until_there_is_enough_history(prices):
    signals = Momentum().generate_signals(prices)

    before_jan_2021 = signals.loc[:"2020-12-31"]
    assert before_jan_2021.isna().all().all()
    assert signals.loc["2021-01-01":].notna().all().all()


def test_shape_matches_the_wide_price_frame(prices):
    signals = Momentum().generate_signals(prices)

    assert list(signals.columns) == ["A", "B"]
    assert signals.index.equals(pd.DatetimeIndex(sorted(prices["date"].unique()), name="date"))


def test_most_recent_month_is_skipped(prices):
    base = signal_at(Momentum().generate_signals(prices), prices, "2021-02")

    # Changing any price in the signal day's own month must not matter.
    changed = prices.copy()
    in_feb = changed["date"].dt.to_period("M") == pd.Period("2021-02")
    changed.loc[in_feb, "adj_close"] *= 3
    after = signal_at(Momentum().generate_signals(changed), changed, "2021-02")

    pd.testing.assert_series_equal(after, base)


def test_only_the_two_reference_closes_matter(prices):
    base = signal_at(Momentum().generate_signals(prices), prices, "2021-02")

    # Mid-month prices in the reference months, and any month in between, are ignored.
    changed = prices.copy()
    last_days = changed.groupby(["ticker", changed["date"].dt.to_period("M")])["date"].transform(
        "max"
    )
    mid_month = changed["date"] != last_days
    changed.loc[mid_month, "adj_close"] *= 5
    after = signal_at(Momentum().generate_signals(changed), changed, "2021-02")

    pd.testing.assert_series_equal(after, base)

    # But the two reference closes themselves do matter.
    for reference_month in ["2020-02", "2021-01"]:
        moved = prices.copy()
        row = (moved["ticker"] == "A") & (moved["date"] == month_end_date(moved, reference_month))
        moved.loc[row, "adj_close"] *= 1.5
        result = signal_at(Momentum().generate_signals(moved), moved, "2021-02")
        assert result["A"] != pytest.approx(base["A"])
        assert result["B"] == pytest.approx(base["B"])


def test_signal_is_the_same_on_every_day_of_the_month(prices):
    signals = Momentum().generate_signals(prices)

    feb_2021 = signals.loc["2021-02"]
    assert len(feb_2021) > 1
    assert (feb_2021 == feb_2021.iloc[0]).all().all()


def test_ticker_that_lists_late_is_nan_until_it_has_12_months(prices):
    # C has no prices for the first 3 months (Jan-Mar 2020), so its first close is Apr 2020.
    late = [NAN, NAN, NAN, 10.0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    prices = make_prices({"A": A, "C": late})
    signals = Momentum().generate_signals(prices)

    # Valid once Apr 2020 (index 3) is the "12 months ago" month, i.e. in Apr 2021 (index 15).
    # The data only reaches Feb 2021, so C is still NaN everywhere while A is valid.
    assert signals["C"].isna().all()
    assert signals["A"].loc["2021-01":].notna().all()


def test_late_ticker_becomes_eligible_exactly_when_its_history_is_long_enough():
    months = 16  # Jan 2020 .. Apr 2021
    a = [100.0 + i for i in range(months)]
    c = [NAN, NAN, NAN] + [10.0 + i for i in range(months - 3)]  # first close: Apr 2020
    prices = make_prices({"A": a, "C": c})
    signals = Momentum().generate_signals(prices)

    assert pd.isna(signal_at(signals, prices, "2021-03")["C"])
    assert signal_at(signals, prices, "2021-04")["C"] == pytest.approx(
        c[14] / c[3] - 1  # close of Mar 2021 over close of Apr 2020
    )


def test_a_missing_reference_month_gives_nan_not_a_wrong_return():
    a = list(A)
    a[1] = NAN  # no prices at all in Feb 2020, which is the reference month for Feb 2021
    prices = make_prices({"A": a, "B": B})
    signals = Momentum().generate_signals(prices)

    at_feb = signal_at(signals, prices, "2021-02")
    assert pd.isna(at_feb["A"])
    assert at_feb["B"] == pytest.approx(38 / 49 - 1)


def test_a_gap_in_the_months_does_not_shift_the_lookback():
    # A has no data for Jun 2020 (index 5). Row-based shifting would be off by one month.
    a = list(A)
    a[5] = NAN
    prices = make_prices({"A": a})
    signals = Momentum().generate_signals(prices)

    assert signal_at(signals, prices, "2021-01")["A"] == pytest.approx(122 / 100 - 1)
    assert signal_at(signals, prices, "2021-02")["A"] == pytest.approx(124 / 102 - 1)


def test_custom_window():
    prices = make_prices({"A": A})
    signals = Momentum(lookback_months=3, skip_months=1).generate_signals(prices)

    # Apr 2020 (index 3): close of Mar (index 2) over close of Jan (index 0).
    assert signal_at(signals, prices, "2020-04")["A"] == pytest.approx(104 / 100 - 1)
    assert signals.loc[:"2020-03-31"].isna().all().all()


def test_longer_skip_uses_an_earlier_month_end():
    prices = make_prices({"A": A})
    signals = Momentum(lookback_months=4, skip_months=2).generate_signals(prices)

    # May 2020 (index 4): close of Mar (index 2) over close of Jan (index 0).
    assert signal_at(signals, prices, "2020-05")["A"] == pytest.approx(104 / 100 - 1)


def test_signal_on_a_date_does_not_change_when_later_data_is_removed(prices):
    # What the backtest engine relies on: the same answer with only the data up to t.
    full = Momentum().generate_signals(prices)

    for month in ["2021-01", "2021-02"]:
        t = month_end_date(prices, month)
        truncated = Momentum().generate_signals(prices[prices["date"] <= t])
        pd.testing.assert_series_equal(truncated.loc[t], full.loc[t])


@pytest.mark.parametrize(
    ("lookback", "skip"),
    [(12, 12), (1, 3), (12, 0), (12, -1)],
)
def test_invalid_windows_raise(lookback, skip):
    with pytest.raises(ValueError):
        Momentum(lookback_months=lookback, skip_months=skip)
