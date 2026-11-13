"""The interface every strategy implements."""

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    """Turns prices into signals."""

    @abstractmethod
    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Compute signals from prices.

        Args:
            prices: Long-format frame with columns ``date, ticker, adj_close, volume``.
                Use ``signal_lab.data.to_wide`` to pivot the fields you need.

        Returns:
            A frame indexed by date with one column per ticker. A signal dated ``t``
            must only use data available at the close of ``t``.
        """
