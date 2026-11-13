# quant-signal-lab

A research platform for testing equity trading signals.

**Available now:** a data pipeline that downloads daily adjusted prices and volume from yfinance to parquet, and a base interface for strategies.

**Planned:** a backtesting engine with look-ahead-bias tests, and a REST API for running and comparing experiments, built with FastAPI and PostgreSQL.

## Quick start with Docker

```bash
docker build -t quant-signal-lab .

# Sample download: AAPL and MSFT for 2024
docker run --rm -v "$PWD/data:/app/data" quant-signal-lab

# Choose your own tickers and dates (--end is inclusive)
docker run --rm -v "$PWD/data:/app/data" quant-signal-lab \
  signal-lab download --tickers AAPL MSFT NVDA --start 2020-01-01 --end 2024-12-31

# Run the tests
docker run --rm quant-signal-lab pytest
```

The `-v` mount keeps `data/prices.parquet` on your machine after the container exits. On Linux, the mounted directory must be writable by the container's user, so add `--user "$(id -u):$(id -g)"` if you get a permission error.

## Local development

Requires Python 3.13.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

pytest             # offline tests
pytest -m network  # test against the real yfinance API
signal-lab download --tickers AAPL MSFT --start 2024-01-01 --end 2024-12-31
```

## Dependencies

The Docker image installs exact versions from `requirements.lock`, so every build gets the same libraries. The lock pins versions but not file hashes, so it doesn't protect against a tampered package. Add `--generate-hashes` to the command below, and `--require-hashes` to the Dockerfile's `pip install`, if you need that. Generating hashes downloads every wheel for every platform, which can take hours on a slow connection. `pyproject.toml` only declares the ranges the project supports.

The lock is generated on Linux to match the image. After changing dependencies in `pyproject.toml`, or to pick up newer versions, regenerate it:

```bash
docker run --rm -v "$PWD:/src" -w /src python:3.13-slim sh -c \
  "pip install pip-tools && pip-compile --extra dev --strip-extras --upgrade \
   --output-file requirements.lock pyproject.toml"
```

Then rebuild the image and run `pytest -m network` inside it before committing the new lock, because the offline tests mock yfinance and won't catch a change in its output.

## Price data

`data/prices.parquet` has one row per ticker per day:

| Column | Type | Description |
|---|---|---|
| `date` | datetime | Trading day |
| `ticker` | string | Ticker symbol |
| `adj_close` | float | Close adjusted for splits and dividends |
| `volume` | float | Shares traded, adjusted for splits. Can be missing. |

The file is only written if every requested ticker has data and no ticker has the same date twice. Otherwise the command fails and any existing file is left untouched. The write is atomic: the new file is written under a temporary name and then renamed over the old one, so an interrupted run never leaves a partial file.

From Python:

```python
import pandas as pd
from signal_lab.data import download_and_save_prices, to_wide

download_and_save_prices(["AAPL", "MSFT"], "2024-01-01", "2024-12-31")

prices = pd.read_parquet("data/prices.parquet")
closes = to_wide(prices, "adj_close")  # dates as rows, tickers as columns
```

Keep these limitations in mind when using the data for research:

- **Dollar volume:** `adj_close` reflects dividends but `volume` doesn't, so `adj_close * volume` understates historical dollar volume for dividend payers.
- **Restated history:** adjusted prices are recalculated after each new split or dividend, so a later download can change past values.
- **Survivorship bias:** results depend on the tickers you choose. Picking companies that exist today leaves out those that were delisted.

## Adding a strategy

Create a module in `src/signal_lab/strategies/` with a subclass of `Strategy`. Nothing else needs to change.

```python
# src/signal_lab/strategies/momentum.py
import pandas as pd

from signal_lab.data import to_wide
from signal_lab.strategies.base import Strategy


class Momentum(Strategy):
    def __init__(self, lookback: int = 20) -> None:
        self.lookback = lookback

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        return to_wide(prices, "adj_close").pct_change(self.lookback)
```

## Project layout

```text
src/signal_lab/
  cli.py              signal-lab command
  data/prices.py      download, validate, save and reshape prices
  strategies/base.py  the Strategy interface
tests/                pytest suite
requirements.lock     pinned dependencies for the Docker image
data/                 downloaded data (not in git)
```
