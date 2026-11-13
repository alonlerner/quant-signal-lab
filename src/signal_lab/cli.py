"""Command-line entry point: argument parsing only, the work lives in the packages."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from signal_lab.data import DEFAULT_OUTPUT, PriceDataError, download_and_save_prices


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="signal-lab")
    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser(
        "download", help="Download daily adjusted prices and volume to a parquet file."
    )
    download.add_argument("--tickers", nargs="+", required=True, help="e.g. AAPL MSFT")
    download.add_argument("--start", required=True, help="First date, YYYY-MM-DD")
    download.add_argument("--end", required=True, help="Last date (inclusive), YYYY-MM-DD")
    download.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output path (default: {DEFAULT_OUTPUT})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        prices = download_and_save_prices(args.tickers, args.start, args.end, args.output)
    except ValueError as error:  # includes PriceDataError
        kind = "Data error" if isinstance(error, PriceDataError) else "Invalid input"
        print(f"{kind}: {error}", file=sys.stderr)
        return 1

    print(f"Saved {len(prices)} rows for {prices['ticker'].nunique()} tickers to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
