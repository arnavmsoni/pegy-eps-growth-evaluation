"""
CLI entrypoint: build quarterly dataset, train models, evaluate, interpret.

Run from repo root after ``pip install -e .``::

    python -m pegy_research.main --max-tickers 80

"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pegy_research.build_dataset import build_dataset
from pegy_research.diagnostics import print_dataset_diagnostics
from pegy_research.data_loader import (
    DEFAULT_CACHE_DIR,
    download_prices,
    load_sp500_tickers,
    project_data_dir,
    take_top_n_by_market_cap,
)
from pegy_research.evaluate import evaluate_all
from pegy_research.train_models import fit_all, time_based_split


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="When Does PEG Fail? – PEG reliability classification",
    )
    p.add_argument(
        "--years",
        type=int,
        default=5,
        help="Years of history (default 5)",
    )
    p.add_argument(
        "--max-tickers",
        type=int,
        default=100,
        help="Subset S&P 500 to top N by current market cap for speed (default 100)",
    )
    p.add_argument(
        "--peg-max",
        type=float,
        default=1.5,
        help="Train/evaluate only on rows with PEG below this (default 1.5)",
    )
    p.add_argument(
        "--train-fraction",
        type=float,
        default=0.7,
        help="Fraction of time-ordered rows used for training",
    )
    p.add_argument(
        "--refresh-prices",
        action="store_true",
        help="Bypass parquet cache for monthly prices",
    )
    p.add_argument(
        "--refresh-fundamentals",
        action="store_true",
        help="Re-download yfinance quarterly statements",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Tree figure and optional exports (default ./outputs)",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Override data cache directory (default <project>/data/cache)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cache = args.cache_dir or project_data_dir()
    cache.mkdir(parents=True, exist_ok=True)

    end = pd.Timestamp.utcnow().tz_localize(None).normalize()
    start = end - pd.DateOffset(years=args.years)
    # yfinance end is exclusive; extend for 12-month forward return
    price_end = end + pd.DateOffset(months=15)
    start_s, end_s = str(start.date()), str(price_end.date())

    all_syms = load_sp500_tickers(cache_dir=cache)
    if args.max_tickers and len(all_syms) > args.max_tickers:
        tickers = take_top_n_by_market_cap(
            all_syms, args.max_tickers, cache_dir=cache
        )
    else:
        tickers = all_syms

    need = list(dict.fromkeys(tickers + ["SPY"]))
    print(
        f"Downloading month-end prices {len(need)} symbols, {start_s} → {end_s} ...",
        flush=True,
    )
    prices = download_prices(
        need,
        start_s,
        end_s,
        interval="1mo",
        cache_dir=cache,
        refresh=args.refresh_prices,
    )
    if prices.empty:
        raise SystemExit("No price data returned. Check network or tickers.")

    if "SPY" not in prices.columns or prices["SPY"].dropna().empty:
        print("Refetching SPY alone (batch download sometimes drops SPY) ...", flush=True)
        spy_only = download_prices(
            ["SPY"],
            start_s,
            end_s,
            interval="1mo",
            cache_dir=cache,
            refresh=True,
        )
        if spy_only.empty or "SPY" not in spy_only.columns:
            raise SystemExit("Could not download SPY benchmark.")
        prices = prices.copy()
        prices["SPY"] = spy_only["SPY"]

    spy = prices["SPY"].dropna()
    stock_cols = [c for c in prices.columns if c != "SPY" and c in tickers]
    panel = prices[stock_cols].copy()

    if args.refresh_fundamentals:
        for f in cache.glob("statements_*.pkl"):
            f.unlink(missing_ok=True)
        for f in cache.glob("earnings_*.pkl"):
            f.unlink(missing_ok=True)

    print("Building dataset (quarterly, PEG < %s) ..." % args.peg_max, flush=True)
    df = build_dataset(
        tickers=stock_cols,
        price_panel=panel,
        spy=spy,
        start=start,
        end=end,
        peg_max=args.peg_max,
        cache_dir=cache,
    )
    if df.empty or len(df) < 30:
        print(
            "Dataset too small (%s rows). Try --max-tickers 150, --years 5, or check cache/network."
            % len(df)
        )
        if not df.empty:
            print(df.head())
        raise SystemExit(1)

    print(f"Rows in model (PEG < {args.peg_max}): {len(df)}", flush=True)
    print_dataset_diagnostics(df)

    split = time_based_split(
        df, time_col="quarter_end", label_col="label", train_fraction=args.train_fraction
    )
    models = fit_all(split)
    print(f"  Model feature set: {models.feature_names}", flush=True)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    X_test_mat = split.X_test[models.feature_names]
    evaluate_all(
        models,
        X_test_mat,
        split.y_test,
        split.test_rows,
        output_dir=out,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
