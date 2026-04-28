"""
Download and cache market data from yfinance.

S&P 500 universe, monthly (or daily) prices, and quarterly statements for
point-in-time feature construction. Caches fundamentals on disk to limit
repeated API calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

import yfinance as yf

# Project root: parent of src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "cache"


def project_data_dir() -> Path:
    return DEFAULT_CACHE_DIR


def load_sp500_tickers(cache_dir: Optional[Path] = None) -> list[str]:
    """
    Load S&P 500 ticker symbols from Wikipedia (same constituents list
    as the public S&P 500 table on Wikipedia).
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "sp500_tickers.json"
    if path.is_file():
        with open(path, encoding="utf-8") as f:
            return list(json.load(f))

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url, storage_options={"User-Agent": "pegy-research/1.0"})
    tickers: list[str] = []
    for t in tables:
        if "Symbol" in t.columns:
            tickers = t["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
            break
    if not tickers:
        raise RuntimeError("Could not parse S&P 500 table from Wikipedia")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(tickers, f, indent=0)
    return tickers


def take_top_n_by_market_cap(
    tickers: list[str], n: int, cache_dir: Optional[Path] = None
) -> list[str]:
    """
    Sort by current market cap (yfinance `info`) and return top n.
    Used to speed up research runs; not a perfect point-in-time cap screen.
    """
    if n >= len(tickers):
        return tickers
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    cap_path = cache_dir / "mcap_order.json"
    if cap_path.is_file():
        with open(cap_path, encoding="utf-8") as f:
            ordered = json.load(f)
        if isinstance(ordered, list) and len(ordered) >= n:
            return [s for s in ordered if s in tickers][:n]

    caps: list[tuple[str, float]] = []
    for sym in tickers:
        try:
            info = yf.Ticker(sym).info
            m = info.get("marketCap")
            if m is not None and not (isinstance(m, float) and np.isnan(m)):
                caps.append((sym, float(m)))
        except Exception:
            continue
    caps.sort(key=lambda x: -x[1])
    ordered = [c[0] for c in caps]
    with open(cap_path, "w", encoding="utf-8") as f:
        json.dump(ordered, f, indent=0)
    return [s for s in ordered if s in tickers][:n]


def download_prices(
    tickers: list[str],
    start: str,
    end: str,
    *,
    interval: str = "1mo",
    cache_dir: Optional[Path] = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """
    Adj-close panel (columns = tickers, rows = month-end timestamps).

    Uses auto-adjusted OHLC so splits/dividends are reflected.
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    import hashlib

    key = "|".join(sorted(tickers)) + f"|{len(tickers)}|{start}|{end}|{interval}"
    h = hashlib.sha256(key.encode()).hexdigest()[:16]
    cache_file = cache_dir / f"prices_{h}.parquet"

    if cache_file.is_file() and not refresh:
        return pd.read_parquet(cache_file)

    df = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        actions=False,
        progress=False,
        threads=True,
        group_by="column",
    )
    if df.empty:
        return pd.DataFrame()

    # Multiple tickers: columns MultiIndex ('Close', ticker); single: Series or flat cols
    if isinstance(df.columns, pd.MultiIndex):
        close = df["Close"].copy()
    elif "Close" in df.columns:
        c = df["Close"]
        close = c.to_frame(name=tickers[0]) if isinstance(c, pd.Series) else c
    else:
        close = pd.DataFrame()

    close = close.sort_index()
    close.to_parquet(cache_file)
    return close


def fetch_quarterly_statements(
    ticker: str,
    cache_dir: Optional[Path] = None,
    refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Quarterly income statement and balance sheet as returned by yfinance.

    Rows are line items; columns are fiscal period-end timestamps (typically
    quarter-end dates). Cached as JSON is fragile for DataFrames — cache pickle.
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    pkl = cache_dir / f"statements_{ticker.replace('/', '_')}.pkl"

    if pkl.is_file() and not refresh:
        return pd.read_pickle(pkl)  # noqa: S301

    t = yf.Ticker(ticker)
    out = {
        "income": _as_frame(getattr(t, "quarterly_income_stmt", None)),
        "balance": _as_frame(getattr(t, "quarterly_balance_sheet", None)),
    }
    import pickle as _pkl

    with open(pkl, "wb") as f:
        _pkl.dump(out, f)
    return out


def _as_frame(x: Any) -> pd.DataFrame:
    if x is None:
        return pd.DataFrame()
    if isinstance(x, pd.DataFrame):
        return x
    return pd.DataFrame()


def fetch_earnings_history(
    ticker: str,
    cache_dir: Optional[Path] = None,
    refresh: bool = False,
    limit: int = 60,
) -> pd.DataFrame:
    """
    Historical **quarterly** reported EPS with earnings announcement dates.

    ``get_earnings_dates`` usually has a longer lookback than
    ``quarterly_income_stmt`` (5–6 columns on the free API), which is required
    to form TTM and YoY growth on a 5-year price window.
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    pkl = cache_dir / f"earnings_{ticker.replace('/', '_')}.pkl"
    if pkl.is_file() and not refresh:
        return pd.read_pickle(pkl)  # noqa: S301

    t = yf.Ticker(ticker)
    d: Optional[pd.DataFrame] = None
    if hasattr(t, "get_earnings_dates"):
        try:
            d = t.get_earnings_dates(limit=limit)  # type: ignore[call-arg]
        except Exception:
            d = None
    if d is None or d.empty:
        d = pd.DataFrame()
    else:
        d = d.copy()
    import pickle as _pkl

    with open(pkl, "wb") as f:
        _pkl.dump(d, f)
    return d


def load_spy_monthly(
    start: str, end: str, cache_dir: Optional[Path] = None, refresh: bool = False
) -> pd.Series:
    """SPY adjusted close, month-end series (name: SPY)."""
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    f = cache_dir / f"spy_m_{start}_{end}.parquet"
    if f.is_file() and not refresh:
        s = pd.read_parquet(f)
        return s.iloc[:, 0]

    s = download_prices(
        ["SPY"], start, end, interval="1mo", cache_dir=cache_dir, refresh=refresh
    )
    if s.empty:
        return pd.Series(dtype=float, name="SPY")
    if "SPY" in s.columns:
        out = s["SPY"].dropna()
    else:
        out = s.iloc[:, 0].dropna()
    out.name = "SPY"
    out.to_frame().to_parquet(f)
    return out
