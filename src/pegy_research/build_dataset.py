"""
Assemble (stock, quarter-end) rows with fundamentals, returns, and labels.

Rows with PEG < peg_max. **No** test-set leakage: no global imputation or
outlier z-scores here — that happens inside training (fit on train only).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from pegy_research.data_loader import (
    DEFAULT_CACHE_DIR,
    fetch_earnings_history,
    fetch_quarterly_statements,
)
from pegy_research.feature_engineering import (
    MIN_GROWTH_PCT,
    choose_eps_ttm_for_pe,
    choose_growth,
    debt_to_equity_at,
    forward_price_on_or_before,
    last_price_on_or_before,
    monthly_return_volatility,
    peg_ratio,
    pe_ratio,
    profit_margin_ttm_at,
    shares_at,
    strip_series_timezone,
)


def quarterly_calendar_ends(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    """Calendar quarter-end dates in [start, end] (inclusive)."""
    s, e = start.normalize(), end.normalize()
    try:
        dr = pd.date_range(start=s, end=e, freq="QE-DEC")
    except (ValueError, TypeError):
        dr = pd.date_range(start=s, end=e, freq="Q-DEC")
    return [pd.Timestamp(ts).normalize() for ts in dr]


def _coerce_timestamp(x: object) -> Optional[pd.Timestamp]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return pd.Timestamp(x).normalize()


def build_dataset(
    tickers: list[str],
    price_panel: pd.DataFrame,
    spy: pd.Series,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    peg_max: float = 1.5,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    For each (ticker, calendar quarter-end) in range, build one row when:

    * 12m forward exit price is observable (monthly bars on/before horizon);
    * PEG < peg_max and valid P/E, growth;
    * holding length is plausibly ~1y (filters bad month misalignment).

    Label: 1 if stock 12m simple return > SPY 12m simple return over the same
    **calendar** entry/exit month-ends (strict backward prices, no lookahead).
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    spy = strip_series_timezone(spy)
    q_ends = quarterly_calendar_ends(start, end)
    rows: list[dict] = []

    for sym in tickers:
        if sym not in price_panel.columns:
            continue
        px = strip_series_timezone(price_panel[sym].dropna())
        if px.empty:
            continue
        try:
            earn = fetch_earnings_history(sym, cache_dir=cache_dir)
        except Exception:
            earn = pd.DataFrame()
        try:
            stm = fetch_quarterly_statements(sym, cache_dir=cache_dir)
        except Exception:
            continue
        income = stm.get("income", pd.DataFrame())
        balance = stm.get("balance", pd.DataFrame())

        for t in q_ends:
            horizon_end = t + pd.DateOffset(months=12)
            if px.index.max() < horizon_end - pd.Timedelta(
                days=20
            ) or spy.index.max() < horizon_end - pd.Timedelta(days=20):
                continue

            # Entry / exit: last month-end on or before t and on or before t+12m.
            p0, d0 = last_price_on_or_before(px, t)
            p1, d1 = forward_price_on_or_before(px, t, months_forward=12)
            sp0, _ = last_price_on_or_before(spy, t)
            sp1, _ = forward_price_on_or_before(spy, t, months_forward=12)

            if (
                p0 is None
                or p1 is None
                or sp0 is None
                or sp1 is None
                or p0 <= 0
                or sp0 <= 0
            ):
                continue

            fut = (p1 - p0) / p0
            mret = (sp1 - sp0) / sp0
            d0n = _coerce_timestamp(d0)
            d1n = _coerce_timestamp(d1)
            days_held: Optional[int] = None
            if d0n is not None and d1n is not None:
                days_held = int((d1n - d0n).days)
                # Drop rows where monthly alignment is far from 12 calendar months
                if days_held < 280 or days_held > 450:
                    continue

            eps_ttm = choose_eps_ttm_for_pe(earn, income, t)
            growth, gsrc = choose_growth(income, t, earnings=earn)
            if growth is None or growth <= MIN_GROWTH_PCT:
                continue
            pe = pe_ratio(p0, eps_ttm) if eps_ttm is not None else None
            if pe is None or pe <= 0:
                continue
            peg = peg_ratio(pe, growth)
            if peg is None or peg > peg_max:
                continue

            mcap = None
            log_mkt: Optional[float] = None
            sh = shares_at(income, balance, t)
            if sh is not None and p0 is not None:
                mcap = float(p0) * float(sh)
                if mcap > 0:
                    log_mkt = float(np.log1p(mcap))

            vol = monthly_return_volatility(px, t, lookback_months=12)
            pm = profit_margin_ttm_at(income, t)
            de = debt_to_equity_at(balance, t)

            label = 1 if fut > mret else 0

            rows.append(
                {
                    "ticker": sym,
                    "quarter_end": t,
                    "peg": peg,
                    "pe": pe,
                    "growth": growth,
                    "growth_source": gsrc,
                    "market_cap": mcap,
                    "log_mktcap": log_mkt,
                    "profit_margin": pm,
                    "debt_to_equity": de,
                    "volatility": vol,
                    "future_return": fut,
                    "market_return": mret,
                    "excess_return": fut - mret,
                    "price_entry": p0,
                    "price_exit": p1,
                    "spy_entry": sp0,
                    "spy_exit": sp1,
                    "entry_month_end": d0n,
                    "exit_month_end": d1n,
                    "days_held": days_held,
                    "label": label,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    required = [
        "peg",
        "pe",
        "growth",
        "future_return",
        "market_return",
        "label",
    ]
    df = df.dropna(subset=required)
    return df.reset_index(drop=True)
