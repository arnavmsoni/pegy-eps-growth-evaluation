"""
Monthly long-only backtest: equal-weight stocks with PEGY < threshold.

Separates signal date (fundamentals known as of prior quarter) from return
realization using month-end prices. EPS growth from analyst APIs is treated as
a slow-moving latent factor (constant in-window) — documented limitation vs
true point-in-time estimate databases.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def monthly_long_pegy_backtest(
    *,
    month_ends: pd.DatetimeIndex,
    pegy_wide: pd.DataFrame,
    monthly_returns: pd.DataFrame,
    pegy_threshold: float = 1.0,
    benchmark_returns: pd.Series | None = None,
) -> pd.DataFrame:
    """
    pegy_wide: index = month_ends, columns = tickers (signal known at rebalance).
    monthly_returns: index = month_ends, columns = tickers (simple period returns).

    At each month t, hold equal-weight long all names with pegy.loc[t] < threshold
    (and finite). One-month forward return is computed from t to t+1.
    """
    rows: List[dict] = []
    me = month_ends.sort_values()
    for i in range(len(me) - 1):
        t0, t1 = me[i], me[i + 1]
        if t0 not in pegy_wide.index or t0 not in monthly_returns.index:
            continue
        if t1 not in monthly_returns.index:
            continue
        sig = pegy_wide.loc[t0]
        eligible = sig[sig < pegy_threshold].dropna()
        names = list(eligible.index)
        if not names:
            r_p = np.nan
        else:
            r_vec = monthly_returns.loc[t1, names].astype(float)
            r_p = float(r_vec.mean(skipna=True))
        row = {"date": t0, "portfolio_1m": r_p, "n_names": len(names)}
        if benchmark_returns is not None and t1 in benchmark_returns.index:
            row["benchmark_1m"] = float(benchmark_returns.loc[t1])
            row["excess_1m"] = r_p - row["benchmark_1m"] if not np.isnan(r_p) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def forward_return_12m(prices: pd.DataFrame) -> pd.DataFrame:
    """
    prices: index = dates, columns = tickers (adjusted close).
    For each date t, return[t] = P[t'] / P[t] - 1 where t' is the first index
    >= t + 12 calendar months (avoids using same-bar price as future).
    """
    px = prices.sort_index()
    out = pd.DataFrame(index=px.index, columns=px.columns, dtype=float)
    idx = px.index
    for i, dt in enumerate(idx):
        target = dt + pd.DateOffset(months=12)
        j = idx.searchsorted(target, side="left")
        if j >= len(idx):
            continue
        fut = idx[j]
        out.iloc[i] = (px.loc[fut] / px.loc[dt]) - 1.0
    return out
