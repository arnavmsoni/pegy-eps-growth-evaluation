"""
Point-in-time fundamentals and price-derived features from yfinance data.

No future information: for an as-of date ``t``, only statements and prices
known on or before ``t`` are used.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

# Minimum annualized EPS growth (in percent) to keep for valid PEG
MIN_GROWTH_PCT = 0.5


def strip_series_timezone(s: pd.Series) -> pd.Series:
    """Make a time index tz-naive (UTC wall time) for safe date comparisons."""
    if s is None or s.empty:
        return s
    out = s.copy()
    if isinstance(out.index, pd.DatetimeIndex) and out.index.tz is not None:
        out.index = out.index.tz_convert("UTC").tz_localize(None)
    return out.sort_index()


def _to_ts(x: Any) -> pd.Timestamp:
    return pd.Timestamp(x).normalize()


def last_price_on_or_before(
    series: pd.Series, asof: pd.Timestamp
) -> tuple[Optional[float], Optional[pd.Timestamp]]:
    """
    Last **observed** price strictly on or before ``asof`` (no forward tolerance).

    Using prices after ``asof`` for a signal dated ``asof`` is lookahead. Monthly
    bars are aligned to month-end; ``asof`` is typically a calendar quarter-end.
    """
    if series is None or series.empty:
        return None, None
    s = series.dropna().sort_index()
    asof = _to_ts(asof)
    sub = s.loc[s.index <= asof]
    if sub.empty:
        return None, None
    dt = pd.Timestamp(sub.index[-1])
    return float(sub.iloc[-1]), dt


def price_asof(
    series: pd.Series, asof: pd.Timestamp, *, tolerance_days: int = 0
) -> Optional[float]:
    """
    Last price on or before ``asof``. ``tolerance_days`` is ignored (kept for
    call-site compatibility); do not add forward-looking slack.
    """
    p, _ = last_price_on_or_before(series, asof)
    return p


def forward_price_on_or_before(
    series: pd.Series, start: pd.Timestamp, months_forward: int = 12
) -> tuple[Optional[float], Optional[pd.Timestamp]]:
    """
    Price on the last month-end **on or before** (start + months_forward).
    """
    if series is None or series.empty:
        return None, None
    target = _to_ts(start) + pd.DateOffset(months=months_forward)
    return last_price_on_or_before(series, target)


def forward_price_asof(
    series: pd.Series, start: pd.Timestamp, months_forward: int = 12
) -> Optional[float]:
    """Backward-looking exit price at the 12m horizon (no post-horizon bias)."""
    p, _ = forward_price_on_or_before(series, start, months_forward=months_forward)
    return p


def monthly_return_volatility(
    series: pd.Series, asof: pd.Timestamp, lookback_months: int = 12
) -> Optional[float]:
    """
    Standard deviation of *monthly* simple returns over lookback_months
    ending at asof (no look-ahead).
    """
    s = series.dropna().sort_index()
    asof = _to_ts(asof)
    start = asof - pd.DateOffset(months=lookback_months + 1)
    window = s[(s.index > start) & (s.index <= asof)]
    if len(window) < 4:
        return None
    rets = window.pct_change().dropna()
    if len(rets) < 3:
        return None
    return float(rets.std(ddof=1))


def sorted_statement_columns(stmt: pd.DataFrame) -> list[pd.Timestamp]:
    if stmt is None or stmt.empty or len(stmt.columns) < 1:
        return []
    cols = [pd.Timestamp(c) for c in stmt.columns]
    cols.sort()
    return cols


def _row_value(stmt: pd.DataFrame, names: tuple[str, ...]) -> Optional[pd.Series]:
    for n in names:
        if n in stmt.index:
            return stmt.loc[n].astype(float)
    return None


def ttm_eps_at(
    income: pd.DataFrame, asof: pd.Timestamp
) -> Optional[float]:
    """
    Trailing four-quarter diluted EPS sum using only periods with
    fiscal quarter-end on or before ``asof`` (point-in-time).
    """
    if income is None or income.empty:
        return None
    row = _row_value(income, ("Diluted EPS", "Basic EPS"))
    if row is None:
        return None
    asof = _to_ts(asof)
    cols = sorted_statement_columns(income)
    eligible = [c for c in cols if c <= asof]
    if len(eligible) < 4:
        return None
    last4 = eligible[-4:]
    total = 0.0
    for c in last4:
        v = row.get(c)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        total += float(v)
    return total


def ttm_revenue_and_ni_at(
    income: pd.DataFrame, asof: pd.Timestamp
) -> tuple[Optional[float], Optional[float]]:
    """Trailing twelve months total revenue and net income (sum of last 4 Q)."""
    if income is None or income.empty:
        return None, None
    asof = _to_ts(asof)
    cols = sorted_statement_columns(income)
    eligible = [c for c in cols if c <= asof]
    if len(eligible) < 4:
        return None, None
    last4 = eligible[-4:]

    rev_row = _row_value(income, ("Total Revenue", "Operating Revenue"))
    ni_row = _row_value(income, ("Net Income", "Net Income Common Stockholders"))
    tr: Optional[float] = None
    tni: Optional[float] = None
    if rev_row is not None:
        tr = float(sum(_safe_float(rev_row.get(c)) or 0.0 for c in last4))
        if tr == 0.0:
            tr = None
    if ni_row is not None:
        tni = float(sum(_safe_float(ni_row.get(c)) or 0.0 for c in last4))
        if tni == 0.0 and all(
            _safe_float(ni_row.get(c)) in (None, 0.0) for c in last4
        ):
            tni = None
    return tr, tni


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
        if v != v or np.isinf(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


def eps_growth_pct_yoy_at(income: pd.DataFrame, asof: pd.Timestamp) -> Optional[float]:
    """
    EPS growth % YoY. Prefers TTM YoY when >=8 quarters are available on yfinance;
    otherwise uses single-fiscal-quarter YoY (latest vs five quarters back) when
    >=5 columns exist — yfinance often returns only ~5 recent quarters.
    """
    if income is None or income.empty:
        return None
    row = _row_value(income, ("Diluted EPS", "Basic EPS"))
    if row is None:
        return None
    asof = _to_ts(asof)
    cols = sorted_statement_columns(income)
    eligible = [c for c in cols if c <= asof]
    if len(eligible) < 5:
        return None

    def _sum_eps(qcols: list) -> Optional[float]:
        s = 0.0
        for c in qcols:
            v = _safe_float(row[c])  # type: ignore
            if v is None:
                return None
            s += v
        return s

    if len(eligible) >= 8:
        last4 = eligible[-4:]
        prev4 = eligible[-8:-4]
        ttm_now = _sum_eps(last4)
        ttm_prev = _sum_eps(prev4)
        if ttm_now is not None and ttm_prev is not None and abs(float(ttm_prev)) >= 1e-8:
            return (float(ttm_now) - float(ttm_prev)) / abs(float(ttm_prev)) * 100.0

    # Single-quarter YoY: compare latest reported quarter to ~one year prior
    q0, q4 = eligible[-1], eligible[-5]
    e0 = _safe_float(row[q0])  # type: ignore
    e1 = _safe_float(row[q4])  # type: ignore
    if e0 is None or e1 is None or abs(e1) < 1e-8:
        return None
    return (float(e0) - float(e1)) / abs(float(e1)) * 100.0


def revenue_growth_pct_yoy_at(
    income: pd.DataFrame, asof: pd.Timestamp
) -> Optional[float]:
    """Revenue growth % YoY; TTM if possible else single-quarter YoY."""
    if income is None or income.empty:
        return None
    row = _row_value(income, ("Total Revenue", "Operating Revenue"))
    if row is None:
        return None
    asof = _to_ts(asof)
    cols = sorted_statement_columns(income)
    eligible = [c for c in cols if c <= asof]
    if len(eligible) < 5:
        return None

    def _sum_r(qcols: list) -> Optional[float]:
        s = 0.0
        for c in qcols:
            v = _safe_float(row[c])  # type: ignore
            if v is None:
                return None
            s += v
        return s

    if len(eligible) >= 8:
        last4 = eligible[-4:]
        prev4 = eligible[-8:-4]
        ttm_now = _sum_r(last4)
        ttm_prev = _sum_r(prev4)
        if ttm_now is not None and ttm_prev is not None and abs(float(ttm_prev)) >= 1e-8:
            return (float(ttm_now) - float(ttm_prev)) / abs(float(ttm_prev)) * 100.0

    q0, q4 = eligible[-1], eligible[-5]
    r0 = _safe_float(row[q0])  # type: ignore
    r1 = _safe_float(row[q4])  # type: ignore
    if r0 is None or r1 is None or abs(r1) < 1e-8:
        return None
    return (float(r0) - float(r1)) / abs(float(r1)) * 100.0


def ttm_eps_from_earnings_table(
    earnings: pd.DataFrame, asof: pd.Timestamp
) -> Optional[float]:
    """
    Sum of the four most recent fiscal-quarter ``Reported EPS`` rows whose
    earnings announcement date is on or before ``asof`` (point-in-time).
    """
    if earnings is None or earnings.empty:
        return None
    if "Reported EPS" not in earnings.columns:
        return None
    asof = _to_ts(asof)
    d = earnings.dropna(subset=["Reported EPS"]).sort_index().copy()
    if getattr(d.index, "tz", None) is not None:
        d.index = d.index.tz_convert("UTC").tz_localize(None)
    # Strict: only earnings **announced** on or before asof (no +7d slack).
    d = d[d.index <= asof]
    if len(d) < 4:
        return None
    last4 = d["Reported EPS"].astype(float).iloc[-4:]
    return float(last4.sum())


def ttm_yoy_growth_pct_from_earnings(
    earnings: pd.DataFrame, asof: pd.Timestamp
) -> Optional[float]:
    """YoY % change in trailing-twelve-month EPS built from earnings announcements."""
    t0 = ttm_eps_from_earnings_table(earnings, asof)
    asof_prev = asof - pd.DateOffset(years=1)
    t1 = ttm_eps_from_earnings_table(earnings, asof_prev)
    if t0 is None or t1 is None or abs(float(t1)) < 1e-9:
        return None
    return (float(t0) - float(t1)) / abs(float(t1)) * 100.0


def profit_margin_ttm_at(income: pd.DataFrame, asof: pd.Timestamp) -> Optional[float]:
    """Net income / revenue on TTM basis (fraction, not percent)."""
    rev, ni = ttm_revenue_and_ni_at(income, asof)
    if rev is None or ni is None or abs(rev) < 1e-8:
        return None
    return float(ni) / float(rev)


def shares_at(income: pd.DataFrame, balance: pd.DataFrame, asof: pd.Timestamp) -> Optional[float]:
    """Best-effort diluted share count for quarter ending at or before asof."""
    asof = _to_ts(asof)
    for stmt in (income, balance):
        if stmt is None or stmt.empty:
            continue
        for name in (
            "Diluted Average Shares",
            "Diluted Average Sharess",  # yfinance typo seen in the wild
            "Basic Average Shares",
            "Share Issued",
        ):
            if name in stmt.index:
                row = stmt.loc[name]
                cols = sorted_statement_columns(stmt)
                el = [c for c in cols if c <= asof]
                if not el:
                    continue
                v = _safe_float(row[el[-1]])
                if v is not None and v > 0:
                    return v
    return None


def debt_to_equity_at(
    balance: pd.DataFrame, asof: pd.Timestamp
) -> Optional[float]:
    """
    Total debt / total stockholder equity for the most recent quarter end <= asof.
    """
    if balance is None or balance.empty:
        return None
    asof = _to_ts(asof)
    cols = sorted_statement_columns(balance)
    eligible = [c for c in cols if c <= asof]
    if not eligible:
        return None
    c = eligible[-1]
    debt_row = _row_value(
        balance,
        (
            "Total Debt",
            "Long Term Debt",
        ),
    )
    eq_row = _row_value(
        balance,
        (
            "Total Stockholder Equity",
            "Common Stock Equity",
            "Stockholders Equity",
        ),
    )
    if debt_row is None or eq_row is None:
        return None
    d = _safe_float(debt_row.get(c)) or 0.0
    e = _safe_float(eq_row.get(c))
    if e is None or abs(e) < 1e-8:
        return None
    return float(d) / float(e)


def pe_ratio(price: float, eps_ttm: float) -> Optional[float]:
    if eps_ttm is None or abs(eps_ttm) < 1e-9:
        return None
    return float(price) / float(eps_ttm)


def peg_ratio(pe: float, growth_pct: float) -> Optional[float]:
    """
    PEG = (P/E) / (expected/achieved growth as a *percentage* number, e.g. 12 for 12%).
    Requires positive growth (spec: drop negative / near-zero for PEG).
    """
    if growth_pct is None or growth_pct <= MIN_GROWTH_PCT:
        return None
    if pe is None or pe <= 0:
        return None
    return float(pe) / float(growth_pct)


def choose_growth(
    income: pd.DataFrame,
    asof: pd.Timestamp,
    *,
    earnings: Optional[pd.DataFrame] = None,
) -> tuple[Optional[float], str]:
    """
    EPS YoY % from earnings-announcement TTM when ``earnings`` is provided;
    else statement-based EPS/revenue growth. Only **positive** growth for PEG.
    """
    if earnings is not None and not earnings.empty:
        g = ttm_yoy_growth_pct_from_earnings(earnings, asof)
        if g is not None and g > MIN_GROWTH_PCT:
            return g, "eps_ttm_yoy"
    g = eps_growth_pct_yoy_at(income, asof)
    if g is not None and g > MIN_GROWTH_PCT:
        return g, "eps_stmt"
    g2 = revenue_growth_pct_yoy_at(income, asof)
    if g2 is not None and g2 > MIN_GROWTH_PCT:
        return g2, "revenue"
    return None, "none"


def choose_eps_ttm_for_pe(
    earnings: Optional[pd.DataFrame],
    income: pd.DataFrame,
    asof: pd.Timestamp,
) -> Optional[float]:
    """Prefer earnings-derived TTM EPS for P/E and PEG; fallback to income stmt."""
    if earnings is not None and not earnings.empty:
        e = ttm_eps_from_earnings_table(earnings, asof)
        if e is not None:
            return e
    return ttm_eps_at(income, asof)
