"""
PEGY = P/E / (EPS_growth + Dividend_yield)

All inputs as decimals (growth 0.15 = 15%, yield 0.02 = 2%).
Economic interpretation: lower PEGY suggests cheaper valuation per unit of
expected total payout growth (earnings growth plus dividend yield).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def harmonize_yield_to_decimal(series: pd.Series) -> pd.Series:
    """If values look like percent (e.g. 3.5), convert to 0.035."""
    s = series.astype(float)
    out = s.copy()
    mask = s.abs() > 1.0
    out.loc[mask] = s.loc[mask] / 100.0
    return out


def compute_pegy(
    pe: pd.Series,
    eps_growth: pd.Series,
    dividend_yield: pd.Series,
    *,
    min_denominator: float = 1e-6,
    max_pegy: float = 1e6,
) -> pd.Series:
    """
    Vectorized PEGY. Invalid or extreme inputs -> NaN.

    min_denominator avoids divide-by-zero when growth + yield ~ 0.
    max_pegy caps numerical blow-ups for interpretability in plots.
    """
    g = eps_growth.astype(float)
    y = harmonize_yield_to_decimal(dividend_yield.astype(float))
    p = pe.astype(float)
    den = g + y
    pegy = p / den.replace(0, np.nan)
    pegy = pegy.where(den.abs() >= min_denominator)
    pegy = pegy.where((p > 0) & (p < 1e6))
    pegy = pegy.clip(upper=max_pegy)
    return pegy
