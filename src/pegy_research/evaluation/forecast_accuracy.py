"""Forecast vs realized EPS growth: MAE, RMSE, bias, correlation, directional hit rate."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def forecast_accuracy_table(
    df: pd.DataFrame,
    *,
    forecast_cols: Iterable[str],
    realized_col: str,
) -> pd.DataFrame:
    rows = []
    y = df[realized_col].astype(float)
    for col in forecast_cols:
        f = df[col].astype(float)
        m = f.notna() & y.notna()
        if m.sum() < 5:
            rows.append(
                {
                    "provider": col,
                    "n": int(m.sum()),
                    "mae": np.nan,
                    "rmse": np.nan,
                    "bias_mean_f_minus_y": np.nan,
                    "corr": np.nan,
                    "directional_accuracy": np.nan,
                }
            )
            continue
        e = f[m] - y[m]
        hit = np.sign(f[m]) == np.sign(y[m])
        rows.append(
            {
                "provider": col,
                "n": int(m.sum()),
                "mae": float(np.mean(np.abs(e))),
                "rmse": float(np.sqrt(np.mean(e**2))),
                "bias_mean_f_minus_y": float(np.mean(e)),
                "corr": float(f[m].corr(y[m])),
                "directional_accuracy": float(hit.mean()),
            }
        )
    return pd.DataFrame(rows)
