"""
Dataset sanity checks: missingness, class balance, return ranges (printed to stdout).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def print_dataset_diagnostics(df: pd.DataFrame, *, label_col: str = "label") -> None:
    """Call on the raw panel **before** train-only imputation."""
    if df.empty:
        print("\n[diagnostics] empty dataframe")
        return

    print("\n" + "=" * 60)
    print("DATASET DIAGNOSTICS (pre-train imputation)")
    print("=" * 60)
    print(f"  rows: {len(df)}")
    if "quarter_end" in df.columns:
        print(
            f"  quarter_end range: {df['quarter_end'].min()} → {df['quarter_end'].max()}"
        )
        nq = df["quarter_end"].nunique()
        print(f"  distinct quarter_end values: {nq}")

    if label_col in df.columns:
        vc = df[label_col].value_counts().sort_index()
        print(f"  label distribution: {dict(vc)}")
        if len(vc) >= 2:
            p = vc.get(1, 0) / len(df)
            print(
                f"  positive rate (label=1): {p:.1%}  (imbalanced if <10% or >90%)"
            )

    num_cols = [
        "peg",
        "pe",
        "growth",
        "market_cap",
        "log_mktcap",
        "profit_margin",
        "debt_to_equity",
        "volatility",
        "future_return",
        "market_return",
    ]
    print("\n  Missing-value fraction (raw):")
    for c in num_cols:
        if c not in df.columns:
            continue
        m = float(df[c].isna().mean())
        print(f"    {c:18} {m:6.1%}")

    for c in ("market_cap", "profit_margin", "debt_to_equity"):
        if c in df.columns and df[c].notna().any():
            v = df[c].dropna()
            print(
                f"  {c}: std={v.std():.6g}  min={v.min():.6g}  max={v.max():.6g}"
            )

    if "future_return" in df.columns and "market_return" in df.columns:
        fr = df["future_return"].astype(float)
        mr = df["market_return"].astype(float)
        ex = fr - mr
        print("\n  12m forward return summary (simple per row):")
        print(
            f"    stock return  mean={fr.mean():.2%}  median={fr.median():.2%}  "
            f"min={fr.min():.2%}  max={fr.max():.2%}"
        )
        print(
            f"    SPY return    mean={mr.mean():.2%}  median={mr.median():.2%}"
        )
        print(
            f"    excess (r-m)  mean={ex.mean():.2%}  median={ex.median():.2%}"
        )

    if "days_held" in df.columns:
        dh = df["days_held"].dropna()
        if len(dh):
            print(
                f"\n  holding calendar days (entry→exit): "
                f"mean={dh.mean():.0f}  median={dh.median():.0f}  "
                f"min={dh.min():.0f}  max={dh.max():.0f}"
            )

    print("=" * 60 + "\n")
