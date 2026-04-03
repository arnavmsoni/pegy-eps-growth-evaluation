"""Trading-signal metrics: forward excess return, precision/recall for top performers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def signal_ic_and_excess(
    pegy: pd.Series,
    fwd_excess: pd.Series,
) -> dict:
    """Cross-sectional Spearman IC and mean fwd excess for PEGY < 1 vs rest."""
    d = pd.DataFrame({"pegy": pegy.astype(float), "xs": fwd_excess.astype(float)}).dropna()
    if len(d) < 10:
        return {"n": len(d), "ic_spearman": np.nan, "mean_xs_low_pegy": np.nan, "mean_xs_high_pegy": np.nan}
    ic = d["pegy"].corr(d["xs"], method="spearman")
    low = d["pegy"] < 1.0
    return {
        "n": len(d),
        "ic_spearman": float(ic) if ic == ic else np.nan,
        "mean_xs_low_pegy": float(d.loc[low, "xs"].mean()) if low.any() else np.nan,
        "mean_xs_high_pegy": float(d.loc[~low, "xs"].mean()) if (~low).any() else np.nan,
    }


def precision_recall_top_performers(
    pegy: pd.Series,
    fwd_excess: pd.Series,
    *,
    pegy_threshold: float = 1.0,
    top_q: float = 0.30,
) -> dict:
    """
    Positive class = top `top_q` forward excess return names in the cross section.
    Predicted positive = PEGY < pegy_threshold (value screen).
    """
    d = pd.DataFrame({"pegy": pegy.astype(float), "xs": fwd_excess.astype(float)}).dropna()
    if len(d) < 20:
        return {"precision": np.nan, "recall": np.nan, "f1": np.nan, "n": len(d)}
    thr = d["xs"].quantile(1.0 - top_q)
    actual_pos = d["xs"] >= thr
    pred_pos = d["pegy"] < pegy_threshold
    tp = (pred_pos & actual_pos).sum()
    fp = (pred_pos & ~actual_pos).sum()
    fn = (~pred_pos & actual_pos).sum()
    prec = tp / (tp + fp) if (tp + fp) else np.nan
    rec = tp / (tp + fn) if (tp + fn) else np.nan
    f1 = 2 * prec * rec / (prec + rec) if prec == prec and rec == rec and (prec + rec) > 0 else np.nan
    return {
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "n": len(d),
    }
