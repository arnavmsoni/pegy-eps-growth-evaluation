"""
Classification metrics, backtests, interpretability.

Backtests: see module docstring on what “mean return” means (overlapping 12m
windows vs quarter-aggregated portfolio series).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.tree import export_text, plot_tree

from pegy_research.train_models import FittedModels, classifier_step


def classification_report_dict(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "accuracy": float(acc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f),
    }


def print_metrics_table(name: str, metrics: dict[str, float]) -> None:
    print(f"\n--- {name} ---")
    for k in ("accuracy", "precision", "recall"):
        print(f"  {k.capitalize():12} {metrics[k]:.4f}")
    print(f"  {'F1':12} {metrics['f1']:.4f}")


def print_confusion(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print(f"\n  Confusion matrix [{name}]  (rows=true 0,1 / cols=pred 0,1):")
    print(f"    [[TN={cm[0,0]:4d}  FP={cm[0,1]:4d}]")
    print(f"     [FN={cm[1,0]:4d}  TP={cm[1,1]:4d}]]")


def sharpe_ratio(
    series: np.ndarray | pd.Series, *, periods_per_year: float
) -> Optional[float]:
    """Sample Sharpe: mean/xstd of *independent* observations; scale by periods."""
    r = np.asarray(series, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return None
    m, s = float(np.mean(r)), float(np.std(r, ddof=1))
    if s < 1e-12:
        return None
    return (m / s) * float(np.sqrt(periods_per_year))


def backtest_strategies(
    test_df: pd.DataFrame,
    pred_label1: np.ndarray,
    *,
    ret_col: str = "future_return",
    time_col: str = "quarter_end",
) -> None:
    """
    Strategy A: equal-weight **all** low-PEG rows in the test table.
    Strategy B: equal-weight rows where the model predicts class 1.

    We report **two** return summaries:

    1) **Cross-sectional** mean/median of per-row 12m simple returns. These are
       **overlapping** windows (same stock can appear in adjacent quarters) — the
       mean is **not** a compounded annual portfolio return; it is the average
       of many independent forward windows (interpret as “typical 12m trade”).

    2) **By quarter_end**: for each calendar quarter, equal-weight mean return
       across names; then mean/median **across quarters** (one value per
       quarter). This reduces double-counting of the same calendar period and is
       closer to a time-series of portfolio outcomes (still not compound).
    """
    d = test_df.copy()
    d["pred_pos"] = pred_label1.astype(int)

    base = d[ret_col].astype(float)
    sel = d.loc[d["pred_pos"] == 1, ret_col].astype(float)

    print("\n--- Return backtest (test set, realized 12m simple returns) ---")
    print(
        "  Note: each row is one (ticker, signal quarter) with its own 12m window."
    )
    print(
        f"  Strategy A  mean={base.mean():.2%}  median={base.median():.2%}  "
        f"n_positions={len(base)}"
    )
    if len(sel) > 0:
        print(
            f"  Strategy B  mean={sel.mean():.2%}  median={sel.median():.2%}  "
            f"n_positions={len(sel)}"
        )
    else:
        print("  Strategy B  (no positive predictions)")

    # Quarterly aggregation (one portfolio return proxy per signal quarter)
    if time_col in d.columns:
        q_a: list[float] = []
        q_b: list[float] = []
        for _, g in d.groupby(time_col, sort=True):
            q_a.append(float(g[ret_col].mean()))
            g1 = g.loc[g["pred_pos"] == 1]
            if len(g1) > 0:
                q_b.append(float(g1[ret_col].mean()))
        qa = np.array(q_a, dtype=float)
        print(
            f"\n  By {time_col} (equal-weight within quarter, then across time):"
        )
        print(
            f"    Strategy A  mean_of_quarterly_means={qa.mean():.2%}  "
            f"median={np.median(qa):.2%}  n_quarters={len(qa)}"
        )
        if len(q_b) > 0:
            qb = np.array(q_b, dtype=float)
            print(
                f"    Strategy B  mean_of_quarterly_means={qb.mean():.2%}  "
                f"median={np.median(qb):.2%}  n_quarters_with_longs={len(qb)}"
            )
            sh_a = sharpe_ratio(qa, periods_per_year=4.0)
            sh_b = sharpe_ratio(qb, periods_per_year=4.0)
            if sh_a is not None:
                print(
                    f"    Sharpe (on quarterly means, ~4/yr): A={sh_a:.3f}"
                )
            if sh_b is not None:
                print(
                    f"    Sharpe (on quarterly means, ~4/yr): B={sh_b:.3f}"
                )
            print(
                f"    Delta (B−A) quarterly-mean series: "
                f"{(qb.mean() - qa.mean()):.2%}  (only quarters where B invests)"
            )


def save_tree_figure(
    models: FittedModels, out_dir: Path, filename: str = "decision_tree.png"
) -> Optional[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    fig, ax = plt.subplots(figsize=(20, 10))
    tree_est = classifier_step(models.tree)
    plot_tree(
        tree_est,
        feature_names=models.feature_names,
        class_names=["lose", "beat_mkt"],
        filled=True,
        rounded=True,
        fontsize=8,
        ax=ax,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def print_random_forest_importance(models: FittedModels) -> None:
    print("\n--- Random forest feature importances ---")
    rf = classifier_step(models.forest)
    im = getattr(rf, "feature_importances_", None)
    if im is None:
        print("  (unavailable)")
        return
    for name, v in sorted(zip(models.feature_names, im), key=lambda x: -x[1]):
        print(f"  {name:18} {v:.4f}")


def print_decision_tree_rules(models: FittedModels, max_depth: int = 5) -> None:
    tree_est = classifier_step(models.tree)
    text = export_text(
        tree_est,
        feature_names=models.feature_names,
        max_depth=max_depth,
    )
    print("\n--- Decision tree rules (export_text) ---")
    print(text)


def evaluate_all(
    models: FittedModels,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    test_rows: pd.DataFrame,
    *,
    output_dir: Optional[Path] = None,
) -> None:
    from pegy_research.train_models import predict_all

    preds = predict_all(models, X_test)
    for mname, yhat in preds.items():
        m = classification_report_dict(y_test, yhat)
        print_metrics_table(mname.upper(), m)
        print_confusion(mname, y_test, yhat)

    if output_dir is not None:
        p = save_tree_figure(models, output_dir)
        if p is not None:
            print(f"\nSaved decision tree figure to {p}")

    print_random_forest_importance(models)
    print_decision_tree_rules(models)

    print("\n--- Backtests by model (test set) ---")
    for mname, yhat in preds.items():
        print(f"\n* {mname}")
        backtest_strategies(test_rows, yhat)
