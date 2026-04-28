"""
Time-aware train/test split + sklearn pipelines (median imputer fit on train only).

Using pipelines avoids leakage from global preprocessing on the full dataset.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

FEATURE_COLUMNS: list[str] = [
    "peg",
    "pe",
    "growth",
    "log_mktcap",
    "debt_to_equity",
    "volatility",
    # profit_margin often 100% missing on yfinance for the S&P subset; see
    # diagnostics and re-add if your build has coverage
]


def _numeric_imputer_pipeline(estimator, *, include_scale: bool = False) -> Pipeline:
    steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if include_scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", estimator))
    return Pipeline(steps)


@dataclass
class TrainSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    train_rows: pd.DataFrame
    test_rows: pd.DataFrame
    feature_names: list[str]


def time_based_split(
    df: pd.DataFrame,
    *,
    time_col: str = "quarter_end",
    label_col: str = "label",
    train_fraction: float = 0.7,
) -> TrainSplit:
    """Sort by ``time_col`` — chronological split (no shuffle)."""
    d = df.sort_values([time_col, "ticker"], kind="mergesort").reset_index(drop=True)
    n = len(d)
    n_train = int(max(1, min(n - 1, round(n * train_fraction))))
    tr = d.iloc[:n_train]
    te = d.iloc[n_train:]

    feats = [c for c in FEATURE_COLUMNS if c in d.columns]
    X_tr = tr[feats].copy()
    X_te = te[feats].copy()
    y_tr = tr[label_col].values.astype(int)
    y_te = te[label_col].values.astype(int)

    return TrainSplit(
        X_train=X_tr,
        X_test=X_te,
        y_train=y_tr,
        y_test=y_te,
        train_rows=tr,
        test_rows=te,
        feature_names=feats,
    )


@dataclass
class FittedModels:
    tree: Pipeline
    logit: Pipeline
    forest: Pipeline
    feature_names: list[str]


def _usable_features(X_train: pd.DataFrame, want: list[str]) -> list[str]:
    """
    Drop columns that are **entirely** missing in train (sklearn would drop them
    and break `feature_names` / `export_text` alignment) and skip near-constant
    dead features.
    """
    out: list[str] = []
    for c in want:
        if c not in X_train.columns:
            continue
        s = X_train[c]
        if not s.notna().any():
            continue
        v = s.dropna()
        if len(v) < 1:
            continue
        # single non-NaN: keep (imputer median = that value)
        if len(v) >= 2 and float(v.std()) < 1e-12:
            continue
        out.append(c)
    return out


def fit_all(
    split: TrainSplit,
    *,
    tree_max_depth: int = 5,
    rf_n_estimators: int = 200,
    random_state: int = 42,
) -> FittedModels:
    """
    All models wrap scikit-learn Pipelines: **SimpleImputer(median)** is fit
    only on ``X_train`` (no leakage from test medians).
    """
    base = [c for c in split.feature_names if c in split.X_train.columns]
    names = _usable_features(split.X_train, base)
    if not names:
        raise ValueError("No usable feature columns in training (all-NaN or constant).")
    Xtr = split.X_train[names]
    Xte = split.X_test[names]

    tree = _numeric_imputer_pipeline(
        DecisionTreeClassifier(
            max_depth=tree_max_depth,
            min_samples_leaf=15,
            class_weight="balanced",
            random_state=random_state,
        ),
        include_scale=False,
    )
    tree.fit(Xtr, split.y_train)

    logit = _numeric_imputer_pipeline(
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
        ),
        include_scale=True,
    )
    logit.fit(Xtr, split.y_train)

    forest = _numeric_imputer_pipeline(
        RandomForestClassifier(
            n_estimators=rf_n_estimators,
            max_depth=10,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        include_scale=False,
    )
    forest.fit(Xtr, split.y_train)

    return FittedModels(
        tree=tree, logit=logit, forest=forest, feature_names=names
    )


def predict_all(models: FittedModels, X: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "tree": models.tree.predict(X),
        "logit": models.logit.predict(X),
        "forest": models.forest.predict(X),
    }


def classifier_step(pipeline: Pipeline) -> object:
    """The final estimator inside a Pipeline (for trees, importances)."""
    if isinstance(pipeline, Pipeline):
        return pipeline.named_steps["clf"]
    return pipeline
