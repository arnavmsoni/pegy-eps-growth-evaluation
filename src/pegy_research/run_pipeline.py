"""
End-to-end research runner: ingest → PEGY panel → backtest → evaluation → outputs.

Design notes (bias / validity):
- Universe: FMP S&P 500 list (cached) or PEGY_TICKERS override; not point-in-time
  constituents (survivorship caveat documented).
- Realized EPS growth: YoY change in reported annual EPS from FMP income
  statements (lagging fiscal reporting, not synchronized to fiscal calendar across
  names).
- Analyst / vendor EPS growth: contemporary API snapshot — not a true archived
  forecast path; interpret forecast metrics as contemporaneous alignment with
  fundamentals and relative ranking stability, not literal ex-ante forecast error.
- Monthly PEGY: time-varying P/E and dividend yield from quarterly ratios with
  backward as-of merge; EPS growth held fixed from snapshot (constant within run).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from pegy_research.backtest.engine import forward_return_12m, monthly_long_pegy_backtest
from pegy_research.config import ResearchConfig, apply_seed
from pegy_research.data.alpha_vantage import AlphaVantageClient
from pegy_research.data.cache import DiskCache
from pegy_research.data.fmp import FMPClient, _safe_float
from pegy_research.data.sec_edgar import SECEdgarClient
from pegy_research.data.twelve_data import TwelveDataClient
from pegy_research.data.universe import load_universe_tickers
from pegy_research.data.yahoo_finance import YahooFinanceClient
from pegy_research.evaluation.forecast_accuracy import forecast_accuracy_table
from pegy_research.evaluation.signal_quality import precision_recall_top_performers, signal_ic_and_excess
from pegy_research.features.pegy import compute_pegy


def _http_status(e: requests.HTTPError) -> int | str:
    return e.response.status_code if e.response is not None else "HTTP error"


def _quarterly_ratios_df(fmp: FMPClient, ticker: str) -> pd.DataFrame:
    rows = fmp.fetch_ratios_quarterly(ticker)
    out = []
    for r in rows:
        d = r.get("date")
        if not d:
            continue
        pe = _safe_float(
            r.get("priceEarningsRatio")
            or r.get("priceToEarningsRatio")
            or r.get("peRatio")
            or r.get("peRatioTTM")
        )
        dy = _safe_float(r.get("dividendYield") or r.get("dividendYieldTTM"))
        if dy is not None and dy > 1.0:
            dy = dy / 100.0
        out.append({"date": pd.Timestamp(d), "pe_ttm": pe, "dividend_yield_ttm": dy})
    df = pd.DataFrame(out)
    if df.empty:
        return df
    return df.drop_duplicates("date").sort_values("date").set_index("date")


def _daily_prices_to_month_end(fmp: FMPClient, ticker: str, start: str, end: str) -> pd.Series:
    hist = fmp.fetch_historical_prices(ticker, start, end)
    if not hist:
        return pd.Series(dtype=float)
    s = pd.Series(
        {pd.Timestamp(h["date"]): float(h.get("adjClose") or h.get("close") or np.nan) for h in hist}
    )
    s = s.sort_index().astype(float)
    return s.resample("ME").last().dropna()


def _build_monthly_pegy_for_provider(
    month_index: pd.DatetimeIndex,
    tickers: list[str],
    growth_by_ticker: dict[str, float | None],
    quarterly_pe_dy: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Rows = month-end dates, columns = tickers."""
    wide = pd.DataFrame(index=month_index, columns=tickers, dtype=float)
    month_df = pd.DataFrame({"m": np.sort(pd.DatetimeIndex(month_index))})
    for t in tickers:
        g = growth_by_ticker.get(t)
        if g is not None and isinstance(g, float) and np.isnan(g):
            g = None
        q = quarterly_pe_dy.get(t)
        if q is None or q.empty:
            continue
        if g is None:
            continue
        q_reset = q.reset_index().sort_values("date")
        m_sorted = month_df.sort_values("m")
        merged = pd.merge_asof(
            m_sorted,
            q_reset,
            left_on="m",
            right_on="date",
            direction="backward",
        )
        merged = merged.set_index("m").reindex(month_index)
        pe = merged["pe_ttm"].astype(float)
        dy = merged["dividend_yield_ttm"].astype(float)
        gs = pd.Series(float(g), index=month_index, dtype=float)
        wide[t] = compute_pegy(pe, gs, dy).values
    return wide


def _build_monthly_fundamental_panels(
    month_index: pd.DatetimeIndex,
    tickers: list[str],
    quarterly_pe_dy: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pe_wide = pd.DataFrame(index=month_index, columns=tickers, dtype=float)
    dy_wide = pd.DataFrame(index=month_index, columns=tickers, dtype=float)
    month_df = pd.DataFrame({"m": np.sort(pd.DatetimeIndex(month_index))})
    for t in tickers:
        q = quarterly_pe_dy.get(t)
        if q is None or q.empty:
            continue
        merged = pd.merge_asof(
            month_df.sort_values("m"),
            q.reset_index().sort_values("date"),
            left_on="m",
            right_on="date",
            direction="backward",
        ).set_index("m").reindex(month_index)
        pe_wide[t] = merged["pe_ttm"].astype(float).values
        dy_wide[t] = merged["dividend_yield_ttm"].astype(float).values
    return pe_wide, dy_wide


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_pred).astype(int)
    if len(y) == 0:
        return {"accuracy": np.nan, "precision": np.nan, "recall": np.nan, "f1": np.nan}
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    accuracy = float((p == y).mean())
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    f1 = 2 * precision * recall / (precision + recall) if (
        precision == precision and recall == recall and (precision + recall) > 0
    ) else np.nan
    return {"accuracy": accuracy, "precision": float(precision), "recall": float(recall), "f1": float(f1)}


def _rank_top_by_month(df: pd.DataFrame, score_col: str, top_q: float) -> pd.Series:
    preds = pd.Series(0, index=df.index, dtype=int)
    for _, idx in df.groupby("date").groups.items():
        scores = df.loc[idx, score_col].astype(float)
        if scores.notna().sum() == 0:
            continue
        cutoff = scores.quantile(1.0 - top_q)
        preds.loc[idx] = (scores >= cutoff).astype(int)
    return preds


def _fit_logistic_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    *,
    seed: int,
) -> np.ndarray:
    x_train = train[feature_cols].to_numpy(dtype=float)
    y_train = train["target_top30"].to_numpy(dtype=float)
    x_test = test[feature_cols].to_numpy(dtype=float)

    mu = x_train.mean(axis=0)
    sigma = x_train.std(axis=0)
    sigma[sigma < 1e-9] = 1.0
    x_train = (x_train - mu) / sigma
    x_test = (x_test - mu) / sigma

    x_train = np.column_stack([np.ones(len(x_train)), x_train])
    x_test = np.column_stack([np.ones(len(x_test)), x_test])
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 0.01, size=x_train.shape[1])
    lr = 0.08
    l2 = 0.01
    for _ in range(1200):
        z = np.clip(x_train @ w, -40, 40)
        pred = 1.0 / (1.0 + np.exp(-z))
        grad = (x_train.T @ (pred - y_train)) / len(y_train)
        grad[1:] += l2 * w[1:]
        w -= lr * grad

    z_test = np.clip(x_test @ w, -40, 40)
    return 1.0 / (1.0 + np.exp(-z_test))


def _build_ml_dataset(
    *,
    providers: dict[str, dict[str, float | None]],
    month_index: pd.DatetimeIndex,
    tickers: list[str],
    pe_wide: pd.DataFrame,
    dy_wide: pd.DataFrame,
    pegy_wides: dict[str, pd.DataFrame],
    fwd: pd.DataFrame,
    bench_fwd: pd.Series,
    top_q: float,
    benchmark_available: bool,
) -> pd.DataFrame:
    rows: list[dict] = []
    for provider, growth_map in providers.items():
        pegy = pegy_wides[provider]
        for dt in month_index:
            if dt not in fwd.index:
                continue
            bench = float(bench_fwd.loc[dt]) if dt in bench_fwd.index and pd.notna(bench_fwd.loc[dt]) else 0.0
            for sym in tickers:
                fwd_ret = fwd.loc[dt, sym] if sym in fwd.columns else np.nan
                if pd.isna(fwd_ret):
                    continue
                rows.append(
                    {
                        "date": dt,
                        "ticker": sym,
                        "provider": provider,
                        "eps_growth": growth_map.get(sym),
                        "pe_ttm": pe_wide.loc[dt, sym] if sym in pe_wide.columns else np.nan,
                        "dividend_yield_ttm": dy_wide.loc[dt, sym] if sym in dy_wide.columns else np.nan,
                        "pegy": pegy.loc[dt, sym] if sym in pegy.columns else np.nan,
                        "forward_12m_return": float(fwd_ret),
                        "benchmark_12m_return": bench if benchmark_available else np.nan,
                        "model_return_target": float(fwd_ret - bench) if benchmark_available else float(fwd_ret),
                    }
                )
    ml = pd.DataFrame(rows)
    if ml.empty:
        return ml
    ml = ml.replace([np.inf, -np.inf], np.nan)
    ml["pegy_lt_1"] = (ml["pegy"] < 1.0).astype(int)
    ml["target_top30"] = 0
    for _, idx in ml.groupby(["provider", "date"]).groups.items():
        returns = ml.loc[idx, "model_return_target"].astype(float)
        if returns.notna().sum() < 3:
            continue
        cutoff = returns.quantile(1.0 - top_q)
        ml.loc[idx, "target_top30"] = (returns >= cutoff).astype(int)
    return ml


def _run_ml_models(
    ml: pd.DataFrame,
    *,
    top_q: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = ["eps_growth", "pe_ttm", "dividend_yield_ttm", "pegy", "pegy_lt_1"]
    results: list[dict] = []
    predictions: list[pd.DataFrame] = []
    if ml.empty:
        return pd.DataFrame(), pd.DataFrame()

    for provider, sub in ml.groupby("provider"):
        model_df = sub.dropna(subset=feature_cols + ["target_top30"]).copy()
        model_df = model_df.sort_values(["date", "ticker"])
        dates = sorted(model_df["date"].dropna().unique())
        if len(dates) < 4 or len(model_df) < 30:
            results.append({"provider": provider, "n_train": 0, "n_test": 0, "status": "insufficient_data"})
            continue
        split_i = max(1, min(len(dates) - 1, int(len(dates) * 0.70)))
        split_date = pd.Timestamp(dates[split_i])
        train = model_df[model_df["date"] < split_date].copy()
        test = model_df[model_df["date"] >= split_date].copy()
        if train["target_top30"].nunique() < 2 or test["target_top30"].nunique() < 2:
            results.append(
                {
                    "provider": provider,
                    "n_train": len(train),
                    "n_test": len(test),
                    "split_date": str(split_date.date()),
                    "status": "single_class_split",
                }
            )
            continue

        prob = _fit_logistic_predict(train, test, feature_cols, seed=seed)
        test = test.copy()
        test["ml_probability"] = prob
        test["ml_pred_top30"] = _rank_top_by_month(test, "ml_probability", top_q)
        test["pegy_screen_pred"] = (test["pegy"] < 1.0).astype(int)
        ml_metrics = _classification_metrics(test["target_top30"].to_numpy(), test["ml_pred_top30"].to_numpy())
        pegy_metrics = _classification_metrics(test["target_top30"].to_numpy(), test["pegy_screen_pred"].to_numpy())
        results.append(
            {
                "provider": provider,
                "n_train": len(train),
                "n_test": len(test),
                "split_date": str(split_date.date()),
                "positive_rate_test": float(test["target_top30"].mean()),
                "status": "ok",
                **{f"ml_{k}": v for k, v in ml_metrics.items()},
                **{f"pegy_rule_{k}": v for k, v in pegy_metrics.items()},
            }
        )
        predictions.append(
            test[
                [
                    "date",
                    "ticker",
                    "provider",
                    "target_top30",
                    "model_return_target",
                    "ml_probability",
                    "ml_pred_top30",
                    "pegy_screen_pred",
                    *feature_cols,
                ]
            ]
        )
    pred_df = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    return pd.DataFrame(results), pred_df


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    p = argparse.ArgumentParser(description="PEGY EPS growth provider evaluation")
    p.add_argument("--from-date", default="2018-01-01")
    p.add_argument("--to-date", default="2024-12-31")
    p.add_argument("--max-tickers", type=int, default=35, help="Subset for API cost control")
    p.add_argument("--skip-av", action="store_true")
    p.add_argument("--skip-twelve", action="store_true")
    p.add_argument("--skip-yfinance", action="store_true")
    p.add_argument("--skip-sec", action="store_true")
    args = p.parse_args(argv)

    cfg = ResearchConfig.from_env()
    apply_seed(cfg)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cache = DiskCache(cfg.cache_dir)
    session = requests.Session()

    fmp_key = os.environ.get("FMP_API_KEY", "").strip()
    av_key = os.environ.get("ALPHA_VANTAGE_API_KEY", "").strip()
    td_key = os.environ.get("TWELVE_DATA_API_KEY", "").strip()

    if not fmp_key:
        print("FMP_API_KEY required (universe + prices + financials).", file=sys.stderr)
        sys.exit(1)

    fmp = FMPClient(fmp_key, cfg, cache, session)
    tickers = load_universe_tickers(cfg, cache, session, fmp_key)
    rng = np.random.default_rng(cfg.random_seed)
    if len(tickers) > args.max_tickers:
        tickers = sorted(rng.choice(tickers, size=args.max_tickers, replace=False).tolist())

    av_client = AlphaVantageClient(av_key, cfg, cache, session) if av_key and not args.skip_av else None
    td_client = TwelveDataClient(td_key, cfg, cache, session) if td_key and not args.skip_twelve else None
    yf_client = YahooFinanceClient() if not args.skip_yfinance else None
    sec_client = SECEdgarClient(
        cache,
        session,
        user_agent=os.environ.get("SEC_USER_AGENT", "pegy-research/0.1 contact@example.com"),
    ) if not args.skip_sec else None

    rows = []
    quarterly_pe_dy: dict[str, pd.DataFrame] = {}
    monthly_close: dict[str, pd.Series] = {}

    for sym in tickers:
        try:
            qdf = _quarterly_ratios_df(fmp, sym)
            quarterly_pe_dy[sym] = qdf
            mc = _daily_prices_to_month_end(fmp, sym, args.from_date, args.to_date)
            monthly_close[sym] = mc

            snap = fmp.fetch_snapshot(sym)
            f_fmp = snap["fields"].get("eps_growth_forecast")
            realized = fmp.trailing_eps_yoy_growth(sym)
            pe0 = snap["fields"].get("pe_ttm")
            dy0 = snap["fields"].get("dividend_yield_ttm")
        except requests.HTTPError as e:
            print(f"Skipping {sym}: FMP access failed ({_http_status(e)})")
            quarterly_pe_dy.pop(sym, None)
            monthly_close.pop(sym, None)
            continue

        f_av = None
        if av_client:
            try:
                f_av = av_client.fetch_snapshot(sym)["fields"].get("eps_growth_forecast")
            except Exception as e:
                f_av = None
                print(f"Alpha Vantage {sym}: {e}")

        f_td = None
        if td_client:
            try:
                f_td = td_client.fetch_snapshot(sym)["fields"].get("eps_growth_forecast")
            except Exception as e:
                f_td = None
                print(f"Twelve Data {sym}: {e}")

        f_yf = None
        if yf_client:
            try:
                f_yf = yf_client.fetch_snapshot(sym)["fields"].get("eps_growth_forecast")
            except Exception as e:
                f_yf = None
                print(f"Yahoo Finance {sym}: {e}")

        f_sec = None
        if sec_client:
            try:
                f_sec = sec_client.fetch_snapshot(sym)["fields"].get("eps_growth_forecast")
            except Exception as e:
                f_sec = None
                print(f"SEC EDGAR {sym}: {e}")

        prov = json.dumps(
            {
                "fmp": snap.get("provenance", []),
            }
        )
        rows.append(
            {
                "ticker": sym,
                "eps_growth_fmp": f_fmp,
                "eps_growth_av": f_av,
                "eps_growth_td": f_td,
                "eps_growth_yf": f_yf,
                "eps_growth_sec": f_sec,
                "eps_growth_naive_trailing": realized,
                "eps_realized_fmp": realized,
                "pe_ttm_snapshot": pe0,
                "dividend_yield_snapshot": dy0,
                "provenance_json": prov,
            }
        )

    cross = pd.DataFrame(rows)
    tickers = cross["ticker"].tolist()

    # Daily wide prices for forward 12m (month-end grid)
    all_m = None
    for sym, s in monthly_close.items():
        if s is None or s.empty or not isinstance(s.index, pd.DatetimeIndex):
            continue
        all_m = s.index if all_m is None else all_m.union(s.index)
    if all_m is None or len(all_m) < 24:
        print("Insufficient price history for month-end grid.", file=sys.stderr)
        sys.exit(1)
    month_index = all_m.sort_values()

    price_wide = pd.DataFrame(index=month_index, columns=tickers, dtype=float)
    for sym in tickers:
        s = monthly_close.get(sym)
        if s is not None and not s.empty and isinstance(s.index, pd.DatetimeIndex):
            price_wide[sym] = s.reindex(month_index).values
    try:
        spy_series = _daily_prices_to_month_end(fmp, cfg.benchmark_symbol, args.from_date, args.to_date)
    except requests.HTTPError as e:
        print(
            f"Benchmark {cfg.benchmark_symbol}: FMP access failed "
            f"({_http_status(e)}); ML/signal targets will fall back to raw forward returns."
        )
        spy_series = pd.Series(dtype=float)
    benchmark_available = not spy_series.empty
    if benchmark_available:
        bench_wide = spy_series.reindex(month_index)
    else:
        bench_wide = pd.Series(1.0, index=month_index, dtype=float)

    fwd = forward_return_12m(price_wide)
    bench_fwd = forward_return_12m(pd.DataFrame({"SPY": bench_wide.reindex(month_index)}))
    bf_ser = bench_fwd["SPY"].dropna()
    snap_m = bf_ser.index[-1] if len(bf_ser) else month_index[-1]
    fwd_row = fwd.loc[snap_m]
    bench_xs = float(bench_fwd.loc[snap_m, "SPY"]) if snap_m in bench_fwd.index else float("nan")

    growth_maps = {
        "fmp": {r["ticker"]: r["eps_growth_fmp"] for _, r in cross.iterrows()},
        "av": {r["ticker"]: r["eps_growth_av"] for _, r in cross.iterrows()},
        "td": {r["ticker"]: r["eps_growth_td"] for _, r in cross.iterrows()},
        "yf": {r["ticker"]: r["eps_growth_yf"] for _, r in cross.iterrows()},
        "sec": {r["ticker"]: r["eps_growth_sec"] for _, r in cross.iterrows()},
        "naive_trailing": {r["ticker"]: r["eps_growth_naive_trailing"] for _, r in cross.iterrows()},
    }

    pe_wide, dy_wide = _build_monthly_fundamental_panels(month_index, tickers, quarterly_pe_dy)
    pegy_wides = {}
    pegy_snap = {}
    for prov_name, gmap in growth_maps.items():
        pw = _build_monthly_pegy_for_provider(month_index, tickers, gmap, quarterly_pe_dy)
        pegy_wides[prov_name] = pw
        pegy_snap[prov_name] = pw.loc[snap_m]

    if bench_xs == bench_xs:
        excess = fwd_row.astype(float) - bench_xs
    else:
        excess = pd.Series(np.nan, index=fwd_row.index)

    fc_cols = [
        c
        for c in ["eps_growth_fmp", "eps_growth_av", "eps_growth_td", "eps_growth_yf", "eps_growth_sec"]
        if c in cross.columns
    ]
    acc = forecast_accuracy_table(
        cross,
        forecast_cols=fc_cols,
        realized_col="eps_realized_fmp",
    )
    acc.to_csv(cfg.output_dir / "forecast_accuracy.csv", index=False)

    sig_rows = []
    for prov_name, pegy_s in pegy_snap.items():
        d = signal_ic_and_excess(pegy_s, excess.reindex(pegy_s.index))
        pr = precision_recall_top_performers(
            pegy_s,
            excess.reindex(pegy_s.index),
            pegy_threshold=cfg.pegy_threshold,
            top_q=cfg.top_performer_quantile,
        )
        sig_rows.append({"provider": prov_name, **d, **pr})
    sig_df = pd.DataFrame(sig_rows)
    sig_df.to_csv(cfg.output_dir / "signal_quality_snapshot.csv", index=False)

    ml_dataset = _build_ml_dataset(
        providers=growth_maps,
        month_index=month_index,
        tickers=tickers,
        pe_wide=pe_wide,
        dy_wide=dy_wide,
        pegy_wides=pegy_wides,
        fwd=fwd,
        bench_fwd=bench_fwd["SPY"],
        top_q=cfg.top_performer_quantile,
        benchmark_available=benchmark_available,
    )
    ml_results, ml_predictions = _run_ml_models(
        ml_dataset,
        top_q=cfg.top_performer_quantile,
        seed=cfg.random_seed,
    )
    ml_dataset.to_csv(cfg.output_dir / "ml_dataset.csv", index=False)
    ml_results.to_csv(cfg.output_dir / "ml_results.csv", index=False)
    ml_predictions.to_csv(cfg.output_dir / "ml_predictions.csv", index=False)

    monthly_rets = price_wide.pct_change()
    bench_mret = bench_wide.pct_change()

    bt_tables = []
    for prov_name, gmap in growth_maps.items():
        pw = pegy_wides[prov_name]
        bt = monthly_long_pegy_backtest(
            month_ends=month_index,
            pegy_wide=pw,
            monthly_returns=monthly_rets,
            pegy_threshold=cfg.pegy_threshold,
            benchmark_returns=bench_mret,
        )
        bt["provider"] = prov_name
        bt_tables.append(bt)
    bt_all = pd.concat(bt_tables, ignore_index=True)
    bt_all.to_csv(cfg.output_dir / "monthly_backtest.csv", index=False)

    # Cumulative excess by provider
    fig, ax = plt.subplots(figsize=(9, 5))
    for prov_name in growth_maps:
        sub = bt_all[bt_all["provider"] == prov_name].dropna(subset=["excess_1m"])
        if sub.empty:
            continue
        cum = (1 + sub["excess_1m"].astype(float)).cumprod() - 1
        ax.plot(sub["date"], cum, label=prov_name)
    ax.legend()
    ax.set_title("Cumulative 1m excess vs benchmark (equal-weight PEGY<1, monthly)")
    ax.set_ylabel("Cumulative excess (compounded monthly)")
    fig.tight_layout()
    fig.savefig(cfg.output_dir / "cumulative_excess_by_provider.png", dpi=150)
    plt.close()

    # Forecast error distribution (FMP vs realized as example driver)
    fig, axes = plt.subplots(1, len(fc_cols), figsize=(4 * max(len(fc_cols), 1), 4), squeeze=False)
    for ax, col in zip(axes[0], fc_cols):
        e = (cross[col] - cross["eps_realized_fmp"]).dropna().astype(float)
        ax.hist(e, bins=20, alpha=0.85)
        ax.set_title(col.replace("eps_growth_", ""))
        ax.set_xlabel("forecast - realized")
    fig.suptitle("EPS growth forecast error vs FMP realized YoY (snapshot)")
    fig.tight_layout()
    fig.savefig(cfg.output_dir / "forecast_error_hist.png", dpi=150)
    plt.close()

    cross.to_csv(cfg.output_dir / "cross_section_snapshot.csv", index=False)

    limitations = [
        "EPS growth from APIs is a contemporary snapshot, not point-in-time archives.",
        "S&P 500 universe is current list, not historical membership.",
        "Realized EPS is fiscal YoY from statements; misaligned vs calendar.",
    ]
    if not benchmark_available:
        limitations.append(
            f"{cfg.benchmark_symbol} benchmark data was unavailable; excess-return fields use raw returns as a fallback."
        )

    summary = {
        "n_tickers": len(tickers),
        "from_date": args.from_date,
        "to_date": args.to_date,
        "signal_month_end": str(snap_m.date()),
        "ml_rows": int(len(ml_dataset)),
        "ml_models": int((ml_results.get("status") == "ok").sum()) if not ml_results.empty else 0,
        "limitations": limitations,
    }
    (cfg.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(acc.to_string(index=False))
    print()
    print(sig_df.to_string(index=False))
    print()
    print(ml_results.to_string(index=False) if not ml_results.empty else "No ML results.")
    print(f"\nOutputs written to {cfg.output_dir}")


if __name__ == "__main__":
    main()
