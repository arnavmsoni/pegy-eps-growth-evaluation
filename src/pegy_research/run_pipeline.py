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
from pegy_research.data.twelve_data import TwelveDataClient
from pegy_research.data.universe import load_universe_tickers
from pegy_research.evaluation.forecast_accuracy import forecast_accuracy_table
from pegy_research.evaluation.signal_quality import precision_recall_top_performers, signal_ic_and_excess
from pegy_research.features.pegy import compute_pegy


def _quarterly_ratios_df(fmp: FMPClient, ticker: str) -> pd.DataFrame:
    rows = fmp.fetch_ratios_quarterly(ticker)
    out = []
    for r in rows:
        d = r.get("date")
        if not d:
            continue
        pe = _safe_float(r.get("priceEarningsRatio") or r.get("peRatio") or r.get("peRatioTTM"))
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


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    p = argparse.ArgumentParser(description="PEGY EPS growth provider evaluation")
    p.add_argument("--from-date", default="2018-01-01")
    p.add_argument("--to-date", default="2024-12-31")
    p.add_argument("--max-tickers", type=int, default=35, help="Subset for API cost control")
    p.add_argument("--skip-av", action="store_true")
    p.add_argument("--skip-twelve", action="store_true")
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

    rows = []
    quarterly_pe_dy: dict[str, pd.DataFrame] = {}
    monthly_close: dict[str, pd.Series] = {}

    for sym in tickers:
        qdf = _quarterly_ratios_df(fmp, sym)
        quarterly_pe_dy[sym] = qdf
        mc = _daily_prices_to_month_end(fmp, sym, args.from_date, args.to_date)
        monthly_close[sym] = mc

        snap = fmp.fetch_snapshot(sym)
        f_fmp = snap["fields"].get("eps_growth_forecast")

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

        realized = fmp.trailing_eps_yoy_growth(sym)
        pe0 = snap["fields"].get("pe_ttm")
        dy0 = snap["fields"].get("dividend_yield_ttm")
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
                "eps_growth_naive_trailing": realized,
                "eps_realized_fmp": realized,
                "pe_ttm_snapshot": pe0,
                "dividend_yield_snapshot": dy0,
                "provenance_json": prov,
            }
        )

    cross = pd.DataFrame(rows)

    # Daily wide prices for forward 12m (month-end grid)
    all_m = None
    for sym, s in monthly_close.items():
        all_m = s.index if all_m is None else all_m.union(s)
    if all_m is None or len(all_m) < 24:
        print("Insufficient price history for month-end grid.", file=sys.stderr)
        sys.exit(1)
    month_index = all_m.sort_values()

    price_wide = pd.DataFrame(index=month_index, columns=tickers, dtype=float)
    for sym in tickers:
        s = monthly_close.get(sym)
        if s is not None and not s.empty:
            price_wide[sym] = s.reindex(month_index).values
    spy_series = _daily_prices_to_month_end(fmp, cfg.benchmark_symbol, args.from_date, args.to_date)
    bench_wide = spy_series.reindex(month_index)

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
        "naive_trailing": {r["ticker"]: r["eps_growth_naive_trailing"] for _, r in cross.iterrows()},
    }

    pegy_snap = {}
    for prov_name, gmap in growth_maps.items():
        pw = _build_monthly_pegy_for_provider(month_index, tickers, gmap, quarterly_pe_dy)
        pegy_snap[prov_name] = pw.loc[snap_m]

    if bench_xs == bench_xs:
        excess = fwd_row.astype(float) - bench_xs
    else:
        excess = pd.Series(np.nan, index=fwd_row.index)

    fc_cols = [c for c in ["eps_growth_fmp", "eps_growth_av", "eps_growth_td"] if c in cross.columns]
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

    monthly_rets = price_wide.pct_change()
    bench_mret = bench_wide.pct_change()

    bt_tables = []
    for prov_name, gmap in growth_maps.items():
        pw = _build_monthly_pegy_for_provider(month_index, tickers, gmap, quarterly_pe_dy)
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

    summary = {
        "n_tickers": len(tickers),
        "from_date": args.from_date,
        "to_date": args.to_date,
        "signal_month_end": str(snap_m.date()),
        "limitations": [
            "EPS growth from APIs is a contemporary snapshot, not point-in-time archives.",
            "S&P 500 universe is current list, not historical membership.",
            "Realized EPS is fiscal YoY from statements; misaligned vs calendar.",
        ],
    }
    (cfg.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(acc.to_string(index=False))
    print()
    print(sig_df.to_string(index=False))
    print(f"\nOutputs written to {cfg.output_dir}")


if __name__ == "__main__":
    main()
