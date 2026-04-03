# PEGY EPS Growth Source Evaluation

Research pipeline comparing **PEGY** variants built from different EPS growth vendors (Alpha Vantage, Financial Modeling Prep, Twelve Data) plus an optional **naive trailing** baseline from reported financials.

## Architecture (modular)

| Layer | Role |
|--------|------|
| **Data** | Per-provider clients, SHA-keyed JSON cache, `provenance.jsonl`, rate limiting |
| **Features** | Harmonized yields, \( \mathrm{PEGY} = P/E \div (g + \mathrm{div\ yield}) \) |
| **Backtest** | Monthly equal-weight long of names with PEGY \< 1; time-varying P/E and yield from **quarterly** ratios (`merge_asof` backward); EPS growth fixed per run from snapshot |
| **Evaluation** | Forecast table (MAE, RMSE, bias, corr, directional accuracy) vs FMP realized YoY EPS; signal IC, precision/recall for top forward-excess quintile |

**Validity caveats (by design):** vendor EPS growth is a *contemporary API snapshot*, not a point-in-time estimate archive; the universe is the current S&P 500 list unless `PEGY_TICKERS` is set; realized EPS is fiscal YoY from statements and may be misaligned across names. Interpret forecast metrics as *relative* calibration, not literal ex-ante forecast error.

## Setup

```bash
cd pegy-eps-growth-evaluation
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add FMP_API_KEY (required); optional AV / Twelve Data
```

## Run

```bash
export PYTHONPATH=src
python -m pegy_research.run_pipeline --from-date 2018-01-01 --to-date 2024-12-31 --max-tickers 40
```

- **Universe:** FMP `/v3/sp500_constituent` (cached), or override with `PEGY_TICKERS=AAPL,MSFT,...`.
- **Outputs** under `outputs/`: `forecast_accuracy.csv`, `signal_quality_snapshot.csv`, `monthly_backtest.csv`, `cross_section_snapshot.csv`, `cumulative_excess_by_provider.png`, `forecast_error_hist.png`, `run_summary.json`.

Flags: `--skip-av`, `--skip-twelve` if keys are absent.

## Data Sources

- **Alpha Vantage** — `OVERVIEW` growth / valuation fields  
- **FMP** — analyst estimates, ratios TTM, quarterly ratios (historical P/E & yield), prices, income statement for realized EPS  
- **Twelve Data** — `statistics` / `earnings` (best-effort growth)