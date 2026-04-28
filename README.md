# When Does PEG Fail? – Predicting Reliability of the PEG Ratio

Python pipeline that builds a **quarterly** panel (S&P 500 or a cap-ranked subset), keeps names with **PEG &lt; 1.5**, and trains **sklearn** models to classify whether a low-PEG name **beats SPY** over the next 12 months on a **monthly price** grid.

**Data:** `yfinance` only (prices, quarterly statements for margins / leverage, `get_earnings_dates` for TTM EPS and YoY growth). No deep learning.

## Setup

```bash
cd pegy-eps-growth-evaluation
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Run

```bash
export PYTHONPATH=src   # or rely on `pip install -e .`
python -m pegy_research.main --max-tickers 100 --years 5
```

Outputs: console metrics, `outputs/decision_tree.png`, cached downloads under `data/cache/`.

**Flags:** `--peg-max`, `--train-fraction`, `--refresh-prices`, `--refresh-fundamentals`, `--output-dir`, `--cache-dir`.

## Layout

| Module | Role |
|--------|------|
| `data_loader.py` | S&P 500 list, price panel, statements cache, **earnings history** |
| `feature_engineering.py` | PIT prices, TTM EPS, growth, PEG, optional vol / margins / D/E |
| `build_dataset.py` | Labels vs SPY, PEG filter, outlier trim |
| `train_models.py` | Time-ordered split; tree, logistic regression, random forest |
| `evaluate.py` | Accuracy / precision / recall, backtest A vs B, tree plot & rules |
| `main.py` | CLI |

## Caveats

Point-in-time earnings use **announcement dates** in `get_earnings_dates`; statement rows are the usual yfinance **current** snapshot (limited history). Results are for research / education, not live trading.
