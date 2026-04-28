#!/usr/bin/env python3
"""
Write current S&P 500 tickers from Wikipedia to a text file (no paid API).

Usage:
  python scripts/fetch_sp500_universe_wikipedia.py -o examples/universe_sp500.txt

Requires: pandas (reads HTML table). Wikipedia structure may change — verify output.
"""

from __future__ import annotations

import argparse
import re
import sys


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("-o", "--output", default="examples/universe_sp500.txt")
    args = p.parse_args()
    try:
        import pandas as pd
    except ImportError:
        print("Install pandas: pip install pandas", file=sys.stderr)
        sys.exit(1)

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    if not tables:
        print("No tables found on Wikipedia page.", file=sys.stderr)
        sys.exit(1)
    df = tables[0]
    col = "Symbol" if "Symbol" in df.columns else df.columns[0]
    raw = df[col].astype(str).tolist()
    tickers = []
    for s in raw:
        s = str(s).strip()
        s = re.sub(r"\.0$", "", s)
        s = s.replace(".", "-")
        if s:
            tickers.append(s.upper())
    tickers = sorted(set(tickers))
    out = "# S&P 500 constituents from Wikipedia (run script to refresh)\n"
    out += "\n".join(tickers) + "\n"
    path = args.output
    with open(path, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"Wrote {len(tickers)} tickers to {path}")


if __name__ == "__main__":
    main()
