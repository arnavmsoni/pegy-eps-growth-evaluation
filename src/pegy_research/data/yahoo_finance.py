"""Yahoo Finance via yfinance: best-effort fundamentals snapshot."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import yfinance as yf

from pegy_research.schema import Provenance


def _safe_float(x: Any) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


class YahooFinanceClient:
    def fetch_snapshot(self, ticker: str) -> dict[str, Any]:
        t = ticker.upper()
        info = yf.Ticker(t).get_info()
        out: dict[str, Any] = {
            "ticker": t,
            "provider": "yfinance",
            "eps_growth_forecast": None,
            "pe_ttm": None,
            "dividend_yield_ttm": None,
            "eps_ttm_proxy": None,
        }

        growth = _safe_float(info.get("earningsGrowth"))
        if growth is None:
            trailing_eps = _safe_float(info.get("trailingEps"))
            forward_eps = _safe_float(info.get("forwardEps"))
            if trailing_eps is not None and forward_eps is not None and abs(trailing_eps) > 1e-9:
                growth = (forward_eps - trailing_eps) / abs(trailing_eps)
        out["eps_growth_forecast"] = growth

        out["pe_ttm"] = _safe_float(info.get("trailingPE"))
        dy = _safe_float(info.get("dividendYield"))
        if dy is not None:
            out["dividend_yield_ttm"] = dy if dy <= 0.20 else dy / 100.0
        out["eps_ttm_proxy"] = _safe_float(info.get("trailingEps"))

        prov = Provenance(
            "yfinance",
            "Ticker.get_info",
            datetime.now(timezone.utc).isoformat(),
            f"yfinance:{t}",
        )
        return {"fields": out, "provenance": [prov.to_dict()]}
