"""Financial Modeling Prep: estimates, ratios, prices, financial statements."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, List, Optional

import requests

from pegy_research.config import ResearchConfig
from pegy_research.data.cache import DiskCache, append_provenance, cache_key
from pegy_research.data.rate_limit import RateLimiter
from pegy_research.schema import Provenance


def _safe_float(x: Any) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


class FMPClient:
    base = "https://financialmodelingprep.com/stable"

    def __init__(
        self,
        api_key: str,
        cfg: ResearchConfig,
        cache: DiskCache,
        session: Optional[requests.Session] = None,
    ):
        self.api_key = api_key
        self.cfg = cfg
        self.cache = cache
        self.session = session or requests.Session()
        self.limiter = RateLimiter(cfg.min_interval_fmp)

    def _get_json(
        self,
        path: str,
        extra_params: dict[str, Any] | None = None,
        *,
        cache_path: str | None = None,
    ) -> tuple[Any, str]:
        self.limiter.wait("fmp")
        params = {"apikey": self.api_key, **(extra_params or {})}
        url = f"{self.base}{path}"
        key = cache_key("fmp", cache_path or path, params)
        cached = self.cache.read_json(key)
        if cached is not None:
            return cached, key
        r = self.session.get(url, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
        self.cache.write_json(key, data)
        append_provenance(
            self.cache,
            provider="fmp",
            endpoint=url,
            cache_key_str=key,
            status="ok",
            notes=cache_path or path,
        )
        return data, key

    def fetch_snapshot(self, ticker: str) -> dict[str, Any]:
        prov: list[Provenance] = []
        t = ticker.upper()
        out: dict[str, Any] = {
            "ticker": t,
            "provider": "fmp",
            "eps_growth_forecast": None,
            "pe_ttm": None,
            "dividend_yield_ttm": None,
            "eps_ttm_proxy": None,
        }

        ratios, k1 = self._get_json("/ratios-ttm", {"symbol": t}, cache_path=f"/ratios-ttm/{t}")
        prov.append(Provenance("fmp", "ratios-ttm", datetime.now(timezone.utc).isoformat(), k1))
        if isinstance(ratios, list) and ratios:
            r0 = ratios[0]
            out["pe_ttm"] = _safe_float(
                r0.get("priceEarningsRatioTTM") or r0.get("priceToEarningsRatioTTM")
            )
            dy = _safe_float(r0.get("dividendYieldTTM"))
            if dy is not None:
                out["dividend_yield_ttm"] = dy if dy <= 1.0 else dy / 100.0

        km, k2 = self._get_json("/key-metrics-ttm", {"symbol": t}, cache_path=f"/key-metrics-ttm/{t}")
        prov.append(Provenance("fmp", "key-metrics-ttm", datetime.now(timezone.utc).isoformat(), k2))
        if isinstance(km, list) and km:
            m0 = km[0]
            if out["pe_ttm"] is None:
                out["pe_ttm"] = _safe_float(m0.get("peRatioTTM"))
            eps = _safe_float(m0.get("netIncomePerShareTTM"))
            if eps is not None:
                out["eps_ttm_proxy"] = eps

        est, k3 = self._get_json(
            "/analyst-estimates",
            {"symbol": t, "period": "annual"},
            cache_path=f"/analyst-estimates/{t}",
        )
        prov.append(Provenance("fmp", "analyst-estimates", datetime.now(timezone.utc).isoformat(), k3))
        if isinstance(est, list) and len(est) >= 2:
            est_sorted: List[dict] = sorted(
                [e for e in est if e.get("date")],
                key=lambda e: str(e.get("date")),
            )
            fy0 = _safe_float(est_sorted[0].get("estimatedEpsAvg") or est_sorted[0].get("epsAvg"))
            fy1 = _safe_float(est_sorted[1].get("estimatedEpsAvg") or est_sorted[1].get("epsAvg"))
            if fy0 and fy1 and abs(fy0) > 1e-9:
                out["eps_growth_forecast"] = (fy1 - fy0) / abs(fy0)

        return {"fields": out, "provenance": [p.to_dict() for p in prov]}

    def fetch_historical_prices(self, ticker: str, from_date: str, to_date: str) -> list[dict]:
        """Daily adjusted close from FMP historical-price-full."""
        t = ticker.upper()
        data, _ = self._get_json(
            "/historical-price-eod/full",
            {"symbol": t, "from": from_date, "to": to_date},
            cache_path=f"/historical-price-full/{t}",
        )
        if isinstance(data, list):
            return data
        if not isinstance(data, dict):
            return []
        hist = data.get("historical") or []
        return hist if isinstance(hist, list) else []

    def trailing_eps_yoy_growth(self, ticker: str, limit: int = 8) -> Optional[float]:
        """
        Realized YoY EPS growth from annual income statements (GAAP EPS).
        Uses two most recent reported annual EPS values in the payload.
        """
        t = ticker.upper()
        free_plan_limit = min(limit, 5)
        inc, _ = self._get_json(
            "/income-statement",
            {"symbol": t, "period": "annual", "limit": free_plan_limit},
            cache_path=f"/income-statement/{t}",
        )
        if not isinstance(inc, list) or len(inc) < 2:
            return None
        sorted_inc = sorted(inc, key=lambda r: str(r.get("date") or r.get("fillingDate") or ""))
        e0 = _safe_float(sorted_inc[-2].get("eps"))
        e1 = _safe_float(sorted_inc[-1].get("eps"))
        if e0 is None or e1 is None or abs(e0) < 1e-9:
            return None
        return (e1 - e0) / abs(e0)

    def fetch_ratios_quarterly(self, ticker: str, limit: int = 80) -> list[dict]:
        """Ratios for time-varying P/E and dividend yield (point-in-time proxy).

        FMP's stable quarterly ratio endpoint is plan-gated for this account, so
        this uses annual ratios as the available historical valuation proxy.
        """
        t = ticker.upper()
        free_plan_limit = min(limit, 5)
        data, _ = self._get_json(
            "/ratios",
            {"symbol": t, "limit": free_plan_limit},
            cache_path=f"/ratios/{t}",
        )
        return data if isinstance(data, list) else []
