"""Alpha Vantage: OVERVIEW + EARNINGS for EPS path and growth proxies."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import requests

from pegy_research.config import ResearchConfig
from pegy_research.data.cache import DiskCache, append_provenance, cache_key
from pegy_research.data.rate_limit import RateLimiter
from pegy_research.schema import Provenance


class AlphaVantageClient:
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
        self.limiter = RateLimiter(cfg.min_interval_alpha_vantage)

    def _get(self, params: dict[str, Any]) -> tuple[Any, str]:
        self.limiter.wait("alpha_vantage")
        base = "https://www.alphavantage.co/query"
        p = {**params, "apikey": self.api_key}
        endpoint = str(params.get("function", "unknown"))
        key = cache_key("alpha_vantage", endpoint, {k: v for k, v in p.items() if k != "apikey"})
        cached = self.cache.read_json(key)
        if cached is not None:
            return cached, key
        r = self.session.get(base, params=p, timeout=60)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and data.get("Note"):
            raise RuntimeError(f"Alpha Vantage rate note: {data.get('Note')}")
        self.cache.write_json(key, data)
        append_provenance(
            self.cache,
            provider="alpha_vantage",
            endpoint=base,
            cache_key_str=key,
            status="ok",
            notes=endpoint,
        )
        return data, key

    def fetch_snapshot(self, ticker: str) -> dict[str, Any]:
        """Return normalized fields + provenance for one ticker (best-effort)."""
        prov: list[Provenance] = []
        out: dict[str, Any] = {
            "ticker": ticker.upper(),
            "provider": "alpha_vantage",
            "eps_growth_forecast": None,
            "pe_ttm": None,
            "dividend_yield_ttm": None,
            "eps_ttm_proxy": None,
        }
        ov, k1 = self._get({"function": "OVERVIEW", "symbol": ticker})
        prov.append(
            Provenance(
                "alpha_vantage",
                "OVERVIEW",
                datetime.now(timezone.utc).isoformat(),
                k1,
            )
        )
        if isinstance(ov, dict):
            # QuarterlyEarningsGrowthYOY is a percentage string in AV docs
            qeg = ov.get("QuarterlyEarningsGrowthYOY")
            if qeg not in (None, "None"):
                try:
                    out["eps_growth_forecast"] = float(qeg) / 100.0
                except (TypeError, ValueError):
                    pass
            pe = ov.get("PERatio") or ov.get("PE")
            if pe not in (None, "None"):
                try:
                    out["pe_ttm"] = float(pe)
                except (TypeError, ValueError):
                    pass
            dy = ov.get("DividendYield")
            if dy not in (None, "None"):
                try:
                    # AV often returns decimal fraction
                    d = float(dy)
                    out["dividend_yield_ttm"] = d if d <= 1.0 else d / 100.0
                except (TypeError, ValueError):
                    pass
            eps = ov.get("EPS")
            if eps not in (None, "None"):
                try:
                    out["eps_ttm_proxy"] = float(eps)
                except (TypeError, ValueError):
                    pass

        return {"fields": out, "provenance": [p.to_dict() for p in prov]}
