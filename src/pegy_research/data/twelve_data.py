"""Twelve Data: statistics + earnings where available."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

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


class TwelveDataClient:
    base = "https://api.twelvedata.com"

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
        self.limiter = RateLimiter(cfg.min_interval_twelve_data)

    def _get(self, path: str, params: dict[str, Any]) -> tuple[Any, str]:
        self.limiter.wait("twelve_data")
        p = {**params, "apikey": self.api_key}
        url = f"{self.base}{path}"
        key = cache_key("twelve_data", path, p)
        cached = self.cache.read_json(key)
        if cached is not None:
            return cached, key
        r = self.session.get(url, params=p, timeout=60)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and data.get("status") == "error":
            raise RuntimeError(data.get("message", str(data)))
        self.cache.write_json(key, data)
        append_provenance(
            self.cache,
            provider="twelve_data",
            endpoint=url,
            cache_key_str=key,
            status="ok",
            notes=path,
        )
        return data, key

    def fetch_snapshot(self, ticker: str) -> dict[str, Any]:
        prov: list[Provenance] = []
        t = ticker.upper()
        out: dict[str, Any] = {
            "ticker": t,
            "provider": "twelve_data",
            "eps_growth_forecast": None,
            "pe_ttm": None,
            "dividend_yield_ttm": None,
            "eps_ttm_proxy": None,
        }

        stats, k1 = self._get("/statistics", {"symbol": t})
        prov.append(Provenance("twelve_data", "/statistics", datetime.now(timezone.utc).isoformat(), k1))
        if isinstance(stats, dict):
            st = stats.get("statistics") or stats
            if isinstance(st, dict):
                out["pe_ttm"] = _safe_float(st.get("pe_ratio") or st.get("pe"))
                dy = _safe_float(st.get("dividend_yield"))
                if dy is not None:
                    out["dividend_yield_ttm"] = dy if dy <= 1.0 else dy / 100.0
                eps = _safe_float(st.get("eps") or st.get("eps_ttm"))
                if eps is not None:
                    out["eps_ttm_proxy"] = eps
                # Some plans expose growth fields
                eg = st.get("earnings_growth") or st.get("eps_growth")
                if eg is not None:
                    g = _safe_float(eg)
                    if g is not None:
                        out["eps_growth_forecast"] = g if abs(g) < 5 else g / 100.0

        earn, k2 = self._get("/earnings", {"symbol": t})
        prov.append(Provenance("twelve_data", "/earnings", datetime.now(timezone.utc).isoformat(), k2))
        if isinstance(earn, dict) and out["eps_growth_forecast"] is None:
            ed = earn.get("earnings") or earn.get("data")
            if isinstance(ed, list) and len(ed) >= 2:
                # try to infer YoY from last two actual EPS if present
                def eps_of(row: dict) -> Optional[float]:
                    return _safe_float(row.get("eps_actual") or row.get("eps"))

                rows = [x for x in ed if isinstance(x, dict)]
                vals = [eps_of(x) for x in rows[:8]]
                vals = [v for v in vals if v is not None]
                if len(vals) >= 2:
                    out["eps_growth_forecast"] = (vals[0] - vals[1]) / abs(vals[1]) if abs(vals[1]) > 1e-9 else None

        return {"fields": out, "provenance": [p.to_dict() for p in prov]}

    def fetch_time_series(self, ticker: str, start_date: str, end_date: str) -> list[dict]:
        """Daily adjusted close (OHLC) series."""
        t = ticker.upper()
        data, _ = self._get(
            "/time_series",
            {
                "symbol": t,
                "interval": "1day",
                "start_date": start_date,
                "end_date": end_date,
                "outputsize": "5000",
            },
        )
        if not isinstance(data, dict):
            return []
        vals = data.get("values")
        return vals if isinstance(vals, list) else []
