"""SEC EDGAR Company Facts: reported EPS actuals baseline."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import requests

from pegy_research.data.cache import DiskCache, append_provenance, cache_key
from pegy_research.schema import Provenance


def _safe_float(x: Any) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


class SECEdgarClient:
    tickers_url = "https://www.sec.gov/files/company_tickers.json"
    facts_url = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

    def __init__(
        self,
        cache: DiskCache,
        session: Optional[requests.Session] = None,
        *,
        user_agent: str = "pegy-research/0.1 contact@example.com",
    ):
        self.cache = cache
        self.session = session or requests.Session()
        self.user_agent = user_agent
        self._ticker_map: dict[str, str] | None = None

    def _get_json(self, url: str, key_prefix: str) -> tuple[Any, str]:
        key = cache_key("sec", key_prefix, {"url": url})
        cached = self.cache.read_json(key)
        if cached is not None:
            return cached, key
        r = self.session.get(url, headers={"User-Agent": self.user_agent}, timeout=60)
        r.raise_for_status()
        data = r.json()
        self.cache.write_json(key, data)
        append_provenance(
            self.cache,
            provider="sec",
            endpoint=url,
            cache_key_str=key,
            status="ok",
            notes=key_prefix,
        )
        return data, key

    def _load_ticker_map(self) -> dict[str, str]:
        if self._ticker_map is not None:
            return self._ticker_map
        data, _ = self._get_json(self.tickers_url, "company_tickers")
        mapping: dict[str, str] = {}
        if isinstance(data, dict):
            for row in data.values():
                if not isinstance(row, dict):
                    continue
                ticker = str(row.get("ticker", "")).upper()
                cik = row.get("cik_str")
                if ticker and cik is not None:
                    mapping[ticker] = str(cik).zfill(10)
        self._ticker_map = mapping
        return mapping

    def _eps_growth_from_facts(self, facts: dict[str, Any]) -> Optional[float]:
        us_gaap = facts.get("facts", {}).get("us-gaap", {})
        eps_fact = us_gaap.get("EarningsPerShareDiluted") or us_gaap.get("EarningsPerShareBasic")
        if not isinstance(eps_fact, dict):
            return None
        units = eps_fact.get("units", {})
        rows: list[dict[str, Any]] = []
        for unit_rows in units.values():
            if not isinstance(unit_rows, list):
                continue
            for row in unit_rows:
                if not isinstance(row, dict):
                    continue
                if row.get("form") not in {"10-K", "10-K/A"}:
                    continue
                if row.get("fp") not in {None, "FY"}:
                    continue
                val = _safe_float(row.get("val"))
                fy = row.get("fy")
                end = row.get("end")
                if val is None or fy is None or not end:
                    continue
                rows.append({"fy": int(fy), "end": str(end), "val": val})
        if len(rows) < 2:
            return None
        by_fy: dict[int, dict[str, Any]] = {}
        for row in sorted(rows, key=lambda r: (r["fy"], r["end"])):
            by_fy[row["fy"]] = row
        vals = [by_fy[fy]["val"] for fy in sorted(by_fy)]
        if len(vals) < 2:
            return None
        prev, latest = vals[-2], vals[-1]
        if abs(prev) < 1e-9:
            return None
        return (latest - prev) / abs(prev)

    def fetch_snapshot(self, ticker: str) -> dict[str, Any]:
        t = ticker.upper()
        mapping = self._load_ticker_map()
        cik = mapping.get(t)
        out = {
            "ticker": t,
            "provider": "sec",
            "eps_growth_forecast": None,
            "pe_ttm": None,
            "dividend_yield_ttm": None,
            "eps_ttm_proxy": None,
        }
        prov: list[Provenance] = []
        if not cik:
            return {"fields": out, "provenance": []}
        url = self.facts_url.format(cik=cik)
        facts, key = self._get_json(url, f"companyfacts/{cik}")
        prov.append(Provenance("sec", "companyfacts", datetime.now(timezone.utc).isoformat(), key))
        if isinstance(facts, dict):
            out["eps_growth_forecast"] = self._eps_growth_from_facts(facts)
        return {"fields": out, "provenance": [p.to_dict() for p in prov]}
