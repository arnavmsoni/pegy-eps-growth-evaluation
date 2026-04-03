"""
Investable universe loader.

Justification: using FMP's published S&P 500 constituent list (cached) avoids
embedding an unmaintained hardcoded ticker array while still anchoring to a
standard benchmark universe. For full survivorship-bias control, a point-in-time
constituent database would be required; we document that limitation in outputs.
"""

from __future__ import annotations

import os
from typing import List

import requests

from pegy_research.config import ResearchConfig
from pegy_research.data.cache import DiskCache, append_provenance, cache_key


def load_universe_tickers(
    cfg: ResearchConfig,
    cache: DiskCache,
    session: requests.Session,
    fmp_api_key: str | None,
) -> List[str]:
    override = os.environ.get("PEGY_TICKERS", "").strip()
    if override:
        tickers = [t.strip().upper() for t in override.split(",") if t.strip()]
        append_provenance(
            cache,
            provider="config",
            endpoint="PEGY_TICKERS",
            cache_key_str="manual_override",
            status="ok",
            notes=f"count={len(tickers)}",
        )
        return sorted(set(tickers))

    if not fmp_api_key:
        raise RuntimeError(
            "FMP_API_KEY is required to fetch S&P 500 constituents unless "
            "PEGY_TICKERS is set (comma-separated)."
        )

    endpoint = "sp500_constituent"
    url = "https://financialmodelingprep.com/api/v3/sp500_constituent"
    params = {"apikey": fmp_api_key}
    key = cache_key("fmp", endpoint, {"endpoint": endpoint})
    cached = cache.read_json(key)
    if cached is None:
        from pegy_research.data.rate_limit import RateLimiter

        lim = RateLimiter(cfg.min_interval_fmp)
        lim.wait("fmp")
        r = session.get(url, params=params, timeout=60)
        r.raise_for_status()
        cached = r.json()
        cache.write_json(key, cached)
        append_provenance(
            cache,
            provider="fmp",
            endpoint=url,
            cache_key_str=key,
            status="ok",
            notes="sp500_constituent",
        )
    symbols = [row["symbol"] for row in cached if row.get("symbol")]
    return sorted(set(symbols))
