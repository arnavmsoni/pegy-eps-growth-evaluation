"""
Normalized longitudinal schema for cross-provider comparison.

All dates are calendar dates in America/New_York trading context unless noted.
EPS growth is stored as a decimal (e.g. 0.12 for 12% YoY expected growth).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Optional


# Column names for the unified panel (pandas-friendly)
COL_TICKER = "ticker"
COL_SIGNAL_DATE = "signal_date"
COL_PROVIDER = "provider"
COL_EPS_GROWTH_FORECAST = "eps_growth_forecast"
COL_EPS_REALIZED_GROWTH = "eps_realized_growth"
COL_PE = "pe_ttm"
COL_DIV_YIELD = "dividend_yield_ttm"
COL_PRICE = "adj_close"
COL_PEGY = "pegy"
COL_BENCHMARK_RETURN_FWD = "benchmark_fwd_return_12m"
COL_STOCK_RETURN_FWD = "stock_fwd_return_12m"
COL_EXCESS_RETURN_FWD = "excess_fwd_return_12m"
COL_TOP_PERFORMER = "top_performer_30pct"  # cross-sectional label at horizon
COL_PROVENANCE = "provenance_json"


@dataclass
class Provenance:
    """Audit trail for a single ingested field or row."""

    source: str
    endpoint: str
    fetched_at_utc: str
    cache_key: str
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "endpoint": self.endpoint,
            "fetched_at_utc": self.fetched_at_utc,
            "cache_key": self.cache_key,
            "notes": self.notes,
        }


@dataclass
class UnifiedBar:
    """One observation: one ticker, one signal date, one provider."""

    ticker: str
    signal_date: date
    provider: str
    eps_growth_forecast: Optional[float] = None
    eps_realized_growth: Optional[float] = None
    pe_ttm: Optional[float] = None
    dividend_yield_ttm: Optional[float] = None
    adj_close: Optional[float] = None
    pegy: Optional[float] = None
    provenance: list[Provenance] = field(default_factory=list)
