"""
Central configuration: reproducibility, paths, and API pacing.

Universe: default is S&P 500 constituents from FMP (cached) to avoid ad-hoc ticker lists.
Override with env PEGY_TICKERS for controlled experiments.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ResearchConfig:
    random_seed: int = 42
    cache_dir: Path = field(default_factory=lambda: _repo_root() / "data" / "cache")
    output_dir: Path = field(default_factory=lambda: _repo_root() / "outputs")
    # Minimum seconds between HTTP calls per provider (conservative defaults)
    min_interval_alpha_vantage: float = 12.5  # ~5 calls/min free tier
    min_interval_fmp: float = 0.25
    min_interval_twelve_data: float = 8.0
    # Backtest
    pegy_threshold: float = 1.0
    top_performer_quantile: float = 0.30
    benchmark_symbol: str = "SPY"

    @classmethod
    def from_env(cls) -> "ResearchConfig":
        root = _repo_root()
        return cls(
            random_seed=int(os.environ.get("PEGY_SEED", "42")),
            cache_dir=Path(os.environ.get("PEGY_CACHE_DIR", str(root / "data" / "cache"))),
            output_dir=Path(os.environ.get("PEGY_OUTPUT_DIR", str(root / "outputs"))),
        )


def apply_seed(cfg: ResearchConfig) -> None:
    np.random.seed(cfg.random_seed)
