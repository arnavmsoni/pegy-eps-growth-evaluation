"""Per-provider throttling to respect published rate limits."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class RateLimiter:
    min_interval_sec: float
    _last: Dict[str, float] = field(default_factory=dict)

    def wait(self, bucket: str = "default") -> None:
        now = time.monotonic()
        last = self._last.get(bucket, 0.0)
        elapsed = now - last
        if elapsed < self.min_interval_sec:
            time.sleep(self.min_interval_sec - elapsed)
        self._last[bucket] = time.monotonic()
