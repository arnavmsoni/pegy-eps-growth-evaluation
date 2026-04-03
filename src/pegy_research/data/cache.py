"""Deterministic disk cache and append-only provenance log."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests


def _stable_json(obj: dict[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def cache_key(provider: str, endpoint: str, params: dict[str, Any]) -> str:
    raw = f"{provider}|{endpoint}|{_stable_json(params)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


class DiskCache:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.provenance_path = self.root / "provenance.jsonl"

    def path_for(self, key: str, suffix: str = "json") -> Path:
        return self.root / f"{key}.{suffix}"

    def read_json(self, key: str) -> Optional[Any]:
        p = self.path_for(key, "json")
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def write_json(self, key: str, payload: Any) -> None:
        p = self.path_for(key, "json")
        p.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def get_or_fetch_json(
        self,
        *,
        provider: str,
        endpoint: str,
        url: str,
        params: dict[str, Any],
        session: requests.Session,
        timeout: int = 60,
    ) -> tuple[Any, str]:
        key = cache_key(provider, endpoint, params)
        cached = self.read_json(key)
        if cached is not None:
            return cached, key
        resp = session.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        self.write_json(key, data)
        return data, key


def append_provenance(
    cache: DiskCache,
    *,
    provider: str,
    endpoint: str,
    cache_key_str: str,
    status: str,
    notes: str = "",
) -> None:
    row = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "endpoint": endpoint,
        "cache_key": cache_key_str,
        "status": status,
        "notes": notes,
    }
    with cache.provenance_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
