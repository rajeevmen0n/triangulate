"""Shared HTTP etiquette helpers for Nominatim / Overpass: rate limiter and an on-disk JSON cache."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any


class RateLimiter:
    """At most one call per `min_interval` seconds, per limiter instance (thread-safe)."""

    def __init__(self, min_interval: float, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self.min_interval = float(min_interval)
        self._clock, self._sleep = clock, sleep
        self._last: float | None = None
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            if self._last is not None:
                delay = self.min_interval - (self._clock() - self._last)
                if delay > 0:
                    self._sleep(delay)
            self._last = self._clock()


# One limiter per service host, shared by every client instance in the process, so two jobs in the
# same process still respect the policy together.
_SHARED: dict[str, RateLimiter] = {}
_SHARED_LOCK = threading.Lock()


def shared_limiter(key: str, min_interval: float) -> RateLimiter:
    with _SHARED_LOCK:
        lim = _SHARED.get(key)
        if lim is None or lim.min_interval != min_interval:
            lim = _SHARED[key] = RateLimiter(min_interval)
        return lim


class JsonCache:
    """key -> JSON file under `root`. Keys are hashed; the request is stored next to the response."""

    def __init__(self, root: Path):
        self.root = Path(root)

    @staticmethod
    def key(obj: Any) -> str:
        return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=True).encode()).hexdigest()[:40]

    def get(self, key: str) -> Any | None:
        p = self.root / f"{key}.json"
        try:
            return json.loads(p.read_text())["response"]
        except (OSError, ValueError, KeyError):
            return None

    def put(self, key: str, request: Any, response: Any) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.root / f"{key}.json.tmp"
        tmp.write_text(json.dumps({"request": request, "fetched_at": time.time(), "response": response}))
        tmp.replace(self.root / f"{key}.json")
