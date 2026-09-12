"""Per-stage tracing: timings collected into StageTimings (Section 16)."""
from __future__ import annotations

import time
from contextlib import contextmanager

from mlpipeline.datatypes.results import StageTimings


class StageTracer:
    """Collects wall-clock per stage. Lightweight; no global state."""

    def __init__(self) -> None:
        self.timings = StageTimings()

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            ms = (time.perf_counter() - t0) * 1000.0
            setattr(self.timings, name, round(getattr(self.timings, name) + ms, 3))
