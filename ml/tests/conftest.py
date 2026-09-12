"""Pytest configuration: make `mlpipeline` importable regardless of invocation cwd."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ML = ROOT / "ml"
for p in (str(ROOT), str(ML)):
    if p not in sys.path:
        sys.path.insert(0, p)
