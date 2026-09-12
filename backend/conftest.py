"""Backend pytest bootstrap: make `mlpipeline` and repo-root imports resolvable."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "ml")):
    if p not in sys.path:
        sys.path.insert(0, p)
