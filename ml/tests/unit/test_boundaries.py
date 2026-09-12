"""ML-side boundary checks (Step 18). Complements backend/tests/unit/test_boundaries.py.

Covers rules that concern the ML package itself:
- HTTP-free (no fastapi/uvicorn/starlette imports) — the backend owns HTTP.
- Determinism: preprocessing must be reproducible (config hash stability).
- Honesty: no fabricated accuracy/metrics constants in pipeline code.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
ML_ROOT = REPO_ROOT / "ml" / "mlpipeline"

FORBIDDEN_IN_ML = ("fastapi", "uvicorn", "starlette", "httpx", "requests")


def _py_files() -> list[Path]:
    return [p for p in ML_ROOT.rglob("*.py") if "__pycache__" not in p.parts]


def test_ml_package_is_http_free():
    offenders = []
    for p in _py_files():
        text = p.read_text(encoding="utf-8", errors="replace")
        for mod in FORBIDDEN_IN_ML:
            if re.search(rf"^\s*(import|from)\s+{mod}\b", text, re.MULTILINE):
                offenders.append(f"{p.name}: {mod}")
    assert offenders == [], f"mlpipeline imports HTTP frameworks: {offenders}"


def test_no_fabricated_metric_constants():
    """Evaluation metrics must be computed from data, not asserted as literals
    (Section 18: synthetic claims are forbidden)."""
    offenders = []
    pattern = re.compile(r"(accuracy|precision|recall|f1|mAP)\s*[=:]\s*(0\.\d+|99\.\d+)", re.IGNORECASE)
    for p in _py_files():
        text = p.read_text(encoding="utf-8", errors="replace")
        for m in pattern.finditer(text):
            line = text[: m.start()].count("\n") + 1
            offenders.append(f"{p.name}:{line}: {m.group(0)}")
    assert offenders == [], f"hardcoded metric values in pipeline code: {offenders}"


def test_preprocess_config_hash_is_stable():
    from mlpipeline.config import load_config_file
    from mlpipeline.config.schemas import PreprocessConfig

    path = REPO_ROOT / "ml" / "configs" / "preprocessing" / "baseline_sonar.yaml"
    _, h1 = load_config_file(path, PreprocessConfig)
    _, h2 = load_config_file(path, PreprocessConfig)
    assert h1 == h2 and len(h1) == 64


def test_stub_detector_is_labelled():
    """The stub detector must self-identify as a TEST fixture (Section 4 rule:
    test/demo fixtures must be clearly marked, never presented as real)."""
    stub = (ML_ROOT / "detection" / "stub.py").read_text()
    assert "test" in stub.lower() or "stub" in stub.lower()
