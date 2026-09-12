"""Architecture boundary tests (Step 18, Section 25 gate).

Enforces the hard rules from docs/ARCHITECTURE.md:
- ml/ MUST NOT import FastAPI/HTTP frameworks (ADR-001 module boundary).
- backend MUST NOT contain ML algorithms (it orchestrates mlpipeline only).
- No hardcoded target class names in application logic (Section 5 dataset rule).
"""
from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

FORBIDDEN_IN_ML = ("fastapi", "uvicorn", "starlette", "httpx", "requests")
# Backend may not define detectors/filters itself — it may only call mlpipeline.
FORBIDDEN_BACKEND_IMPORTS = ("ultralytics", "torch", "cv2.experimental")
# Architecture examples that must never become hardcoded application classes
# (Section 5: classes come from dataset/model metadata only).
HARDCODED_CLASS_NAMES = {"shipwreck", "ghost_net", "fishing_net", "sea_litter"}


def _py_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*.py") if "__pycache__" not in p.parts]


class TestMlIndependence:
    def test_ml_never_imports_http_frameworks(self):
        """ADR-001: the ML package is HTTP-free and framework-agnostic."""
        offenders = []
        for p in _py_files(REPO_ROOT / "ml"):
            text = p.read_text(encoding="utf-8", errors="replace")
            for mod in FORBIDDEN_IN_ML:
                if re.search(rf"^\s*(import|from)\s+{mod}\b", text, re.MULTILINE):
                    offenders.append(f"{p.name}: {mod}")
        assert offenders == [], f"ml/ imports HTTP frameworks: {offenders}"

    def test_ml_has_no_route_decorators(self):
        """No HTTP route definitions may leak into the ML package."""
        offenders = []
        for p in _py_files(REPO_ROOT / "ml"):
            text = p.read_text(encoding="utf-8", errors="replace")
            if re.search(r"@(app|router)\.(get|post|put|patch|delete)\b", text):
                offenders.append(p.name)
        assert offenders == [], f"route decorators in ml/: {offenders}"


class TestBackendRole:
    def test_backend_has_no_framework_leakage_from_ml_side(self):
        """Backend orchestrates mlpipeline; it must not re-implement model
        inference (no ultralytics/torch imports outside requirements)."""
        offenders = []
        for p in _py_files(REPO_ROOT / "backend"):
            text = p.read_text(encoding="utf-8", errors="replace")
            for mod in FORBIDDEN_BACKEND_IMPORTS:
                if re.search(rf"^\s*(import|from)\s+{mod}\b", text, re.MULTILINE):
                    offenders.append(f"{p.name}: {mod}")
        assert offenders == [], f"backend imports ML frameworks directly: {offenders}"

    def test_backend_uses_engine_entrypoint(self):
        """Section 7: the backend calls ONE high-level engine, not stages."""
        text = (REPO_ROOT / "backend" / "app" / "services" / "inference_service.py").read_text()
        assert "SonarInferenceEngine" in text  # via mlpipeline.inference


class TestNoHardcodedClasses:
    def test_no_example_class_names_in_application_logic(self):
        """Detector class names must come from model metadata/config, never be
        hardcoded (dataset determines classes — biggest project risk rule)."""
        offenders = []
        for root in ("ml/mlpipeline", "backend/app", "frontend/src"):
            for p in (REPO_ROOT / root).rglob(("*" if "src" in root else "*.py")):
                if p.is_dir() or "node_modules" in p.parts or "dist" in p.parts:
                    continue
                text = p.read_text(encoding="utf-8", errors="replace").lower()
                for name in HARDCODED_CLASS_NAMES:
                    if re.search(rf"[\"']{name}[\"']", text):
                        offenders.append(f"{p}: '{name}'")
        assert offenders == [], f"hardcoded target classes: {offenders}"


class TestNoFabricatedGeo:
    def test_no_zero_zero_fabrication(self):
        """Section 10: never fabricate coordinates (no literal 0.0,0.0 defaults,
        no demo-coordinate constants in application code)."""
        offenders = []
        pattern = re.compile(r"(latitude|longitude|lat|lon)[\"']?\s*[:=]\s*(0\.0|0,|\(0)")
        for root in ("ml/mlpipeline", "backend/app"):
            for p in (REPO_ROOT / root).rglob("*.py"):
                text = p.read_text(encoding="utf-8", errors="replace")
                for m in pattern.finditer(text):
                    line = text[: m.start()].count("\n") + 1
                    offenders.append(f"{p.name}:{line}")
        assert offenders == [], f"possible fabricated coordinates: {offenders}"
