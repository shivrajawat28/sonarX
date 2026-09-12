#!/usr/bin/env python3
"""Verify the development environment for the Marine Debris Sonar AI project.

Checks: python version, required deps, optional ML deps (honest reporting),
disk space for DATA_ROOT, and presence of required config files.
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from pathlib import Path

REQUIRED = ["numpy", "cv2", "yaml", "pandas", "pydantic", "fastapi", "uvicorn", "multipart", "httpx", "jinja2", "pytest"]
OPTIONAL_ML = ["torch", "ultralytics", "onnxruntime"]  # needed only for real model training/inference


def _has(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def main() -> int:
    print("== Marine Debris Sonar AI — environment check ==")
    ok = True

    v = sys.version_info
    print(f"python: {v.major}.{v.minor}.{v.micro}", "OK" if v >= (3, 11) else "TOO OLD (need 3.11+)")
    ok &= v >= (3, 11)

    for m in REQUIRED:
        present = _has(m)
        print(f"dep {m:12s}: {'OK' if present else 'MISSING'}")
        ok &= present

    for m in OPTIONAL_ML:
        present = _has(m)
        print(f"optional {m:12s}: {'OK' if present else 'absent (stub detector path only; needed for real training/inference)'}")

    data_root = Path(os.environ.get("DATA_ROOT", "./data"))
    free_gb = shutil.disk_usage(data_root).free / 1e9 if data_root.exists() else -1
    print(f"DATA_ROOT: {data_root} free={free_gb:.1f}GB", "OK" if free_gb > 1 else "LOW/MISSING")

    for f in ["models/registry.json", "ml/configs/preprocessing/baseline_sonar.yaml", "ml/configs/filtering/rules.yaml"]:
        print(f"file {f}: {'OK' if Path(f).exists() else 'missing (created in later steps)'}")

    print("RESULT:", "OK" if ok else "PROBLEMS FOUND")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
