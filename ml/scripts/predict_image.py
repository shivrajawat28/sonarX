"""CLI: run the full pipeline on one image WITHOUT the API (Section 5 scripts).

Uses the stub detector by default — output is a PIPELINE TEST, clearly labeled,
never a real detection claim. For a real model: --kind yolo --model <version>
with weights registered in models/registry.json.

    python -m ml.scripts.predict_image --image path/to/sonar.png [--out result.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mlpipeline.config import load_config_file
from mlpipeline.config.schemas import PreprocessConfig
from mlpipeline.inference import SonarInferenceEngine
from mlpipeline.registry import get_registry


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="", help="write InferenceResult JSON here")
    ap.add_argument("--kind", default="stub", help="detector kind (stub|yolo|onnx)")
    ap.add_argument("--model", default=None, help="registry model version (default: stub-smoke-v1)")
    ap.add_argument("--preprocess", default="", help="preprocessing config path (default: bundled baseline)")
    ap.add_argument("--no-filter", action="store_true")
    args = ap.parse_args(argv)

    image = Path(args.image)
    if not image.is_file():
        print(f"ERROR: image not found: {image}", file=sys.stderr)
        return 2

    pp_path = args.preprocess or Path(__file__).resolve().parents[1] / "configs/preprocessing/baseline_sonar.yaml"
    pp_cfg, pp_hash = load_config_file(pp_path, PreprocessConfig)

    model_version = args.model or "stub-smoke-v1"
    if args.kind == "stub" and not model_version.startswith("stub"):
        print("ERROR: stub detector requires a 'stub-*' model version", file=sys.stderr)
        return 2

    engine = SonarInferenceEngine(
        detector_kind=args.kind,
        registry=get_registry(),
        preprocess_config=pp_cfg,
        preprocess_hash=pp_hash,
        run_filtering=not args.no_filter,
    )
    try:
        engine.load(model_version)
    except Exception as e:
        print(f"ERROR: model load failed ({e}).", file=sys.stderr)
        print("Note: stub needs no weights; real models must be registered in models/registry.json",
              file=sys.stderr)
        return 3

    result = engine.run_image_file(image)
    payload = result.model_dump()
    payload["notice"] = (
        "STUB/TEST PIPELINE OUTPUT — not a real detection claim"
        if args.kind == "stub" else "model output (see model_version)"
    )

    print(json.dumps({
        "image_id": payload["image_id"],
        "model_version": payload["model_version"],
        "preprocess_config_hash": payload["preprocess_config_hash"],
        "filter_config_hash": payload.get("filter_config_hash"),
        "n_detections": len(payload["detections"]),
        "statuses": [d["filtering_status"] for d in payload["detections"]],
        "timings_ms": payload["timings_ms"],
        "warnings": payload["warnings"],
        "notice": payload["notice"],
    }, indent=2))

    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"full result written: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
