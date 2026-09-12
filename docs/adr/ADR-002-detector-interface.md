# ADR-002: Detector Interface + Canonical Detection

**Status:** Accepted (architecture v1.0, Sections 6.3 & 8)

## Decision

All model access goes through the `Detector` protocol (`ml/mlpipeline/detection/base.py`): `load()`, `predict(ProcessedSonarImage, PredictParams) -> list[RawDetection]`, `metadata() -> ModelMeta`. Concrete implementations (YOLO adapter, stub detector for tests/demo, optional ONNX) register in `ml/mlpipeline/detection/registry.py`. Class names come exclusively from model metadata `class_map` — never from application code.

## Rationale

- YOLO (or the final architecture, OPEN decision #5) can be replaced by adding one adapter file + registry entry; API and frontend are untouched.
- `RawDetection` (model-space boxes, bare class ids, raw scores) is deliberately different from the canonical `Detection` (source-space boxes, class names, both confidences) so post-processing has exactly one translation point.

## Consequences

- Application layers never import `ultralytics`/`torch` types.
- Per-request overrides are limited to thresholds (`PredictParams`), bounded by settings; architecture/weights are never per-request parameters.
