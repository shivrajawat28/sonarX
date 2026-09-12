"""Evaluation runner (Section 6.2/6.5): model x dataset x split -> EvaluationRun.

Reproducibility: uses the model's registered preprocessing config (by hash),
the dataset manifest (by hash), and records split seed + filter ON/OFF runs.
Gate for Step 7/8: smoke train -> eval -> register works end-to-end.
"""
from __future__ import annotations

import json
from pathlib import Path

from mlpipeline.config.loader import load_config_file
from mlpipeline.config.schemas import DetectionConfig, FilterConfig, PreprocessConfig
from mlpipeline.datasets.manifest import DatasetManifest, load_manifest
from mlpipeline.datatypes.detection import BBox
from mlpipeline.datatypes.model import EvaluationRun, MetricsSummary, PerClassMetrics
from mlpipeline.evaluation.metrics import (
    EvalCounts,
    average_precision_50,
    confusion_matrix,
    match_detections,
    precision_recall_f1,
)
from mlpipeline.registry.eval_runs import EvalRunStore, get_eval_store
from mlpipeline.registry.models import ModelRegistry, get_registry, resolve_repo_path


def load_ground_truth(manifest: DatasetManifest, root: Path, split: str) -> dict[str, list[tuple[str, BBox]]]:
    """dataset manifest -> image path -> [(class_name, BBox), ...] in source pixel space.

    YOLO labels are normalized cx,cy,w,h; images are read only for width/height.
    """
    import cv2

    out: dict[str, list[tuple[str, BBox]]] = {}
    # Both dataset layouts: images/{split}/... (flat-first) or {split}/images/...
    # (ultralytics data.yaml convention produced by scripts/prepare_drishti.py)
    prefixes = (f"images/{split}/", f"{split}/images/")
    for entry in manifest.images:
        rel = entry.image.path
        if not rel.startswith(prefixes):
            continue
        img = cv2.imread(str(root / rel), cv2.IMREAD_UNCHANGED)
        h, w = (img.shape[0], img.shape[1]) if img is not None else (1, 1)
        boxes: list[tuple[str, BBox]] = []
        if entry.label is not None:
            lbl = root / entry.label.path
            if lbl.is_file():
                for line in lbl.read_text().splitlines():
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    cid = int(parts[0])
                    cx, cy, bw, bh = (float(x) for x in parts[1:5])
                    cname = (
                        manifest.class_names[cid]
                        if 0 <= cid < len(manifest.class_names)
                        else str(cid)
                    )
                    boxes.append(
                        (cname, BBox(x=(cx - bw / 2) * w, y=(cy - bh / 2) * h, w=bw * w, h=bh * h))
                    )
        out[rel] = boxes
    return out


def evaluate_model(
    model_version: str,
    dataset_manifest_path: str | Path,
    split: str = "test",
    filter_config_path: str | Path | None = None,
    eval_run_id: str | None = None,
    registry: ModelRegistry | None = None,
    store: EvalRunStore | None = None,
    notes: str = "",
) -> EvaluationRun:
    """Evaluate a registered model against a dataset manifest split.

    Records metrics with filter OFF (detector-only) in this record; a companion
    record with filter ON is produced by passing the filter config (recorded
    separately via `filter_enabled` flag per ADR-010).
    """
    import cv2

    from mlpipeline.inference.engine import SonarInferenceEngine

    registry = registry or get_registry()
    store = store or get_eval_store()
    entry = registry.get(model_version)
    if entry is None:
        raise ValueError(f"unknown model version '{model_version}'")

    manifest = load_manifest(Path(dataset_manifest_path))
    ds_root = Path(manifest.root)
    gt = load_ground_truth(manifest, ds_root, split)

    # preprocessing config BY HASH from the model's record (train/serve parity)
    pp_path = resolve_repo_path(entry.preprocess_config_ref.path)
    pp_cfg, pp_hash = load_config_file(pp_path, PreprocessConfig)
    if pp_hash != entry.preprocess_config_ref.sha256:
        raise ValueError(
            f"preprocessing config drift for {model_version}: "
            f"registered {entry.preprocess_config_ref.sha256[:12]} but file hashes {pp_hash[:12]}"
        )

    engine = SonarInferenceEngine(
        detector_kind=entry.architecture_family,
        registry=registry,
        preprocess_config=pp_cfg,
        preprocess_hash=pp_hash,
        run_filtering=False,  # detector-only metrics
    )
    engine.load(model_version)

    filter_cfg_hash = None
    if filter_config_path is not None:
        fcfg, filter_cfg_hash = load_config_file(filter_config_path, FilterConfig)

    all_gt: list[tuple[str, BBox]] = []
    all_preds: list[tuple[str, BBox, float]] = []
    per_image_counts: dict[str, EvalCounts] = {}
    class_names = list(manifest.class_names) or sorted(
        {c for boxes in gt.values() for c, _ in boxes}
    )

    for rel, boxes in sorted(gt.items()):
        img = cv2.imread(str(ds_root / rel), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        result = engine.run_array(img, image_id=rel)
        counts = match_detections(
            boxes, [(d.class_name, d.bbox_source_coords, d.model_confidence) for d in result.detections]
        )
        per_image_counts[rel] = counts
        all_gt.extend(boxes)
        all_preds.extend((d.class_name, d.bbox_source_coords, d.model_confidence) for d in result.detections)

    totals = EvalCounts(
        tp=sum(c.tp for c in per_image_counts.values()),
        fp=sum(c.fp for c in per_image_counts.values()),
        fn=sum(c.fn for c in per_image_counts.values()),
    )
    # Aggregate per-image match records — per-class metrics and the confusion
    # matrix are computed from these (bugfix: previously totals.matches was
    # left empty, zeroing every per-class P/R/F1 while AP50 stayed correct).
    totals.matches = [m for c in per_image_counts.values() for m in c.matches]
    p, r, f1 = precision_recall_f1(totals)

    per_class: dict[str, PerClassMetrics] = {}
    for cname in class_names:
        tp_c = 0
        fp_c = 0
        for m in totals.matches:
            if m.pred_class == cname and m.iou >= 0.5 and m.gt_class == cname:
                tp_c += 1
            elif m.pred_class == cname and m.iou < 0.5:
                fp_c += 1
        n_gt_c = sum(1 for c, _ in all_gt if c == cname)
        p_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) else 0.0
        r_c = tp_c / (tp_c + (n_gt_c - tp_c)) if n_gt_c else 0.0
        f1_c = 2 * p_c * r_c / (p_c + r_c) if (p_c + r_c) else 0.0
        ap50 = average_precision_50(all_gt, all_preds, cname)
        per_class[cname] = PerClassMetrics(
            ap50=None if ap50 != ap50 else ap50,  # NaN (no GT) -> None
            precision=p_c, recall=r_c, f1=f1_c, support=n_gt_c,
        )

    # Macro-average mAP50 across classes that actually have ground truth.
    # (mAP50-95 is NOT computed by this pipeline — it stays None here; the
    # ultralytics val summaries record it separately. Never estimated.)
    ap50_values = [pm.ap50 for pm in per_class.values() if pm.ap50 is not None and pm.support > 0]
    map50_macro = sum(ap50_values) / len(ap50_values) if ap50_values else None

    run = EvaluationRun(
        eval_run_id=eval_run_id or run_id_for(model_version, split),
        model_version=model_version,
        dataset_ref={
            "path": str(dataset_manifest_path),
            "sha256": manifest_sha256(dataset_manifest_path),
        },
        split=split,  # type: ignore[arg-type]
        split_seed=manifest.split_seed,
        preprocess_config_hash=pp_hash,
        filter_config_hash=filter_cfg_hash,
        filter_enabled=False,
        metrics=MetricsSummary(precision=p, recall=r, f1=f1, mAP50=map50_macro),
        per_class=per_class,
        hyperparameters={"iou_match": 0.5},
        notes=notes,
    )
    # Serialize as real JSON. (`.__str__()` produced a Python repr with single
    # quotes, so the artifact named confusion.json could not be parsed by any
    # consumer — the UI included. Values are unchanged; only the encoding is.)
    artifacts = {
        "confusion.json": json.dumps(confusion_matrix(totals.matches, class_names), indent=2),
    }
    store.save(run, artifacts=artifacts)
    return run


def run_id_for(model_version: str, split: str) -> str:
    import datetime as _dt

    stamp = _dt.datetime.now(_dt.UTC).strftime("%Y%m%d_%H%M%S")
    return f"eval_{model_version}_{split}_{stamp}"


def manifest_sha256(path: str | Path) -> str:
    from mlpipeline.config.loader import sha256_of_file

    return sha256_of_file(Path(path))
