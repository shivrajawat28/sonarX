#!/usr/bin/env python3
"""Multi-image real-inference audit (Phases 8-13 of the final audit).

Uploads several real DRISHTI-SSS test images through the API, runs REAL
model inference on each, then verifies:
  - detections come from the active model (provenance fields)
  - model vs final confidence stay distinct; filtering statuses valid
  - for tiles with ground truth: best accepted/flagged box IoU vs GT
    (verifies bbox coordinate scaling end-to-end)
  - JSON + CSV exports carry the full provenance contract
  - honest geo fields (lat/lon null + geo_status) for these tiles

Read-only with respect to datasets; writes only runtime records + tmp files.
"""
from __future__ import annotations

import csv
import io
import json
import random
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://localhost:8000/api/v1"
TEST_IMGS = ROOT / "datasets/processed/drishti-sss/test/images"
TEST_LBL = ROOT / "datasets/processed/drishti-sss/test/labels"
ACTIVE = "drishti-ss_yolov8n_e30_final"
N_IMAGES = 6


def api(path: str, method: str = "GET", body: dict | None = None,
        raw: bytes | None = None, content_type: str | None = None):
    url = f"{BASE}{path}"
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    if raw is not None:
        data = raw
        headers["Content-Type"] = content_type or "application/octet-stream"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=120) as r:
        payload = r.read()
        try:
            return r.status, json.loads(payload)
        except Exception:
            return r.status, payload


def upload_image(path: Path) -> str:
    boundary = "----auditboundary1234"
    body = (
        f"--{boundary}\r\n"
        f"Content-Disposition: form-data; name=\"file\"; filename=\"{path.name}\"\r\n"
        f"Content-Type: image/jpeg\r\n\r\n"
    ).encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    _, resp = api("/uploads/image", "POST", raw=body,
                  content_type=f"multipart/form-data; boundary={boundary}")
    return resp["image_id"]


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def main() -> int:
    random.seed(7)
    # Pick a deterministic mix: some with GT boxes, some background tiles.
    candidates = sorted(TEST_IMGS.glob("*.jpg")) + sorted(TEST_IMGS.glob("*.png"))
    with_gt = [f for f in candidates if (TEST_LBL / (f.stem + ".txt")).is_file()
               and f.stem.startswith(("wreckA", "wreckR", "pipe_", "mine_"))]
    bg = [f for f in candidates if f.stem.startswith("bg_")]
    picks = random.sample(with_gt, 4) + random.sample(bg, 2)

    failures: list[str] = []
    # Model-quality observations are NOT pipeline failures: this audit checks that
    # the pipeline is wired correctly and really calls the trained model. A low-recall
    # class (shipwreck: test recall 0.398) legitimately misses hard tiles, so misses
    # are reported and only an across-the-board zero is treated as broken.
    observations: list[str] = []
    gt_images = 0
    gt_images_with_detections = 0
    print(f"active model expected: {ACTIVE}")
    print(f"picked {len(picks)} images\n")

    for img in picks:
        name = img.name
        image_id = upload_image(img)
        status, det = api("/detections/run", "POST",
                          body={"image_id": image_id, "save": True})
        if status != 201:
            failures.append(f"{name}: inference HTTP {status}")
            continue
        run = det.get("run", det)
        model_version = det.get("model_version") or run.get("model_version")
        detections = det.get("detections") or []
        print(f"== {name}  ({model_version}, {len(detections)} detections)")

        if model_version != ACTIVE:
            failures.append(f"{name}: served model {model_version} != {ACTIVE}")

        # GT check (bbox scaling + class)
        lbl = TEST_LBL / (img.stem + ".txt")
        gts = []
        if lbl.is_file():
            for line in lbl.read_text().splitlines():
                p = line.split()
                if len(p) >= 5:
                    cx, cy, w, h = map(float, p[1:5])
                    gts.append((cx * 640 - w * 320, cy * 640 - h * 320, w * 640, h * 640, int(p[0])))

        pred_boxes = []
        for d in detections:
            mc, fc = d.get("model_confidence"), d.get("final_confidence")
            st = d.get("filtering_status")
            if st not in ("accepted", "flagged", "rejected"):
                failures.append(f"{name}: invalid filtering_status {st!r}")
            if mc is None or fc is None:
                failures.append(f"{name}: missing confidence fields")
            elif abs(mc - fc) < 1e-12 and d.get("filter_reasons"):
                failures.append(f"{name}: final==model confidence despite reasons")
            if d.get("latitude") is not None or d.get("longitude") is not None:
                failures.append(f"{name}: fabricated coordinates on a nav-less tile")
            if d.get("geo_status") != "unavailable":
                failures.append(f"{name}: unexpected geo_status {d.get('geo_status')!r}")
            b = d.get("bbox_source_coords") or {}
            if st in ("accepted", "flagged"):
                pred_boxes.append((b.get("x", 0), b.get("y", 0), b.get("w", 0), b.get("h", 0),
                                   d.get("class_name")))

        if gts:
            names = {0: "submarine_pipeline", 1: "shipwreck", 2: "ghost_net", 3: "mine_cylinder"}
            best = 0.0
            for g in gts:
                for p in pred_boxes:
                    if p[4] == names[g[4]]:
                        best = max(best, iou(g[:4], p[:4]))
            print(f"   GT instances: {len(gts)} | best IoU (kept boxes, same class): {best:.3f}")
            if gts and best == 0.0 and pred_boxes:
                failures.append(f"{name}: kept boxes never overlap GT (scaling/coord bug?)")
        else:
            print("   background tile (no GT)")
        if gts:
            gt_images += 1
            if detections:
                gt_images_with_detections += 1
            else:
                observations.append(
                    f"{name}: no detections above conf threshold ({len(gts)} GT instance(s)) "
                    f"— a recall miss, see docs/EVALUATION.md per-class recall"
                )

    # ---- export audit (Phases 12) ----
    _, resp = api("/exports/detections.json")
    dets = resp.get("detections", [])
    required = {"detection_id", "run_id", "image_id", "class_name", "model_confidence",
                "final_confidence", "filtering_status", "filter_reasons",
                "bbox_source_coords", "latitude", "longitude", "geo_status",
                "model_version", "preprocess_config_hash"}
    missing = required - set(dets[0].keys()) if dets else set()
    if missing:
        failures.append(f"JSON export missing fields: {sorted(missing)}")
    e30 = [d for d in dets if d.get("model_version") == ACTIVE]
    print(f"\nJSON export: {len(dets)} detections, {len(e30)} from {ACTIVE}")
    if not e30:
        failures.append("no e30_final detections in export")

    req = urllib.request.Request(f"{BASE}/exports/detections.csv")
    with urllib.request.urlopen(req, timeout=30) as r:
        csv_text = r.read().decode()
    rows = list(csv.DictReader(io.StringIO(csv_text)))
    print(f"CSV export: {len(rows)} rows, columns: {len(rows[0]) if rows else 0}")
    if rows:
        need_cols = {"class_name", "model_confidence", "final_confidence",
                     "filtering_status", "filter_reasons", "geo_status",
                     "model_version", "preprocess_config_hash"}
        miss = need_cols - set(rows[0].keys())
        if miss:
            failures.append(f"CSV export missing columns: {sorted(miss)}")
        if not any(r["model_version"] == ACTIVE for r in rows):
            failures.append("no e30_final rows in CSV export")

    # ---- report audit (Phase 13) ----
    runs = json.loads((ROOT / "data/db/detection_runs.json").read_text())
    e30_runs = [v for v in runs.values() if v.get("model_version") == ACTIVE]
    if e30_runs:
        rid = e30_runs[-1]["run_id"]
        _, j = api("/reports", "POST", body={"run_id": rid})
        job_id = j["job_id"]
        final = None
        for _ in range(30):
            import time
            time.sleep(1)
            _, job = api(f"/jobs/{job_id}")
            if job["status"] in ("succeeded", "failed"):
                final = job
                break
        if not final or final["status"] != "succeeded":
            failures.append(f"report job did not succeed: {final and final['status']}")
        else:
            rpt_id = final["result_ref"]["report_id"]
            req = urllib.request.Request(f"{BASE}/reports/{rpt_id}?format=pdf")
            with urllib.request.urlopen(req, timeout=30) as r:
                pdf = r.read()
            if pdf[:5] != b"%PDF-":
                failures.append("report PDF download is not a PDF")
            else:
                print(f"\nReport: {rpt_id} | PDF {len(pdf)} bytes (valid header)")
            req = urllib.request.Request(f"{BASE}/reports/{rpt_id}")
            with urllib.request.urlopen(req, timeout=30) as r:
                html = r.read()
            if b"<html" not in html[:200].lower():
                failures.append("report HTML download is not HTML")

    # Pipeline-level check: the model must actually fire on this sample at all.
    if gt_images and gt_images_with_detections == 0:
        failures.append(
            f"model produced ZERO detections across all {gt_images} GT-bearing images "
            f"— inference pipeline is probably broken"
        )

    if observations:
        print(f"\nObservations (not failures) — {len(observations)}:")
        for o in observations:
            print("  -", o)
        print(
            f"  ({gt_images_with_detections}/{gt_images} GT-bearing images produced detections)"
        )

    print("\n" + "=" * 60)
    if failures:
        print(f"AUDIT FAILURES ({len(failures)}):")
        for f in failures:
            print("  -", f)
        return 1
    print("AUDIT PASSED: provenance, filtering, bbox scaling, exports, report all verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
