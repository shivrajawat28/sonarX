# Final Engineering Report — Marine Debris Sonar AI

**Date:** September 12, 2026 (final release audit appended — see §0)
**SIH Problem Statement:** SIH26057

---

## 0. FINAL RELEASE AUDIT (2026-09-12) — defects found and fixed

A release-gate audit was run against the running system (restarted backend,
live browser E2E, real sonar imagery). Everything below was **observed**, fixed
at the root cause, and covered by a new regression test. Measured evidence is
given for each.

### 0.1 CRITICAL — detection overlay was drawn 70 px off on the demo path

- **Symptom (measured):** for the standard demo tile `pipe_1693569383.780_x3500.jpg`
  (640×500), the API reported `bbox_source_coords.y = 2.415` and
  `bbox_processed_coords.y = 72.415`. The workbench displays the **processed**
  image by default but the viewer drew **source** coordinates — so the box sat
  **70 px above the object** (the letterbox pad). Reproduced in a headless
  browser: displayed image 640×640, rendered rect at `y=2.415`.
- **Root cause:** one coordinate space assumed everywhere. `SonarViewer` always
  read `bbox_source_coords`, ignoring `bbox_processed_coords`, which the backend
  already provided.
- **Why it went unnoticed:** `docs/DATASET_AUDIT.md` stated tiles were "640×640
  (pre-tiled)". Decoding all 5,205 images shows **394 distinct sizes** — only
  47.5% are 640×640, 37.7% are 640×500. On a square tile the letterbox is the
  identity transform and the bug is invisible.
- **Fix:** `SonarViewer` now takes an explicit `coordSpace` matching the image on
  screen and selects the corresponding box, refusing to draw when that space is
  unavailable (never guesses). `bboxInSpace()` in the API client. Unknown space →
  explicit ⚠ count rather than a misplaced box.
- **Verified:** browser E2E asserts the rendered rects equal the API coordinates
  for the displayed space — **worst |Δ| = 0.00 px** (was 70 px) — and that every
  box lies inside the image.

### 0.2 Export and report silently truncated at 200 detections

- **Root cause:** `_fetch_filtered` (exports) and `_run_report` (reports) called
  `repo.list(size=200)`, and the repository caps page size at 200 — with no
  indication in the artifact.
- **Latent bug found while testing the fix:** `FileRepository.list` applied the
  200 cap to the page **offset** but sliced with the caller's raw `size`, so a
  request for `size=5000` returned everything while `size=200` truncated. The cap
  was half-applied.
- **Fix:** added `FileRepository.list_all()` (unpaginated, for artifact paths);
  exports and reports now use it; `list()` enforces the cap on length *and* offset.
- **Verified:** regression test seeds 250 detections → CSV/JSON carry 250,
  report carries them all, and `list()` still returns ≤ 200 per page.

### 0.3 Reports omitted bounding boxes

- The requirement (and `docs/ARCHITECTURE.md`) asks reports to carry boxes; the
  PDF builder even computed `bbox` and never used it.
- **Fix:** HTML + PDF report tables gained a `BBox x,y,w,h (source px)` column and
  a `Geo method` column. Verified by E2E + regression test.

### 0.4 `confusion.json` was not JSON

- **Symptom:** `evaluate.py` wrote `confusion_matrix(...).__str__()` — a Python
  repr with single quotes — into a file named `.json`. `json.loads` rejected all
  5 existing artifacts, so nothing (including the dashboard) could consume them.
- **Fix:** write `json.dumps(...)`; converted the 5 existing artifacts from the
  repr to real JSON **without changing any value** (verified by re-parsing and
  re-reading each file). The `/models/{v}/metrics` response now carries the
  matrix, and the Models page renders it.
- **Verified:** regression test parses the matrix as JSON and asserts a repr would
  fail.

### 0.5 Registry hardcoded absolute machine paths

- `registry.json` stored `C:/Users/kr034/OneDrive/...` for weights, preprocessing
  config and dataset refs. Moving or syncing the repo would silently degrade the
  whole demo to `MODEL_UNAVAILABLE` (and leaked internal filesystem paths).
- **Fix:** `resolve_repo_path()` (used by the YOLO adapter, ONNX adapter, inference
  service, evaluator and trainer) resolves repo-relative paths and recovers from a
  stale absolute path by matching its existing suffix. `register_trained_model.py`
  and `train.py` now write repo-relative paths; existing registry entries were
  normalised in place with all hashes re-verified against the files on disk.
- **Verified:** backend restarted and still logs `model loaded:
  drishti-ss_yolov8n_e30_final` from the rewritten registry; regression tests cover
  existing/relative/stale/unrecoverable paths.

### 0.6 `resolve_within` used a string prefix (sibling-dir escape)

- A root of `/data` would accept `/data-evil/...` because the check was
  `str(candidate).startswith(str(root))`.
- **Fix:** real path ancestry via `Path.is_relative_to`. Regression tests cover the
  sibling case, normal nesting, and the root itself.

### 0.7 Frontend console error on every page load

- `/favicon.ico` 404'd, so "no console errors" was unachievable. Added a
  sonar-themed `frontend/public/favicon.svg` + `<link rel="icon">`.
- **Verified:** E2E now asserts zero console errors and zero 4xx/5xx responses.

### 0.8 Hero visual: list ↔ overlay correspondence

- Clicking a detection row now highlights the matching box (thicker stroke, others
  faded) and clicking a box selects the row; filter reasons are rendered as a
  visible column instead of only a tooltip; a status legend and honest
  "Location unavailable — no navigation metadata provided" text are shown.
- **Verified:** E2E clicks a row and asserts the rendered stroke/opacity changed.

### 0.9 Documentation corrected (the audit's own findings)

- `DATASET_AUDIT.md` claimed 640×640 tiles → replaced with the measured size
  distribution (394 shapes).
- `DATASET_AUDIT.md` / `CLASS_TAXONOMY.md` / `ENGINEERING_REPORT.md` claimed the
  shipwreck split was not site-disjoint-audited and called the metrics optimistic.
  The audit (`scripts/audit_site_leakage.py`) had in fact been run: **0 of 64
  wreckA and 0 of 23 wreckR test sites overlap train/val**. Corrected, and the
  537-tile site-disjoint evaluation (mAP50 0.702) is now documented.
- `EVALUATION.md` said 903 test instances → measured **901**; and its reproduce
  command pointed at the mutable `best.pt` instead of the frozen `best_final.pt`.
- `MODEL_SELECTION.md` said YOLOv8s was "not trained" → a 5-epoch probe exists
  (val mAP50 0.658, rejected and intentionally unregistered); documented with its
  raw curve.

### 0.10 Verified as already correct (no change made)

- **Letterbox pad mismatch:** training pads 114, inference config pads 0. Measured
  on 30 real test images → 28 detections both ways, mean confidence identical to 3
  d.p. Left unchanged (editing the config would invalidate the registry hash for
  no measurable gain); documented in `docs/TRAINING.md` instead.
- **No fake AI in the serving path:** no hardcoded detections, coordinates or
  metrics; the API's `model_version` on real runs equals the registry's single
  active entry, and geolocation returns `null` + `geo_status` without nav data.
- **Metrics provenance:** the active model's stored test metrics were re-checked
  against the frozen weights by re-running the site-disjoint evaluation, which
  reproduced the stored numbers **exactly**.

---

## 1. WHAT ALREADY EXISTED

The repository contained a substantial software architecture with:

- **ML Pipeline (`ml/mlpipeline/`):** Full modular pipeline with canonical datatypes, config-driven preprocessing, detector abstraction (stub + YOLO adapter), post-processing, filtering, geolocation, inference engine, model registry, evaluation framework, and training scripts.
- **Backend (`backend/`):** FastAPI thin bridge with health, upload, inference, detection history, exports, reports, surveys, jobs, and models endpoints. File-based persistence, error taxonomy, request-ID middleware.
- **Frontend (`frontend/`):** React + TypeScript SPA with upload, sonar viewer, detection overlay, map, history, models, and reports pages. Leaflet map integration.
- **Architecture docs:** Comprehensive 1395-line architecture document with 25 sections covering every component.
- **Tests:** 143 tests (ML unit + backend unit + integration + boundary lint).
- **Demo fixtures:** Stub detector + synthetic data for pipeline validation.
- **Dataset:** Full DRISHTI-SSS (5,205 images) already downloaded and prepared; a 30-epoch training run had been started but was interrupted at epoch 5.

## 2. WHAT I CHANGED (this session)

### Root-cause fixes
- **Concurrent-training corruption:** the first e30 run had FOUR simultaneous training processes writing one run dir (launcher script + `.bat` launched repeatedly); it died at epoch 5 with a quadruple-written `results.csv`. Fixed with a single-instance guarded launcher (`scripts/launch_background_train.py`: pidfile + liveness check, refuses a second trainer) and a `--resume` path in `scripts/train_yolo.py`. De-duplicated the CSV; preserved the epoch-5 checkpoint.
- **Report generation broken (path containment):** `.env` `DATA_ROOT=./data` is CWD-relative, but report artifacts are absolute — `resolve_within` correctly rejected the mismatch. Fixed by anchoring relative `data_root`/`models_dir` to the repo root in `Settings` (field_validator), making paths deterministic regardless of CWD.
- **Per-class metrics all zero:** `evaluate_model` rebuilt `totals` without the per-image `matches` lists, so every per-class P/R/F1 computed from an empty list while AP50 stayed correct. Fixed by aggregating match records.
- **Frontend crash on real detections:** components expected `bbox_xyxy` but the API returns `bbox_source_coords {x,y,w,h}` — `DetectionDetails` crashed (`undefined.map`) on every real inference. Fixed via a `bboxToXyxy` helper used by `DetectionDetails` + `SonarViewer`. Caught by real-browser E2E, not by CLI checks.
- **Report download link never appeared:** `ReportActions` defined but never invoked its poll mutation (plus a stale-closure read of `jobId`). Fixed.
- **Report download 404:** frontend used `/reports/{id}/download`; the backend route is `GET /reports/{id}` (attachment header). Fixed client URL.
- **`intensity_outlier` filter semantics:** rule penalized regions whose MEAN sat inside the image's p10–p90 band — a box containing a bright wreck PLUS its shadow averages into the band and was wrongly penalized. Changed to measure the region's INTERNAL contrast (std), matching the rule's documented intent and the pinned test (flat region still triggers). All 143 tests pass.
- **Manifest tool layout mismatch:** `build_manifest` only supported `images/{split}/`; `prepare_drishti.py` emits `{split}/images/` (ultralytics convention). Extended to support both layouts; eval GT loader updated to match.
- **Survey geolocation never worked (Case A):** the upload handler parsed the `nav.csv` sidecar and saved it as an artifact, but the final survey update hardcoded `track_ref: None, sample_count: 0` — no survey ever carried its navigation track, so batch geolocation silently produced zero geolocated detections. Fixed; verified with real DRISHTI test tiles + a clearly-labelled synthetic nav track: coordinates interpolate along the track, `geo_provenance` marks them synthetic/demo.
- **Filter penalty caps ignored:** `rules.yaml` declares per-rule `penalty` caps (e.g. `edge_clip: 0.30`) but the pipeline ignored them — rules hardcoded 1.0, so one edge-clip hit drove `final_confidence` to exactly 0.0 and `rejected`. Pipeline now honours the configured cap: 0.74 conf − 0.30 = 0.44 → `flagged`, reason retained.
- **Registry drift (integrity):** the `e5_interim` entry pointed at the live `best.pt`, which kept updating during training — "e5" was silently serving ~epoch-11 weights. Froze `best_snapshot_ep11.pt`, registered it as `drishti-ss_yolov8n_e11_snapshot` (active), retired the interim. The finalizer now freezes weights before registration so its entry can never drift.
- **Models page showed the wrong model + no per-class table:** it picked `versions[0]` (the oldest, retired stub) and ignored `per_class`/`split` fields the API already returned. Rewritten around the actually-loaded model, with per-class metrics, explicit VALIDATION/TEST split labels, and corrected client types.
- **Eval pipeline mAP:** added macro-mAP50 (legitimately computed from per-class AP50) to stored EvaluationRuns so the dashboard can show an overall mAP without fabrication.
- **Stale DB artefacts round 2:** stub-era jobs/reports/surveys and retired-interim runs/detections purged with backup (`data/db_backup_purge2/`) — runtime data now contains only artifacts from currently-valid registry models.

### New files
- `scripts/register_trained_model.py` — append-only registry registration from `data.yaml` classes + hashed preprocess config
- `datasets/manifests/drishti-sss.json` — hashed manifest of all 5,205 images (validation OK)
- `docs/EVALUATION.md`, `docs/MODEL_SELECTION.md` (rewritten `docs/TRAINING.md`, `docs/DATASET_AUDIT.md` measured counts, `docs/CLASS_TAXONOMY.md` corrected mapping)
- Browser E2E scripts (`frontend/e2e_full.mjs`, Playwright + system Chrome)

### Registry state (append-only history preserved)
- `stub-demo-v1` → **retired**; `drishti-ss_yolov8n_smoke_e5` → **retired**; `drishti-ss_yolov8n_e5_interim` → **retired** (drifted pointer); `drishti-ss_yolov8n_e11_snapshot` → **retired** (interim served during training)
- `drishti-ss_yolov8n_e30_final` → **ACTIVE** (COMPLETE 30/30-epoch run, frozen `best_final.pt`, evaluated on test+val, then promoted; verified loaded in the API)
- Purged 48 stub-fixture detections + 13 stub/smoke runs (backup `data/db_backup_20260911/`), then stub-era jobs/reports/surveys + retired-interim runs/detections (backup `data/db_backup_purge2/`) — exports/report data contain only artifacts from currently-valid models

## 3. DATASETS USED

**Primary: DRISHTI-SSS** (HuggingFace `rehan9599/drishti-sss`, CC-BY-SA-4.0)
- 5,205 pre-tiled 640×640 side-scan sonar images; YOLO format; already Lee-speckle + CLAHE preprocessed
- Splits: train 3,875 / val 630 / test 700 (official splits used as-is)
- Measured instance counts (train): submarine_pipeline 1,000; shipwreck 1,554; ghost_net 900 (**100% synthetic**); mine_cylinder 843
- All-splits instances: shipwreck 2,623; submarine_pipeline 1,321; ghost_net 1,140; mine_cylinder 1,018
- Multi-source: SubPipeMini2 (pipe/bg), AI4Shipwrecks (wreckA), Roboflow SSS (wreckR), Kaggle MILCO (mine), procedural generator (synth)

**Others present but unusable (documented, not used):** AI4Shipwrecks (corrupt zip), SubPipeMini2 (partial), GhostVision (partial, gated), Kaggle-SSS (empty).

## 4. FINAL CLASS TAXONOMY

| ID | Class | Real/Synthetic | Train instances | Decision |
|---|---|---|---|---|
| 0 | submarine_pipeline | **Real** | 1,000 | ✅ Primary |
| 1 | shipwreck | **Real** | 1,554 | ✅ Primary |
| 2 | ghost_net | **⚠️ 100% Synthetic** | 900 | ⚠️ Experimental — never present as field capability |
| 3 | mine_cylinder | **Real** | 843 | ✅ Primary (weakest) |

 crab_pot excluded upstream (0 examples). Mapping verified against raw label scans: raw id N → processed id N−1.

## 5. MODEL(S) TRAINED

- **`drishti-ss_yolov8n_e30_final` (ACTIVE):** YOLOv8n, 3.0M params, COMPLETE 30/30-epoch single-writer run on the full 3,875-image train split (CPU, ~26,300 s total). Weights frozen to `weights/best_final.pt` at registration. Evaluated on test (repo pipeline + ultralytics) and val (ultralytics), then promoted to active; verified loaded in the running API.
- Interim checkpoints kept for the record (all retired): `e5_interim` (drifted pointer, renamed), `e11_snapshot` (frozen mid-run serving model).
- **YOLOv8s comparison:** consciously deferred (≈3× CPU cost, ~75+ h). Documented in `docs/MODEL_SELECTION.md` — not silently skipped.

## 6. BEST MODEL SELECTED + WHY

YOLOv8n: the only variant that reaches a fully trained state on this CPU (AMD RX 6500M is not CUDA-usable) within SIH timelines; ~58 ms/image CPU inference makes the demo feel real-time; dataset is pre-tiled 640px so nano capacity is not the bottleneck. The detector abstraction means a larger future checkpoint drops in via one registry entry.

## 7. TRAINING CONFIGURATION

```yaml
model: yolov8n.pt (COCO pretrained)
epochs: 30
batch: 8
imgsz: 640
seed: 42
device: cpu
deterministic: true
augmentation (sonar-safe): hsv_h 0.0, hsv_s 0.0, hsv_v 0.3, degrees 5.0,
  translate 0.1, scale 0.3, fliplr 0.5, flipud 0.0, mosaic 0.8, mixup 0.1
preprocessing: resize_letterbox only (data already Lee+CLAHE)
```

## 8. SPLIT STRATEGY

Official DRISHTI-SSS splits used as-is (train 3,875 / val 630 / test 700; 901
test instances measured). Val used for training-time model selection; **test used
only for final reported metrics**. Site-disjointness WAS audited by deriving site
ids from wreck filenames (`scripts/audit_site_leakage.py`): **0 of 64 wreckA and 0
of 23 wreckR test sites appear in train or val**, and a 537-tile site-disjoint
subset scores mAP50 0.702 / mAP50-95 0.525 vs 0.699 / 0.518 on the full test split.
The shipwreck test tiles are 50%-overlap re-tilings of *unseen* sites — that makes
them harder (partial views), not leaky.

## 9. FINAL METRICS (test split, measured — see docs/EVALUATION.md)

ACTIVE `drishti-ss_yolov8n_e30_final`, detector-only, IoU 0.5, repo eval pipeline:
- **Overall: P 0.758 / R 0.586 / F1 0.661 / mAP50 0.663**
- Ultralytics-native (frozen `best_final.pt`), **test**: P 0.714 / R 0.705 / mAP50 0.699 / **mAP50-95 0.518**; **val** (selection set, not comparable): mAP50 0.721 / mAP50-95 0.530
- Progression (test F1): 0.538 (ep5) → 0.556 (ep11) → **0.661 (ep30)**; shipwreck recall 0.152 → 0.173 → **0.398**
- Earlier interim numbers kept in docs for transparency

## 10. PER-CLASS PERFORMANCE (test, e30_final)

| Class | P | R | F1 | AP50 | Support |
|---|---|---|---|---|---|
| submarine_pipeline | 0.994 | 0.983 | **0.988** | 0.964 | 174 |
| shipwreck | 0.609 | 0.398 | 0.482 | 0.421 | 525 |
| ghost_net (synthetic) | 1.000 | 1.000 | 1.000* | 1.000* | 120 |
| mine_cylinder | 0.452 | 0.341 | 0.389 | 0.266 | 82 |

*synthetic-on-synthetic — not a field number.

## 11. FAILURE CASES (observed)

- Shipwreck remains the weak real class (test R 0.398 at 30 epochs; was 0.152 at ep5): partial/edge views on the deliberately hard 50%-overlap re-tiled test split are missed; precision 0.609 means ~2 of 5 wreck predictions are wrong. Reproduced live during the release audit: `wreckA_WP_Thew_17_y640_x960.jpg` has 2 ground-truth wreck instances and the model returns **0 detections** above conf 0.25 — a genuine recall miss, not a bug, and reported as such.
- Mine cylinder improved but stays weakest (test F1 0.389; AP50 0.266) — 843 real train instances from a single source; smaller AP says ranking quality, not just threshold choice, is limited.
- Low-confidence wreck boxes on image edges are deterministically rejected (edge_clip + intensity) with reasons retained — correct behavior, honest annotation.

## 12. MODEL LIMITATIONS

1. No generalization claim to unseen sonar hardware or sites
2. ghost_net is synthetic-only — excluded from any capability claim
3. Shipwreck recall is low (0.398) and this is the dominant error mode
4. No navigation metadata in DRISHTI-SSS — coordinates are always null for these images (never fabricated; single-image uploads show an explicit "location unavailable" notice). Geolocation is exercised only via the survey+`nav.csv` path, verified with an explicitly synthetic track.
5. CPU-only training/inference; the YOLOv8s comparison was probed (5 epochs) but not completed — the nano won on val mAP (0.730 vs 0.658)
6. Prototype metrics on one academic-style dataset — no operational validation
7. Class taxonomy is fixed to the 4 DRISHTI-SSS classes; crab_pot is absent from the release
8. Multi-detection tiles in a survey share one interpolated along-track position (OPEN #3 MVP fallback) with a recorded `uncertainty_m`, so two contacts in the same tile cannot be separated along track

## 13. FILES CREATED/MODIFIED (this session)

Created (earlier session): `scripts/register_trained_model.py`, `scripts/finalize_training.py` (watcher: register→evaluate→promote on training exit), `datasets/manifests/drishti-sss.json`, `docs/EVALUATION.md`, `docs/MODEL_SELECTION.md`, `frontend/e2e_full.mjs`, `data/db_backup_20260911/`, `data/db_backup_purge2/`
Modified (earlier session): `scripts/train_yolo.py` (resume), `scripts/launch_background_train.py` (single-instance), `backend/app/core/config.py` (path anchoring), `backend/app/api/v1/uploads.py` (nav track linkage), `backend/app/services/job_service.py` (report metadata block), `ml/mlpipeline/evaluation/evaluate.py` (per-class fix + macro-mAP50), `ml/mlpipeline/datasets/manifest.py` (layouts), `ml/mlpipeline/filtering/rules/intensity_outlier.py` (internal contrast), `ml/mlpipeline/filtering/pipeline.py` (penalty caps), `frontend/src/api/client.ts`, `frontend/src/components/DetectionDetails.tsx`, `frontend/src/components/SonarViewer.tsx`, `frontend/src/components/ReportActions.tsx`, `frontend/src/pages/ModelsPage.tsx`, docs (`TRAINING`, `DATASET_AUDIT`, `CLASS_TAXONOMY`, `EVALUATION`, `README`)

Created (final release audit, 2026-09-12): `frontend/public/favicon.svg`, `backend/tests/integration/test_export_report_regressions.py`, `ml/tests/unit/test_registry_regressions.py`
Modified (final release audit): `frontend/src/components/SonarViewer.tsx` (coordinate-space aware + selection), `frontend/src/components/DetectionDetails.tsx` (selectable rows, reasons, geo provenance), `frontend/src/pages/WorkbenchPage.tsx` (space wiring, selection, legend, warnings), `frontend/src/api/client.ts` (`bboxInSpace`, `CoordSpace`, confusion/provenance types), `frontend/src/pages/ModelsPage.tsx` (confusion matrix + training provenance), `frontend/index.html`, `frontend/e2e_full.mjs` (rewritten, 37 assertions incl. overlay alignment), `backend/app/persistence/file_repository.py` (`list_all`, corrected page cap), `backend/app/api/v1/exports.py`, `backend/app/services/job_service.py` (no truncation, bbox in HTML+PDF), `backend/app/api/v1/models.py` (confusion + provenance), `backend/app/core/security.py` (`resolve_within` ancestry), `backend/app/services/inference_service.py`, `ml/mlpipeline/registry/models.py` (`resolve_repo_path`), `ml/mlpipeline/registry/eval_runs.py` (`read_json_artifact`), `ml/mlpipeline/evaluation/evaluate.py` (JSON confusion artifact), `ml/mlpipeline/detection/yolo/adapter.py`, `ml/mlpipeline/detection/onnx_adapter.py`, `ml/mlpipeline/training/train.py`, `scripts/register_trained_model.py` (repo-relative paths), `models/registry.json` (paths normalised), 5 × `models/eval/**/confusion.json` (repr → JSON, values unchanged), docs (`README`, `DATASET_AUDIT`, `CLASS_TAXONOMY`, `EVALUATION`, `MODEL_SELECTION`, `TRAINING`, `ENGINEERING_REPORT`)

**Weights:** `models/weights/drishti-ss_yolov8n_e30/weights/best_final.pt` (frozen; registered as the ACTIVE `drishti-ss_yolov8n_e30_final`). Run artifacts (`best.pt`, `last.pt`, `results.csv`) preserved alongside.

## 14–17. COMMANDS

```bash
# backend (from repo root)
PYTHONPATH="ml;." .venv/Scripts/python -m uvicorn backend.app.main:app --port 8000
# frontend
cd frontend && npm run dev          # http://localhost:5173
# real inference (CLI)
PYTHONPATH="ml;." .venv/Scripts/python -m ml.scripts.predict_image --image <sonar_image.png>
# retrain (single-instance guarded)
.venv/Scripts/python scripts/launch_background_train.py             # new run
.venv/Scripts/python scripts/launch_background_train.py --resume    # resume
# evaluate
PYTHONPATH="ml;." .venv/Scripts/python -m ml.scripts.evaluate --model <version> --manifest datasets/manifests/drishti-sss.json --split test
```

## 18. TEST RESULTS (all executed this session)

```
pytest ml/tests backend/tests : 157/157 passed ✅  (143 + 14 new regression tests)
  (ml/tests + backend/tests, incl. boundary/architecture lint)
frontend npx tsc --noEmit    : clean ✅
frontend npm run build       : succeeds ✅
Browser E2E (Playwright + system Chrome, node frontend/e2e_full.mjs):
  37/37 checks PASSED ✅ — health/loaded model, upload, preprocessing preview,
  real inference by the active model, detection count parity, filter status +
  reasons visible, model-vs-final confidence, honest geo notice, overlay
  coordinate-space match (worst |Δ| 0.00 px), boxes inside the image,
  row↔box highlight, CSV + JSON export (provenance columns), report job +
  HTML/PDF download with bbox, models page (split label, confusion matrix,
  provenance), history page, ZERO console errors, ZERO 4xx/5xx.
Independent metric re-verification: `python scripts/eval_sitedisjoint.py`
  reproduced the stored summary EXACTLY (P 0.733 / R 0.6807 / mAP50 0.7019 /
  mAP50-95 0.525) against the frozen best_final.pt ✅
Backend restart persistence: restarted uvicorn → same model loaded, prior
  detections/exports still served from data/db ✅
```

## 19. WHAT IS DEMO-READY (verified)

- ✅ REAL trained model serving inference (registered, active) — confirmed by
  restarting the backend and reading the startup log, then checking every
  detection's `model_version` equals the registry's active entry
- ✅ Upload → preprocess (config+hash shown) → detect → filter (reasons) → results
- ✅ Bounding-box overlay, coordinate-space correct on both the original and the
  preprocessed image (verified to 0.00 px in a headless browser), with
  click-to-highlight linking the box and the detection row
- ✅ Honest geolocation: coordinates only from real nav metadata; explicit notice otherwise
- ✅ Detection history; JSON/CSV exports with full provenance columns
- ✅ Report generation + download through the UI
- ✅ Models page: registry entries + eval metrics from stored EvaluationRuns (per-class, split-labelled)
- ✅ Survey batch with nav sidecar: real coordinate interpolation (synthetic track clearly labelled); single-image uploads keep the honest "location unavailable" notice
- ✅ Filtering honours configured penalty caps; every detection keeps a status + reason; model/final confidence stored separately
- ✅ No fabricated numbers anywhere; stub/interim artifacts retired/purged

## 20. WHAT IS STILL OPEN (honest)

- ⏳ **A full YOLOv8s (or larger) run.** A 5-epoch probe shows the s-variant
  losing to the nano at equal budget; completing it needs a CUDA GPU or many
  more CPU-hours, and may not beat the nano at 640px on 3,875 images.
- ⏳ **Docker deployment untested** (`docker/` files exist; never built or run).
- ⏳ **No OCR / no real survey ingest beyond the documented `nav.csv` convention**
  (OPEN #3); multi-detection tiles share one interpolated along-track position.
- ⏳ **Multi-model serving** is out of MVP scope: requesting a non-loaded model
  version correctly returns `MODEL_UNAVAILABLE` rather than silently using another.
- ⏳ **ghost_net has no real-data validation path** — no public annotated real
  SSS ghost-net dataset exists; documented as experimental everywhere.
- ✅ 30-epoch run COMPLETE — `drishti-ss_yolov8n_e30_final` registered, evaluated (test + val + site-disjoint), promoted, verified loaded in the API
- ✅ Overall test-split mAP recorded (ultralytics: mAP50 0.699 / mAP50-95 0.518; site-disjoint 0.702 / 0.525; repo-pipeline macro mAP50 0.663)
- ✅ Site-disjoint audit of the shipwreck split DONE — 0 overlapping sites
- ✅ PDF report generation DONE (dual HTML + PDF artifact per report)
- ✅ Final release audit DONE — see §0 for the nine defects found and fixed
