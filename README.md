# Marine Debris Sonar AI

**AI-Powered Automated Underwater Marine Debris & Anomaly Detection** (Smart India Hackathon).

Side-scan sonar imagery/survey data → preprocessing → AI detection → false-positive filtering → geolocation (when real navigation data exists) → structured results → React dashboard.

> 📐 **The architecture lives in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — it is the source of truth for this repository.**

## Current model (REAL, trained, serving)

The active registry model is **`drishti-ss_yolov8n_e30_final`** — a YOLOv8n
trained for the full 30 epochs on the 3,875-image DRISHTI-SSS train split
(CPU, ~26.3k seconds). **Measured test-split results** (held-out; both methods
agree):

| Metric | Repo pipeline (IoU 0.5) | Ultralytics val |
|---|---|---|
| Precision | 0.758 | 0.714 |
| Recall | 0.586 | 0.705 |
| F1 | **0.661** | — |
| mAP50 | 0.663 (macro) | 0.699 |
| mAP50-95 | — | 0.518 |

Per-class (test): submarine_pipeline **F1 0.988** · shipwreck F1 0.482 ·
mine_cylinder F1 0.389 · ghost_net F1 1.000 (**⚠️ 100% synthetic — never a
field claim**). A **site-disjoint** subset (537 tiles, no train↔test site
overlap) corroborates it at **mAP50 0.702 / mAP50-95 0.525**. Full numbers,
per-class tables, and honest limitations:
[`docs/EVALUATION.md`](docs/EVALUATION.md). Weights:
`models/weights/drishti-ss_yolov8n_e30/weights/best_final.pt`.

> **Release audit (2026-09-12):** the report of the final audit — nine defects
> found, fixed at the root cause, and regression-tested — is
> [`docs/ENGINEERING_REPORT.md`](docs/ENGINEERING_REPORT.md) §0. The most
> user-visible one: the detection overlay was drawn in the wrong pixel space
> (the preprocessing letterbox offset), so boxes sat up to ~70 px off the object;
> now verified aligned to 0.00 px in the browser E2E.

## Status

Implementation follows the architecture's Section 25 order (scaffolding → canonical datatypes → config → preprocessing → dataset tooling → detector abstraction → training/eval → filtering → geolocation → inference engine → backend → API → jobs → frontend → E2E → deployment → polish). See that document for what exists at each step.

## Quickstart (development)

Prerequisites: Python 3.11+, Node 18+.

```bash
# Create virtual environment
python -m venv .venv
.venv/Scripts/pip install numpy opencv-python-headless pyyaml pandas pydantic pydantic-settings "fastapi[standard]" uvicorn python-multipart pytest httpx jinja2

# Install ML dependencies (CPU)
.venv/Scripts/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/pip install ultralytics

# Prepare dataset
python scripts/prepare_drishti.py

# Copy environment config
cp .env.example .env

# Start backend + frontend (two terminals)
make dev         # if GNU make is installed
# ...or without make (Windows/Git-Bash):
bash scripts/dev.sh
# backend:  PYTHONPATH="ml;." .venv/Scripts/python -m uvicorn backend.app.main:app --port 8000
# frontend: cd frontend && npm run dev
```

Alternatively, use the Makefile shortcuts (`make venv` + `make dev`) if GNU make
is installed — the Makefile is Windows-compatible; otherwise run the raw
commands above or `bash scripts/dev.sh`.

- API health: <http://localhost:8000/api/v1/health>
- Dashboard: <http://localhost:5173>

## Demo data (clearly labelled fixtures)

```bash
make seed        # seeds a demo survey (SYNTHETIC tiles + SYNTHETIC nav track) + stub model
# (no make? run: .venv/Scripts/python scripts/seed_demo.py)
```

Everything seeded is marked `demo_fixture: true` and labelled in the UI/exports.
It exists to exercise the pipeline (upload → batch → geolocation → map → exports → report),
never to claim real detections or real locations. Idempotent: re-running skips if present.

## ML workflow (CLI)

```bash
# 1. inspect + validate a converted YOLO-format dataset, write manifest
PYTHONPATH="ml;." python -m ml.scripts.inspect_dataset --root datasets/my_ds --name my_ds --classes classA,classB --save-manifest datasets/manifests/my_ds.json

# 2. train (yolo backend = real training; guarded single-instance launcher recommended)
PYTHONPATH="ml;." python -m ml.scripts.train --version yolo-exp-001 --dataset-config ml/configs/datasets/my_ds.yaml --backend yolo
# ...or: python scripts/train_yolo.py --epochs 30 --batch 8   (see docs/TRAINING.md)

# 3. evaluate → immutable EvaluationRun (precision/recall/F1/AP50 per class + macro mAP50)
PYTHONPATH="ml;." python -m ml.scripts.evaluate --model yolo-exp-001 --manifest datasets/manifests/my_ds.json --split test

# 4. ultralytics-native mAP family on the frozen final weights
.venv/Scripts/python -c "from ultralytics import YOLO; YOLO('models/weights/drishti-ss_yolov8n_e30/weights/best_final.pt').val(data='datasets/processed/drishti-sss/data.yaml', split='test', device='cpu')"

# single-image pipeline smoke test (stub output — not a detection claim)
bash -c 'PYTHONPATH="ml;." .venv/Scripts/python -m ml.scripts.predict_image --image ml/tests/data/fixture_sonar.png'

# real-model + API audit: 6 real test tiles through the running API,
# verifying provenance, confidence/filtering separation, bbox scaling,
# exports and report
.venv/Scripts/python scripts/audit_inference_e2e.py

# site-disjoint evaluation (537 unseen-site tiles) + leakage audit
.venv/Scripts/python scripts/eval_sitedisjoint.py
.venv/Scripts/python scripts/audit_site_leakage.py
```

Geolocation in survey batches uses the survey's real parsed navigation track;
per-image along-track position falls back to ordered survey position (OPEN #3
MVP fallback) — coordinates are only ever computed from real navigation data.

## Dataset

**Primary dataset: DRISHTI-SSS** (CC-BY-SA-4.0) — 5,205 side-scan sonar images
with YOLO-format annotations across 4 classes. Already preprocessed with Lee
speckle filter + CLAHE. Splits: train 3,875 / val 630 / test 700 (901 test
instances measured). **Tile sizes are mixed** — only 47.5% are 640×640, 37.7%
are 640×500 — so a letterbox is required and *source* vs *processed* pixel space
differ; the API exposes both and the UI draws the right one. See
[`docs/DATASET_AUDIT.md`](docs/DATASET_AUDIT.md).

**Classes:** submarine_pipeline, shipwreck, ghost_net (⚠️ synthetic), mine_cylinder.
See [`docs/CLASS_TAXONOMY.md`](docs/CLASS_TAXONOMY.md).

**Preparation:** `python scripts/prepare_drishti.py`

**Training:** `python scripts/train_yolo.py --epochs 30 --batch 8`
(see [`docs/TRAINING.md`](docs/TRAINING.md) — single-instance rule + resume)

## Tests

```bash
make test        # ml + backend tests (unit + integration + boundary lint) → 173 passing
make lint        # architecture boundary tests only
make typecheck   # frontend tsc --noEmit

# full browser E2E (needs backend :8000 + frontend :5173 running)
.venv/Scripts/python scripts/make_e2e_survey_fixture.py   # one-off survey fixture
cd frontend && node e2e_full.mjs                          # 51 assertions, exits non-zero on failure
```

## Registering a trained model

```bash
.venv/Scripts/python scripts/register_trained_model.py --version <new-id> \
  --checkpoint models/weights/<run>/weights/best.pt --status active --notes "..."
```

Reads class names from `datasets/processed/drishti-sss/data.yaml`, hashes the
preprocessing config into the record, and appends to the append-only registry.
Never overwrite an existing version id — retire the old entry instead.

## Honest-scope notes (read before demoing)

- **No fake AI.** The REAL trained model (`drishti-ss_yolov8n_e30_final`) is the registry's active entry and what the API loads. Without any trained model registered, inference endpoints run a *stub detector* clearly labeled as a test/demo fixture, or degrade per architecture Section 11.4. Never present stub output as real detection.
- **No fabricated coordinates.** Detections only get lat/lon when real navigation metadata was parsed; otherwise geo fields are `null` with an explicit `geo_status` reason.
- **Model confidence is not a probability of correctness.** The UI shows raw `model_confidence` and filter-adjusted `final_confidence` separately, with `filtering_status` and reasons.
- Classes, thresholds, preprocessing chains, and filter rules are **configuration**, not code — see `ml/configs/`.
- **The Models page confusion matrix is a detection matrix, not a classifier matrix:** rows = ground truth (plus `background` = predictions that matched no object) and columns = predicted class (plus `missed` = objects the model did not detect), so each class row sums to that class's support and the `background` row total is the false-positive count. It agrees with the per-class P/R shown beside it — e.g. shipwreck 209 + 316 = 525 support at recall 0.398.
- **Every report states the detector threshold that produced it** (`applied_confidence_threshold`, read back from the run), because the workbench threshold is a user choice: 0.05 and 0.25 give the same model very different review loads.
- **Geolocation is survey-only.** A standalone image has no navigation data, so lat/lon are `null` with an explicit reason. Coordinates appear only for a survey whose `nav.csv` / `navigation.csv` sidecar was parsed — see the **Survey (batch)** page.
- The final class list is **fixed to the 4 DRISHTI-SSS classes** (see `docs/CLASS_TAXONOMY.md`); the dataset choice is documented and pinned, not still open.

## Analysis Modes: Live AI vs Demo Mode

The Workbench interface provides two modes accessible via the top status toggle:

### 1. LIVE AI ANALYSIS (Default)
- Direct end-to-end inference using the active YOLOv8n checkpoint (`drishti-ss_yolov8n_e30_final`).
- **Data flow:** User uploads sonar tile (`.png`, `.jpg`, `.jpeg`) → FastAPI validates dimensions/decompression bomb limits → OpenCV preprocessing (`Lee speckle filter + CLAHE`) → PyTorch YOLOv8n inference → Postprocessing & NMS → Confidence & False Positive Filtering (edge clipping, size bounds, aspect ratio) → JSON response → React SVG overlay aligned to source pixel space.
- Raw model confidence and filter-adjusted final confidence are preserved separately.
- Real processing latency (~80–120 ms on modern CPU) is measured and reported.

### 2. DEMO MODE — PRECOMPUTED VERIFIED SAMPLES
For guaranteed determinism during hackathon stage presentations without network/system variability, 3 verified DRISHTI-SSS test split samples are embedded with exact ground-truth and real pipeline outputs:
1. **Submarine Pipeline** (`pipe_1693569383.780_x3500.jpg`): High-confidence pipeline detection (~65.3% raw and final), status `accepted`.
2. **Shipwreck with Edge Clipping** (`wreckA_Artificial_Reef_06_y1280_x0.jpg`): Shows separation between raw confidence (84.7%) and final confidence (54.7%) due to penalty for clipping the sonar tile edge (`flagged`). A second co-occurring detection is evaluated and accepted (~65.0%).
3. **Acoustic Background / Negative Control** (`bg_1693569262.760_x0.jpg`): Clean sea-floor acoustic return verifying 0 false positive detections.

> ⚠️ **Integrity Guarantee:** DEMO MODE is visibly labelled in the UI with a persistent amber badge (`DEMO • PRECOMPUTED REAL SAMPLE`). Demo fixtures are never represented as freshly computed live inference.

## Hardware & System Requirements

- **Operating System:** Windows 10/11, Ubuntu 20.04+, or macOS 12+
- **Python:** 3.11 or 3.12 (`.venv` virtual environment)
- **Node.js:** v18.0.0 or higher (npm v9+)
- **CPU:** Quad-core x86_64 processor or Apple Silicon (M-series). Real-time CPU inference executes in ~90ms per tile.
- **RAM:** Minimum 8 GB (16 GB recommended for large batch survey ZIP archives).
- **GPU (Optional):** NVIDIA GPU with CUDA 11.8+ (PyTorch will automatically utilize CUDA if present; defaults gracefully to CPU).
- **Storage:** ~3 GB free space (including DRISHTI-SSS dataset and model weights).

## Verification & Test Status

- **Backend & ML Unit/Integration Tests:** 173 / 173 passing (`pytest ml/tests backend/tests -q`)
- **Frontend Type Safety:** Clean (`npm run typecheck` passes with 0 errors)
- **Production Build:** Clean (`npm run build` generates optimized `dist/` bundle)
- **Browser E2E Testing:** 51 / 51 automated assertions passing across Workbench, Survey, History, and Models pages.
- **Console & Network Errors:** 0 unhandled exceptions or 4xx/5xx errors in normal user flows.

## Security & Prototype Notice

> ⚠️ **Classification: Local SIH Hackathon Prototype**
> This application is built as an engineering prototype for the Smart India Hackathon. It contains comprehensive defensive input validation (image dimension limits up to 4096×4096, decompression bomb mitigation, ZIP slip path canonicalization, max archive size limits, CSV formula injection neutralization, XSS-safe text rendering, and security response headers). It is designed to run in trusted local/intranet environments and **does not** include public multi-tenant user authentication or rate limiting.

## Recommended 3–5 Minute Judge Presentation Flow

1. **System Status Overview:** Open Workbench (`http://localhost:5173`), highlight backend connectivity, active model version (`drishti-ss_yolov8n_e30_final`), and device runtime.
2. **Demo Sample 1 (Submarine Pipeline):** Switch to DEMO MODE, load Pipeline tile. Show detected bounding box, 65.3% confidence, and `accepted` status.
3. **Pipeline Stages Inspection:** Expand the Preprocessing and Filtering accordions to demonstrate the transparent pipeline stages (Lee speckle filter → CLAHE → YOLOv8n → geometric validation).
4. **Demo Sample 2 (Shipwreck - Edge Clipping):** Load Shipwreck tile. Point out the difference between `model_confidence` (84.7%) and `final_confidence` (54.7%) caused by the edge-clipping filter rule (`flagged`).
5. **Demo Sample 3 (Negative Control Background):** Load Background tile. Demonstrate 0 false positive detections on acoustic seabed clutter.
6. **Switch to LIVE AI ANALYSIS:** Toggle back to Live Mode. Upload a real sonar image from the test set or local disk.
7. **Execute Live Inference:** Click "Analyze Sonar Tile". Highlight real processing latency (~90 ms) and real model response.
8. **Inspect Model Governance:** Navigate to `/models`. Show the immutable evaluation metrics (mAP50: 0.663 macro, mAP50-95: 0.518, per-class P/R), the detection confusion matrix, and training provenance.
9. **Export Structured Artifacts:** Download JSON / CSV export from the results panel, demonstrating formula-injection-safe CSV and structured schema.
10. **Survey & History (Batch Geolocation):** Show `/survey` and `/history`, explaining how navigation metadata (`nav.csv`) derives true geographic coordinates, while standalone tiles display honest "Location unavailable" notices.

## Pages

| Route | Purpose |
|---|---|
| `/` Workbench | single image: upload → preprocessing preview → run model → overlay + detections + filtering + exports + report |
| `/survey` Survey (batch) | survey archive with optional nav sidecar → async batch job → **geolocated** detections on a map |
| `/history` History | every stored detection, filterable, with analyst status override |
| `/models` Models | registry entries, the loaded model, and metrics from stored evaluation runs (split-labelled, with confusion matrix + training provenance) |

## Repo layout

See `docs/ARCHITECTURE.md` Section 5 for the full annotated tree: `ml/` (the brain, HTTP-free), `backend/` (thin FastAPI bridge), `frontend/` (visualization only), `datasets/`, `models/`, `data/` (runtime), `docs/`, `e2e/`, `scripts/`.

Hard boundaries (lint-enforced): `ml` never imports FastAPI/HTTP; frontend contains no ML/business logic.

