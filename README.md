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
make test        # ml + backend tests (unit + integration + boundary lint) → 157 passing
make lint        # architecture boundary tests only
make typecheck   # frontend tsc --noEmit

# full browser E2E (needs backend :8000 + frontend :5173 running)
.venv/Scripts/python scripts/make_e2e_survey_fixture.py   # one-off survey fixture
cd frontend && node e2e_full.mjs                          # 45 assertions, exits non-zero on failure
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
- **Geolocation is survey-only.** A standalone image has no navigation data, so lat/lon are `null` with an explicit reason. Coordinates appear only for a survey whose `nav.csv` / `navigation.csv` sidecar was parsed — see the **Survey (batch)** page.
- The final class list is **fixed to the 4 DRISHTI-SSS classes** (see `docs/CLASS_TAXONOMY.md`); the dataset choice is documented and pinned, not still open.

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
