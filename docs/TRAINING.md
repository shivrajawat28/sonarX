# Training Guide — Marine Debris Sonar AI

## Prerequisites

```bash
# Create venv
python -m venv .venv
.venv/Scripts/pip install numpy opencv-python-headless pyyaml pandas pydantic pydantic-settings "fastapi[standard]" uvicorn python-multipart pytest httpx jinja2

# Install ML dependencies
.venv/Scripts/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/pip install ultralytics
```

## Dataset Preparation

The training data is based on **DRISHTI-SSS** (CC-BY-SA-4.0), located at `datasets/raw/drishti-sss/`.

```bash
# Prepare dataset (remaps class IDs, copies to datasets/processed/)
python scripts/prepare_drishti.py
```

This creates `datasets/processed/drishti-sss/` with:
- `train/images/`, `train/labels/` — 3,875 training images
- `val/images/`, `val/labels/` — 630 validation images
- `test/images/`, `test/labels/` — 700 test images
- `data.yaml` — Ultralytics dataset config

### Classes (remapped from DRISHTI)

Instance counts verified by scanning all train-split label files
(`docs/CLASS_TAXONOMY.md` has the full provenance):

| ID | Class | Real/Synthetic | Train instances |
|---|---|---|---|
| 0 | submarine_pipeline | Real | 1,000 |
| 1 | shipwreck | Real | 1,554 |
| 2 | ghost_net | **100% Synthetic** | 900 |
| 3 | mine_cylinder | Real | 843 |

## Training

### Quick Training (smoke test)

```bash
python scripts/train_yolo.py --epochs 5 --batch 4 --name smoke_test
```

### Full Training

```bash
# CPU (recommended for this machine)
python scripts/train_yolo.py --epochs 30 --batch 8 --name drishti-ss_yolov8n_e30

# With GPU (if available)
python scripts/train_yolo.py --epochs 100 --batch 16 --name drishti-ss_yolov8n_e100
```

### Training Configuration

| Parameter | Value | Notes |
|---|---|---|
| Model | YOLOv8n (nano) | 3.0M params, CPU-friendly |
| Image size | 640×640 | Ultralytics letterboxes each tile (source tiles are mixed-size: 47.5% are 640×640, 37.7% are 640×500, rest smaller). Inference letterboxes identically, so train/serve parity holds. |
| Batch size | 8 | Safe for 16GB RAM CPU |
| Epochs | 30 | ~26 hours on CPU |
| Seed | 42 | For reproducibility |
| Device | CPU | No GPU available |
| Augmentation | Sonar-safe | No hue/saturation, horizontal flip only |

## SINGLE-INSTANCE RULE (important)

**Never run two training processes against the same run directory.** The
first `drishti-ss_yolov8n_e30` attempt was destroyed this way: multiple
concurrent trainers wrote the same `results.csv` (4 duplicate rows per epoch)
and the run died at epoch 5. Use the guarded launcher:

```bash
# status / launch (refuses if a trainer is already running)
.venv/Scripts/python scripts/launch_background_train.py --status
.venv/Scripts/python scripts/launch_background_train.py             # new 30-epoch run

# resume an interrupted run from last.pt (replays original hyperparameters)
.venv/Scripts/python scripts/launch_background_train.py --resume
```

`scripts/train_yolo.py --resume <last.pt>` does the same in the foreground.
Resume replays ALL original args from the checkpoint — do not re-pass
hyperparameters on top of a resume.

### Run history (actual)

- **2026-09-10 19:19 UTC** — first e30 attempt: multiple concurrent writers
  (root cause: launcher script + .bat launched repeatedly), interrupted at
  epoch 5. `best.pt` @ epoch 5: val P=0.604 R=0.601 mAP50=0.585 mAP50-95=0.413.
  results.csv de-duplicated; checkpoint preserved.
- **2026-09-11 ~09:00 UTC** — resumed from epoch-5 `last.pt` as a single
  process. Epoch 6: mAP50 0.626; epoch 7: mAP50 0.633, mAP50-95 0.446.
  (Roughly 55–65 min/epoch solo; later epochs ~17 min/epoch.)
- During the run, a frozen epoch-11 snapshot (`best_snapshot_ep11.pt`) was
  registered as `drishti-ss_yolov8n_e11_snapshot` (active) so the app served a
  REAL frozen model — after discovering the earlier `e5_interim` entry pointed
  at the live `best.pt`, which had silently drifted to ~epoch-11 weights.
- **2026-09-11 ~16:20 UTC — RUN COMPLETE: 30/30 epochs** (~26,300 s total).
  Final `results.csv` single-writer, no duplicates.
- **2026-09-11 22:39 → 2026-09-12 08:07 UTC — YOLOv8s 5-epoch probe**
  (`drishti-ss_yolov8s_probe_e5`). Result: val mAP50 0.658 / mAP50-95 0.457 at
  epoch 5 — below the nano's 30-epoch 0.730 / 0.529. Rejected and deliberately
  left unregistered (it must never be served or mistaken for the selected model).
  Rationale in `docs/MODEL_SELECTION.md`.
- **Finalization (`scripts/finalize_training.py`, single watcher):** verified
  30/30 epochs → froze `best.pt` to `weights/best_final.pt` → registered
  `drishti-ss_yolov8n_e30_final` (append-only) → repo eval on TEST split
  (F1 0.661) → ultralytics val on test (mAP50 0.699 / mAP50-95 0.518) and val
  (mAP50 0.721 / mAP50-95 0.530) → promoted statuses (e11_snapshot → retired,
  e30_final → active). Backend restarted and verified loading the final model.

### Sonar-Safe Augmentations

- ✅ Horizontal flip (fliplr=0.5) — safe for side-scan sonar
- ✅ Brightness variation (hsv_v=0.3) — no color shifts
- ✅ Small rotation (±5°), translation, scale
- ✅ Mosaic (0.8) — standard detection augmentation
- ❌ Hue shift (hsv_h=0.0) — sonar is grayscale
- ❌ Saturation (hsv_s=0.0) — sonar is grayscale
- ❌ Vertical flip (flipud=0.0) — changes sonar physics

### Preprocessing Note

**DRISHTI-SSS data is already preprocessed** with Lee speckle filter + CLAHE.
Do NOT apply additional CLAHE or denoising. The preprocessing config
`ml/configs/preprocessing/drishti_preprocessed.yaml` only applies resize_letterbox.

Measured train/serve parity checks (final release audit):

- **Letterbox padding.** Ultralytics pads with **114** during training; the
  inference config pads with **0**. Measured impact on 30 real test images
  (10 pipe / 10 wreckA / 10 mine): **28 detections either way, mean confidence
  0.654 vs 0.654** — no practical difference, so the registered config was left
  unchanged (changing it would also invalidate the registry's config hash for a
  zero-gain edit). Documented here rather than silently "fixed".
- **Double preprocessing.** The dataset README states its Lee+CLAHE pass gave no
  accuracy gain over raw tiles once training augmentation was strong. It is
  retained because CLAHE'd input makes acoustic shadows easier to see for the
  downstream geometry check — not because it is a measured accuracy win.

## Evaluation

After training completes:

```bash
# Using ultralytics built-in validation
.venv/Scripts/python -c "
from ultralytics import YOLO
model = YOLO('models/weights/<run_name>/weights/best.pt')
results = model.val(data='datasets/processed/drishti-sss/data.yaml')
print(f'mAP50: {results.box.map50:.4f}')
print(f'mAP50-95: {results.box.map:.4f}')
print(f'Precision: {results.box.mp:.4f}')
print(f'Recall: {results.box.mr:.4f}')
"
```

## Model Registry

Register a trained checkpoint (reads class names from `data.yaml`, hashes the
preprocessing config, append-only):

```bash
.venv/Scripts/python scripts/register_trained_model.py \
  --version drishti-ss_yolov8n_e30_final \
  --checkpoint models/weights/drishti-ss_yolov8n_e30/weights/best.pt \
  --status active \
  --notes "YOLOv8n 30 epochs on DRISHTI-SSS (CPU, single-instance run)"
```

Entries live in `models/registry.json` with:
- Model version, architecture, weights path
- Class map, preprocessing config reference (+ sha256)
- Training configuration snapshot
- Status: `shadow` → `active` after evaluation

To activate a model:
```python
from mlpipeline.registry.models import ModelRegistry
reg = ModelRegistry('models/registry.json')
reg.set_status('model_version', 'active')
```

## Files Created

- `models/weights/<run_name>/` — training output
- `models/weights/<run_name>/weights/best.pt` — best checkpoint
- `models/weights/<run_name>/training_metadata.json` — run metadata
- `models/registry.json` — model registry entry
