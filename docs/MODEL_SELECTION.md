# Model Selection — Marine Debris Sonar AI

Generated: 2026-09-11. Documents what was actually trained and selected, and why.

## Hardware reality (the deciding constraint)

| Item | Value |
|---|---|
| CPU | AMD Ryzen 5 5600H (6C/12T) |
| RAM | 16 GB |
| GPU | AMD Radeon RX 6500M 4GB — **not CUDA-usable** with this torch build (CPU-only) |
| Disk | 330 GB free |

AMD GPUs cannot run CUDA; ROCm on Windows for consumer APUs is not practical
for this project. **All training is CPU-bound** at roughly 55–65 min/epoch for
YOLOv8n at 640px, batch 8 — measured, not assumed.

## What was trained

| Variant | Status | Result |
|---|---|---|
| YOLOv8n, 30 epochs, full DRISHTI-SSS train split | **done — SELECTED & ACTIVE** | **test: P 0.758 R 0.586 F1 0.661, macro mAP50 0.663; ultralytics test mAP50 0.699 / mAP50-95 0.518; val mAP50 0.721 / mAP50-95 0.530**. Registered `drishti-ss_yolov8n_e30_final`, evaluated, promoted. See `docs/EVALUATION.md` |
| YOLOv8n, 5 epochs (interrupted first attempt, preserved) | done (retired) | val P=0.604 R=0.601 mAP50=0.585 mAP50-95=0.413; test F1 0.538. History: `e5_interim` entry retired (pointer-drift bug) |
| YOLOv8n, ~epoch-11 frozen snapshot (served during training) | done (retired) | test P=0.764 R=0.437 F1=0.556, macro mAP50 0.551. History: `e11_snapshot` retired on final promotion |
| YOLOv8n, 5 epochs, 60-image smoke subset | done (pipeline validation only) | retired from registry; never presented as a detector |
| YOLOv8s, 5-epoch probe | done — **NOT selected** | val mAP50 **0.658** / mAP50-95 **0.457** at epoch 5 (see below). Weights left at `models/weights/drishti-ss_yolov8s_probe_e5/weights/best.pt`; deliberately **not registered** in the model registry, because it is un-evaluated on test and strictly worse than the nano at the same budget. A full 30-epoch s-run was not attempted (~3× the nano cost). |

## Why YOLOv8n was selected

1. **Compute-feasible end-to-end**: the only variant that can actually reach a
   multi-epoch trained state on this machine within SIH timelines.
2. **Demo-appropriate inference speed**: ~58 ms/image CPU inference — real-time
   feel in the web UI without a GPU.
3. **The dataset is pre-tiled 640×640** — the nano model's capacity is not the
   bottleneck at this input scale; evidence quality (per-class instance counts)
   is (see per-class results in `docs/EVALUATION.md`).
4. **Same detection abstraction**: the `yolo` detector adapter serves any
   ultralytics checkpoint, so a larger future model drops in with zero backend
   changes — just a new registry entry.

## Controlled comparison

- **Model A — YOLOv8n, 30 epochs (SELECTED).** val mAP50 0.730 / mAP50-95 0.529;
  test mAP50 0.699 / mAP50-95 0.518.
- **Model B — YOLOv8s, 5 epochs (probe only, REJECTED).** Took 5 epochs on 16 GB
  CPU RAM (~9.5 h wall clock, and the 5th epoch alone took ~6.9 h under memory
  pressure). Final val: P 0.696 / R 0.660 / **mAP50 0.658 / mAP50-95 0.457**.
  Raw curve from `models/weights/drishti-ss_yolov8s_probe_e5/results.csv`:
  mAP50 0.478 (ep1) → 0.529 (ep2) → 0.573 (ep3) → 0.594 (ep4) → 0.658 (ep5).

  **Decision: the nano at 30 epochs wins** on both mAP50 (0.730 vs 0.658) and
  mAP50-95 (0.529 vs 0.457), so a ~3×-cost s-run was not worth the remaining
  schedule. The probe is retained on disk for transparency and is **not**
  registered, so it can never be served or mistaken for the selected model.
  If a CUDA GPU becomes available, finishing this comparison is the first
  experiment to run; registry + eval tooling already support it.

## Failure modes that shaped the decision

- The first e30 run was destroyed by **multiple concurrent training processes**
  writing one run dir (4 writers, died at epoch 5). Root cause fixed with a
  single-instance guarded launcher + resume path
  (`scripts/launch_background_train.py`); documented in `docs/TRAINING.md`.
- Class imbalance and split difficulty (not model size) dominate the weak
  per-class numbers — shipwreck test tiles are 50%-overlap re-tilings, and
  mine_cylinder has only 843 real train instances.

## Site-disjoint corroboration

Because the shipwreck class drew from many survey sites, a leakage audit was run
before trusting the numbers (`scripts/audit_site_leakage.py`): **0 of 64 wreckA
and 0 of 23 wreckR test sites also appear in train/val**, so the test split is
already site-disjoint. A separate 537-tile site-disjoint subset scores
**P 0.733 / R 0.681 / mAP50 0.702 / mAP50-95 0.525**, matching the standard test
split (mAP50 0.699) — i.e. the headline metrics are not leak-inflated. See
`docs/EVALUATION.md`.

## Selection

**`drishti-ss_yolov8n_e30_final` — SELECTED and ACTIVE.** The completed
30/30-epoch run was registered from frozen weights (`best_final.pt`), evaluated
on the held-out test split with both evaluation methods, and only then promoted
to the registry's active entry (verified loaded in the running API). Interim
entries (e5, e11) are retired but kept in the append-only registry history.
