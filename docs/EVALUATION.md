# Evaluation — Marine Debris Sonar AI

Generated: 2026-09-11. All numbers below are **actual measured results** on the
DRISHTI-SSS dataset — none are estimated or copied from other work.

## Models evaluated

| Model version | Status | Checkpoint | Notes |
|---|---|---|---|
| `drishti-ss_yolov8n_e5_interim` | **retired** | ~~`best.pt` (epoch 5)~~ | RETIRED: its entry pointed at the live `best.pt`, which kept updating during training (had silently drifted to ~epoch-11 weights). Renamed honestly — see below |
| `drishti-ss_yolov8n_e11_snapshot` | **retired** | `models/weights/drishti-ss_yolov8n_e30/weights/best_snapshot_ep11.pt` (frozen copy) | Was the interim serving model while training ran; retired on final-model promotion |
| `drishti-ss_yolov8n_e30_final` | **ACTIVE (serving)** | `models/weights/drishti-ss_yolov8n_e30/weights/best_final.pt` (frozen copy of final `best.pt`) | COMPLETE 30/30-epoch run; registered, evaluated on test + val, then promoted |

> Integrity note: a registry entry must point at an immutable checkpoint. The
> interim entry was corrected as soon as the drift was discovered; the frozen
> `e11_snapshot` weights can never change, and its evaluation below matches
> exactly those frozen weights.

## Methodology

- **Split used: `test` (700 images, 901 instances — measured by scanning every
  label file)** — held-out split shipped with DRISHTI-SSS. Val split used during
  training only (ultralytics early-stopping/model selection). Numbers below are
  TEST, not val.
- **Framework:** repo eval pipeline (`ml.scripts.evaluate`) — greedy
  score-ordered matching at **IoU ≥ 0.5**, same-class only.
- **Confidence:** raw `model_confidence` (filter OFF / detector-only).
  A filter-ON run is recorded separately (ADR-010); it does not modify metrics.
- **Preprocessing:** loaded by hash from the model's registry record
  (`drishti_preprocessed`: resize-only — dataset is already Lee+CLAHE'd).
- **Dataset identity:** manifest `datasets/manifests/drishti-sss.json`
  (5,205 images SHA-256-hashed, 6,102 instances).
- Every eval run is stored immutable under `models/eval/`.

## Test-split results — `drishti-ss_yolov8n_e5_interim` (epoch-5 checkpoint)

Overall (IoU 0.5, detector-only):

| Metric | Value |
|---|---|
| Precision | **0.732** |
| Recall | **0.425** |
| F1 | **0.538** |

Per-class:

| Class | P | R | F1 | AP50 | Support (test instances) | Data nature |
|---|---|---|---|---|---|---|
| submarine_pipeline | 0.880 | 0.931 | **0.905** | 0.928 | 174 | Real |
| shipwreck | 0.721 | 0.152 | 0.252 | 0.192 | 525 | Real |
| ghost_net | 0.923 | 1.000 | **0.960** | 0.989 | 120 | **100% synthetic** |
| mine_cylinder | 0.214 | 0.256 | 0.233 | 0.111 | 82 | Real |

Cross-check (ultralytics `model.val()`, **val** split — not comparable to test
numbers above): P=0.604 R=0.601 mAP50=0.585 mAP50-95=0.413 at epoch 5.

## FINAL test-split results — `drishti-ss_yolov8n_e30_final` (30/30 epochs, ACTIVE)

Repo eval pipeline (greedy IoU-0.5 matching, detector-only, filter OFF):

| Metric | Value |
|---|---|
| Precision | **0.758** |
| Recall | **0.586** |
| F1 | **0.661** |
| mAP50 (macro of per-class AP50) | **0.663** |

Per-class:

| Class | P | R | F1 | AP50 | Support (test instances) | Data nature |
|---|---|---|---|---|---|---|
| submarine_pipeline | 0.994 | 0.983 | **0.988** | 0.964 | 174 | Real |
| shipwreck | 0.609 | 0.398 | 0.482 | 0.421 | 525 | Real |
| ghost_net | 1.000 | 1.000 | **1.000*** | 1.000* | 120 | **100% synthetic** |
| mine_cylinder | 0.452 | 0.341 | 0.389 | 0.266 | 82 | Real |

*synthetic-on-synthetic — never present as a field capability.

Ultralytics-native val (mAP family, frozen `best_final.pt`, **test** split):
**P 0.714 / R 0.705 / mAP50 0.699 / mAP50-95 0.518** — and on the **val**
split (training-domain selection set, NOT comparable to test): P 0.796 /
R 0.709 / mAP50 0.721 / mAP50-95 0.530. Both summaries stored at
`models/eval/e30_final_{val,test}_summary.json`.

Progression across this run (all TEST split, repo pipeline): F1 0.538 (ep5)
→ 0.556 (ep11) → **0.661 (ep30)**; shipwreck recall 0.152 → 0.173 → **0.398**.

### Site-disjoint corroboration (`test-site-disjoint`, 537 tiles / 754 instances)

The shipwreck test tiles come from sites that appear **only** in test — the
leakage audit found 0 overlapping sites out of 64 wreckA + 23 wreckR test sites
(`scripts/audit_site_leakage.py`). Re-running the frozen final weights on that
explicitly site-disjoint subset:

| Metric | Value |
|---|---|
| Precision | **0.733** |
| Recall | **0.681** |
| mAP50 | **0.702** |
| mAP50-95 | **0.525** |

Stored at `models/eval/e30_final_sitedisjoint_summary.json`; reproduce with
`python scripts/eval_sitedisjoint.py` (ultralytics val, CPU, 537 images,
~30 s). This was independently re-run during the final release audit and
reproduced the stored values **exactly**, confirming the numbers correspond to
the frozen `best_final.pt`. The tight agreement with the standard test split
(mAP50 0.699) is evidence that the shipwreck test metrics are not inflated by
site leakage.

## Test-split results — earlier checkpoints (transparency record)

### `drishti-ss_yolov8n_e11_snapshot` (frozen, ~epoch 11)

Overall (IoU 0.5, detector-only, filter OFF):

| Metric | Value |
|---|---|
| Precision | **0.764** |
| Recall | **0.437** |
| F1 | **0.556** |
| mAP50 (macro of per-class AP50) | **0.551** |

Per-class:

| Class | P | R | F1 | AP50 | Support (test instances) | Data nature |
|---|---|---|---|---|---|---|
| submarine_pipeline | 0.988 | 0.971 | **0.980** | 0.953 | 174 | Real |
| shipwreck | 0.632 | 0.173 | 0.272 | 0.210 | 525 | Real |
| ghost_net | 0.945 | 1.000 | **0.972** | 1.000 | 120 | **100% synthetic** |
| mine_cylinder | 0.189 | 0.171 | 0.179 | 0.039 | 82 | Real |

Validation-curve context (ultralytics val mAP50 by epoch, single writer):
0.585 (ep5) → 0.626 (ep6) → 0.633 (ep7) → 0.738 (ep25) → 0.726 (ep27);
mAP50-95 reached **0.526** by ep27.

## Honest reading of these results

1. **submarine_pipeline is strong** (F1 0.988 on real data) — the largest
   real-class evidence base (SubPipe strips) pays off.
2. **shipwreck recall is low (0.398)** on the test split. The DRISHTI-SSS test
   set deliberately uses 50%-overlap re-tiling and harder partial views, and 30
   epochs of CPU training is a small budget. This is the weakest real class after
   mine_cylinder and drives the overall recall down. Site-disjointness is *not*
   the cause (0 leaking sites; site-disjoint mAP50 is 0.702 vs 0.699 overall).
3. **ghost_net F1 1.000 is synthetic-on-synthetic** (procedural train + test
   tiles from the same generator). It must NEVER be presented as a field
   capability. It is shown for pipeline completeness only.
4. **mine_cylinder is weak (F1 0.389)** — smallest real class, limited
   diversity. Reported as-is.
5. These are **prototype numbers on one academic-style dataset**. No claim of
   real-world, cross-hardware, or operational generalization is made.
6. **Known failure case:** low-confidence, edge-clipped wreck boxes are the
   dominant error mode — the model does detect partial wrecks but with low
   confidence, and the deterministic `edge_clip` rule then flags/rejects them
   (reason retained, never deleted). See `docs/ENGINEERING_REPORT.md` § Failure
   cases.

## Inference speed (measured, CPU-only machine)

- Ultralytics val: ~58 ms/image inference (AMD Ryzen 5 5600H, no dGPU use —
  AMD RX 6500M is not CUDA-usable with this torch build).
- Full pipeline through the API under concurrent training load: ~4 s/image
  wall clock (contention with the 30-epoch training process).

## Qualitative inspection

- Live pipeline runs against real test images (e.g.
  `wreckA_Artificial_Reef_06_y1280_x0.jpg`) produce plausible wreck boxes at
  low confidence that are then **rejected** by deterministic filters
  (edge-clip + intensity rules) with reasons retained — filtering annotates,
  never deletes (ADR-006/007).
- The low-confidence + rejection behavior on the sampled image is consistent
  with the measured epoch-5 shipwreck recall: the model is not yet confident
  on partial/edge wreck views.

## Reproduce

```bash
# dataset manifest (once)
PYTHONPATH="ml;." python -m ml.scripts.inspect_dataset \
  --root datasets/processed/drishti-sss --name drishti-sss \
  --classes submarine_pipeline,shipwreck,ghost_net,mine_cylinder \
  --save-manifest datasets/manifests/drishti-sss.json

# test-split evaluation (filter OFF, detector-only)
PYTHONPATH="ml;." python -m ml.scripts.evaluate \
  --model drishti-ss_yolov8n_e30_final \
  --manifest datasets/manifests/drishti-sss.json --split test

# ultralytics-native val (mAP family) on the FROZEN final weights
.venv/Scripts/python -c "from ultralytics import YOLO; \
  YOLO('models/weights/drishti-ss_yolov8n_e30/weights/best_final.pt') \
  .val(data='datasets/processed/drishti-sss/data.yaml', split='test', device='cpu')"

# site-disjoint corroboration (537 unseen-site tiles)
python scripts/eval_sitedisjoint.py

# leakage audit (derives site ids from wreck filenames)
python scripts/audit_site_leakage.py
```

> Earlier checkpoints (e5, e11) are kept above for transparency. The ACTIVE
> model is `drishti-ss_yolov8n_e30_final` with the FINAL test metrics above.
