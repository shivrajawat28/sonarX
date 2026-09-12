# Dataset Audit — Marine Debris Sonar AI

Generated: 2026-09-11

## 1. DRISHTI-SSS (PRIMARY — ONLY FULLY AVAILABLE DATASET)

**Location:** `datasets/raw/drishti-sss/`

**Source:** HuggingFace `rehan9599/drishti-sss`
**GitHub:** https://github.com/Rehan9599/Sonar-Drishti
**SIH Problem Statement:** SIH26057 (same as ours)

### Overview

Assembled, preprocessed side-scan sonar training splits for marine debris and anomaly detection. YOLO format, 640px tiles.

### License

CC-BY-SA-4.0 (ShareAlike inherited from CC-BY-SA-4.0 sources: mine set and crab_pot set).

### Attribution

| Portion | Source | Licence | Nature |
|---|---|---|---|
| `pipe`, `bg` | SubPipe / SubPipeMini2 (OceanScan-MST) | CC-BY-4.0 | Real SSS |
| `wreckA` | AI4Shipwrecks (UM Field Robotics / NOAA Thunder Bay) | CC-BY-4.0 | Real SSS |
| `wreckR` | Side Scan Sonar (Dae Hyeok Lee, Roboflow Universe) | CC-BY-4.0 | Real SSS |
| `mine` | Sonar Imaging Mine Detection (MILCO contacts, Kaggle) | CC-BY-SA-4.0 | Real SSS |
| `synth` | Procedural acoustic generator | CC-BY-SA-4.0 | **Synthetic** |
| `bg` | Object-free SubPipeMini2 tiles | CC-BY-4.0 | Real SSS (hard negatives) |

### Image Counts

| Split | JPG | PNG | Total Images | Labels | Empty Labels (bg) |
|---|---|---|---|---|---|
| train | 2,625 | 1,250 | **3,875** | 3,875 | 500 |
| val | 510 | 120 | **630** | 630 | — |
| test | 580 | 120 | **700** | 700 | — |
| **total** | **3,715** | **1,490** | **5,205** | **5,205** | ~500 |

### Annotation Format

YOLO bounding box: `class_id x_center y_center width height` (normalised 0–1).

### Image Properties

- Format: JPEG + PNG (grayscale sonar imagery)
- **Dimensions: MIXED — not uniformly 640×640.** Measured by decoding all 5,205
  images (394 distinct shapes):

  | Size | Count | Share |
  |---|---|---|
  | 640×640 | 2,470 | 47.5% |
  | 640×500 | 1,961 | 37.7% |
  | 1024×1024 | 155 | 3.0% |
  | 416×416 | 150 | 2.9% |
  | 224×224 | 19 | 0.4% |
  | other (389 shapes) | 450 | 8.6% |

  **Consequence:** 52.5% of tiles are not square, so the model input requires a
  letterbox (resize + pad) to 640×640. A box's position therefore differs between
  *source* pixel space and *processed* pixel space — e.g. a 640×500 tile gains a
  70 px top pad, shifting every `y` by 70. The API exposes both spaces
  (`bbox_source_coords`, `bbox_processed_coords`) and the UI draws whichever
  matches the image on screen. Any consumer that mixes the two will misplace boxes.
- Content: Side-scan sonar imagery with acoustic shadows, seabed texture, and man-made objects

### Label Distribution (Train Split)

| Prefix | Count | Source | Real/Synthetic | Class(es) |
|---|---|---|---|---|
| `synth` | 1,250 | Procedural generator | **Synthetic** | ghost_net (3) |
| `pipe` | 1,000 | SubPipeMini2 survey strips | Real | submarine_pipeline (1) |
| `wreckA` | 546 | AI4Shipwrecks transects | Real | shipwreck (2) |
| `bg` | 500 | SubPipeMini2 (no objects) | Real | Background (no labels) |
| `wreckR` | 354 | Roboflow SSS | Real | shipwreck (2) |
| `mine` | 225 | Kaggle MILCO contacts | Real | mine_cylinder (4) |

### Per-Class Annotation Counts (measured by scanning every label file)

Train split (raw DRISHTI ids 1–4 = processed ids 0–3 after crab_pot removal):

| Processed ID | Class Name | Train Instances | Real/Synthetic |
|---|---|---|---|
| — | crab_pot | **0** (excluded upstream) | N/A |
| 0 | submarine_pipeline | 1,000 | Real |
| 1 | shipwreck | 1,554 (multi-box files) | Real |
| 2 | ghost_net | 900 | **100% Synthetic** |
| 3 | mine_cylinder | 843 | Real |

All splits combined (from the hashed dataset manifest
`datasets/manifests/drishti-sss.json` — 5,205 images, 6,102 instances,
validation OK):

| Class | Instances (train+val+test) |
|---|---|
| shipwreck | 2,623 |
| submarine_pipeline | 1,321 |
| ghost_net | 1,140 (synthetic) |
| mine_cylinder | 1,018 |

### Preprocessing Already Applied

**Every tile has been through Lee speckle filter + CLAHE.**

- Lee kernel: 7×7
- CLAHE: clip 3.0, 8×8 grid

**CRITICAL: Do NOT apply CLAHE again. Do NOT apply speckle filtering again.**

The README notes: "an ablation found this preprocessing gave no accuracy gain over raw tiles once training-time augmentation was strong — it is retained because CLAHE'd input makes acoustic shadows more detectable for the downstream geometry check."

### Split Strategy

The dataset ships with pre-defined train/val/test splits. We use them as-is.

### Known Caveats

1. **Shipwreck split IS site-disjoint (audited).** `scripts/audit_site_leakage.py`
   derives a site id from each wreck filename and compares splits. Measured
   result — **zero site overlap** between train/val and test:

   | Subset | train sites | val sites | test sites | overlapping with train/val | leaked test tiles |
   |---|---|---|---|---|---|
   | wreckA (AI4Shipwrecks) | 72 | 13 | 64 | **0** | **0 / 277 (0.0%)** |
   | wreckR (Roboflow) | 289 | 93 | 23 | **0** | **0 / 23 (0.0%)** |

   So the reported shipwreck metrics are on unseen sites, not leaked ones. (An
   earlier revision of this document claimed the audit had not been done and
   called the metrics optimistic; that was incorrect and is corrected here.)
   A corroborating run on the 537-tile site-disjoint subset gives mAP50 0.702 /
   mAP50-95 0.525 — see `docs/EVALUATION.md`.
2. **Test set is deliberately hard.** 50%-overlap re-tile tripled shipwreck test instances with partial and near-duplicate tiles. Not comparable to papers using non-overlapping tiling. These are re-tilings *within* the held-out sites, not cross-split leakage.
3. **ghost_net evaluation is synthetic-on-synthetic.** Not a field number.
4. **Class imbalance is deliberate.** Per-class caps per split.
5. **crab_pot excluded.** Class 0 has zero examples in this release.

### Disk Usage

~2.0 GB total.

### Suitability Assessment

**SUITABLE as primary training dataset.** This is a real, multi-source, YOLO-formatted side-scan sonar dataset specifically assembled for SIH26057. It covers pipeline inspection, shipwreck detection, mine detection, and ghost net detection — directly matching our target use cases.

**Limitations:**
- Relatively small (~3,700 annotated training images)
- Ghost net class is 100% synthetic
- Mixed tile dimensions (only 47.5% are 640×640) → every consumer must be explicit about source vs processed pixel space
- No geolocation/navigation metadata included
- Already preprocessed (doubles preprocessing risk)
- Single sonar system per class: no cross-hardware generalization evidence

---

## 2. AI4Shipwrecks (INCOMPLETE)

**Location:** `datasets/raw/ai4shipwrecks/AI4Shipwrecks.zip`
**Size:** ~95 MB (zip, but validation failed — **corrupt or incomplete download**)
**Status:** Cannot be used. ZIP integrity check failed.
**Source:** https://umfieldrobotics.github.io/ai4shipwrecks/

Note: Portions of AI4Shipwrecks data are already included in DRISHTI-SSS as the `wreckA` prefix.

---

## 3. SubPipeMini2 (INCOMPLETE)

**Location:** `datasets/raw/subpipe/SubPipeMini2.zip.part`
**Size:** ~1.1 GB (partial download — **.zip.part**)
**Status:** Cannot be used. Incomplete download.
**Source:** https://zenodo.org/records/10808161

Note: Portions of SubPipeMini2 data are already included in DRISHTI-SSS as the `pipe` and `bg` prefixes.

---

## 4. GhostVision (INCOMPLETE)

**Location:** `datasets/raw/ghostvision/GhostVision_DatasetAndModels.zip.part`
**Size:** ~640 MB (partial download — **.zip.part**)
**Status:** Cannot be used. Incomplete download.
**Source:** https://huggingface.co/datasets/PINGEcosystem/sss-crab-pot-detection-ds

Note: Access-gated on HuggingFace. Crab pot data was excluded from DRISHTI-SSS anyway (class 0).

---

## 5. Kaggle SSS (EMPTY)

**Location:** `datasets/raw/kaggle-sss/`
**Status:** Empty directory. No data present.

---

## Summary

| Dataset | Status | Images | Usable | Notes |
|---|---|---|---|---|
| **DRISHTI-SSS** | **✅ Complete** | **5,205** | **Yes** | Primary dataset, CC-BY-SA-4.0 |
| AI4Shipwrecks | ❌ Corrupt zip | — | No | Partial data in DRISHTI-SSS |
| SubPipeMini2 | ❌ Partial download | — | No | Partial data in DRISHTI-SSS |
| GhostVision | ❌ Partial download | — | No | Crab pot class excluded anyway |
| Kaggle SSS | ❌ Empty | — | No | — |

**Decision: Train exclusively on DRISHTI-SSS.** It is the only complete, properly formatted dataset available. The other datasets' contributions are already incorporated into DRISHTI-SSS.
