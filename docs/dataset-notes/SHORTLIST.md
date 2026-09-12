# Dataset Shortlist — closing OPEN #1 (classes) & OPEN #9 (dataset source)

> Researched 2026-09-09. Per-dataset details live in sibling review files
> (created from `TEMPLATE-dataset-review.md`). Index source:
> [OpenSonarDatasets (REMARO)](https://github.com/remaro-network/OpenSonarDatasets)
> and [Awesome-Sonar-Image-Resources](https://github.com/Jorwnpay/Awesome-Sonar-Image-Resources).
> **All sizes/licenses must be re-verified at download time.**

## Recommendation (two-dataset MVP)

**Primary pair:** `AI4Shipwrecks` (shipwreck, segmentation→bbox) + `SubPipeMini2` (pipe, native YOLO boxes).
This yields the initial class list **`[shipwreck, pipe]`** — both real side-scan sonar, both
publicly licensed for academic use, both matching provisional classes from the architecture.

**Augmentation (optional, second experiment):** SCTD's SSS+shipwreck subset (VOC XML → convert),
keeping per-domain evaluation runs — never report mixed-domain metrics as one number.

**Explicitly out of scope:** ghost nets. No public annotated ghost-net/fishing-gear sonar
detection dataset exists (GhostNetZero/WWF collects but does not release data). The architecture
anticipated this: `net` stays a config-only class name, the demo presents detections for the
classes we actually train, and honest-limitations language covers the rest.

## Shortlist table

| Dataset | Modality | Classes (verbatim) | Labels | Size | Nav/geo | License | Verdict |
|---|---|---|---|---|---|---|---|
| **AI4Shipwrecks** | real SSS | shipwreck (+seafloor bg) | pixel masks | 286 img / 28 wrecks | site-level | open, cite IJR 2025 | **PRIMARY** |
| **SubPipe (Mini2)** | real SSS | pipeline | COCO+YOLO boxes | 10,030 img / 6,335 boxes | AUV INS GT (per-image geo TBD) | public + attribution text | **PRIMARY** |
| SCTD 1.0 | mixed SSS/FLS/SAS | shipwreck(271), aircraft(57), human(34) | VOC XML boxes | 497 img / 596 targets | none | academic, no OSI license | FALLBACK / augment |
| SeabedObjects-KLSG | real SSS | wreck(385), airplane(62) [full: +victim, mine, seafloor] | class folders only (weak boxes) | 447–1,190 img | none | academic | FALLBACK / negatives |
| SWDD (+Validation) | real SSS | wall, noWall | COCO boxes | 864 img (+6,243 video frames) | mission-level | public + attribution text | SECONDARY (robustness/stream demos) |
| UATD | **FLS** (not SSS) | cube, ball, cylinder, tyre, cages, plane, ROV, body, bucket | detection boxes | 9,200 img | none | public (Pengcheng Lab) | REJECT for training — wrong modality for a side-scan product; usable only as aux pretraining |
| Marine_PULSE | real SSS | pipes, mounds, platforms | class folders (classification) | 627 img | none | per paper | REJECT — classification-only, no boxes |
| GhostNetZero (WWF) | real SSS | ghost nets | internal | not released | — | — | REJECT (data closed); monitor for release |
| BenthiCat | real SSS tiles | 26 classes (benthic habitats) | segmentation tiles | ~950k tiles (~37k labelled) | GeoTIFF/XTF raw | open (2025) | OUT OF SCOPE — habitat mapping, not debris |

## How this closes the OPEN decisions

- **OPEN #1 (final target classes):** evidence-backed MVP proposal → **`classes: [shipwreck, pipe]`**
  (config change in dataset YAML + re-register model — zero code change, as designed).
  `net`/`anomaly` remain future classes pending a released dataset.
- **OPEN #9 (dataset source):** AI4Shipwrecks + SubPipe; downloads are direct public links;
  no signup-gated or proprietary hosts among primaries.
- **OPEN #2 (sonar file formats):** primaries are standard **PNG/JPEG + COCO/VOC/YOLO sidecars** —
  exactly the interim layout our converter consumes. Raw vendor formats (XTF etc.) remain out of
  scope (Section 24), with BenthiCat/Aurora noted as future raw-data sources if ever needed.
- **OPEN #3 (navigation metadata):** SubPipe's AUV INS ground truth is the best chance of a real
  geolocation demo on real data; per-image timestamp/coordinate recovery is the first thing to
  check after download. If absent, geolocation demos stay on the (labelled) synthetic survey —
  never fabricated.
- **OPEN #4 (bbox vs segmentation):** AI4Shipwrecks' masks convert losslessly to detection boxes
  for the YOLO MVP; masks are preserved on disk if segmentation is upgraded later.

## Next actions (in order)

1. Download AI4Shipwrecks + SubPipeMini2 into `datasets/raw/` (record SHA-256 + size in each review doc).
2. Spot-check ≥20 images each (annotation quality per template).
3. Write `mask→bbox` converter for AI4Shipwrecks; drop SubPipe's native YOLO labels into interim layout.
4. `ml/scripts/convert_dataset.py` → manifests under `datasets/manifests/` (group/leakage guard ON for SubPipe chunks).
5. `make train`-equivalent run per dataset (YOLO backend), one eval run per dataset **and** one on the union — three separate EvaluationRuns, honestly labelled.
6. Update `ml/configs/datasets/*.yaml` with `candidate_classes: [shipwreck, pipe]`; re-register the model; promote only after eval.
