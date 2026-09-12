# Dataset Review: SubPipe (SSS portion)

- **Reviewed by / date:** Buffy (agent) / 2026-09-09
- **Source / URL / license:** Zenodo record 12666132 (v3.0.1) · https://zenodo.org/records/12666132 · repo: https://github.com/remaro-network/SubPipe-dataset · public dataset; required attribution text: "SubPipe is a public dataset of a submarine outfall pipeline, property of Oceanscan-MST … within the scope of Challenge Camp 1 of the H2020 REMARO project."
- **Access status:** public download. Full `SubPipe.zip` 28 GB (≈80 GB unzipped); **`SubPipeMini2.zip` 4.9 GB (≈16 GB unzipped) is the SSS-focused subsample** — use that, not the full archive.
- **Modality:** ✅ real **side-scan sonar** (LAUV + SSS, submarine outfall pipeline, OceanScan-MST)
- **Imaging geometry:** waterfall chunks; each SSS image = 20 pings (≈1 image/s). **LF: 5000 images 2500×500** · **HF: 5030 images 5000×500** — extreme aspect ratios (our `resize_letterbox` op is built for exactly this). LF vs HF = two visually distinct domains; treat as separate configs or one dataset with domain tag.
- **Format:** object-detection boxes in **both COCO and YLO formats** ("YOLO annotations are provided for each SSS image file") — directly consumable by `ml/scripts/convert_dataset.py` interim layout (images/ + labels/*.txt), likely with minimal or no conversion.
- **Navigation metadata:** AUV pose ground truth from INS is referenced for SLAM; per-image coordinates inside the SSS annotation files are OPEN — inspect after download. The chunked 1 Hz imagery is ideal for our navigation-track/fraction geolocation path if timestamps survive.
- **Label classes present (verbatim from data):** `pipeline` (single-class detection boxes; seabed otherwise unlabelled)
- **Approx instance counts per class:** 6,335 box annotations across 10,030 images (≈0.63/image — many pure-seabed negatives, good for FP-filter calibration)
- **Annotation quality spot-check:** not yet downloaded — **pending** (sample ≥20 across LF/HF).
- **Overlap with our target classes (provisional, config-only):** direct match for `pipe`. No wreck/net class.
- **License compatible with SIH demo use?** Yes — public with attribution text; REMARO (EU H2020) funded.
- **Verdict:** **PRIMARY CANDIDATE** (paired with AI4Shipwrecks) — real SSS, native YOLO boxes, single clean class matching a provisional class, large volume.
- **Next action:**
  1. Download `SubPipeMini2.zip`; hash; record.
  2. Map its YOLO labels into interim layout; run `convert_dataset.py` with `candidate_classes: [pipe]` and `--group-sep` set from the chunk naming scheme (leakage guard — consecutive waterfall chunks overlap).
  3. Confirm whether per-image timestamps/coords are recoverable → decides if real geolocation can be demoed on real data (closes part of OPEN #3).
