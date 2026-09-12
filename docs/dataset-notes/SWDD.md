# Dataset Review: SWDD (Sonar Wall Detection Dataset)

- **Reviewed by / date:** Buffy (agent) / 2026-09-09
- **Source / URL / license:** Zenodo record 13692547 (v2) · https://zenodo.org/records/13692547 · code: KD-YOLOX-ViT / ROSAR repos · public; required attribution: "SWDD is a public dataset collected with a Light Autonomous Underwater Vehicle by Oceanscan-MST, within the scope of the H2020 REMARO project."
- **Access status:** public download (SWDD.v2.zip, 5.3 GB)
- **Modality:** ✅ real **side-scan sonar** (LAUV + Klein 3500, Porto de Leixões harbor, 900 kHz, 75 m range per side)
- **Imaging geometry:** **true waterfall images** generated from raw ping streams via Neptus (4.168 px/line × 500 lines, later resized to 640×640); a separate 6m57s survey video + 6,243 extracted annotated frames are included — a realistic sequential stream for demoing temporal/batch inference
- **Format:** **COCO annotations**; 70/15/15 pre-split
- **Navigation metadata:** mission-level; per-frame geo OPEN — inspect after download
- **Label classes present (verbatim from data):** `wall`, `noWall` (2 classes; 216 original images + augmentations → 864 images / 2,616 boxes). SWDD-Validation adds SWDD-Clean/Surface/Noisy robustness variants (storm-noise, surface conditions)
- **Approx instance counts per class:** 2,616 boxes total across the augmented set
- **Annotation quality spot-check:** pending; published with two peer-reviewed papers (YOLOX-ViT KD; ROSAR)
- **Overlap with our target classes:** `wall` is a harbor structure, not marine debris — no direct match to provisional classes. Value is (a) the annotated **video stream** for batch/geolocation demos and (b) robustness variants for FP-filter stress-testing
- **License compatible with SIH demo use?** Yes with attribution
- **Verdict:** **SECONDARY (not debris)** — keep as an optional robustness/FP-filter stress dataset and as the best public source of *sequential* SSS frames; not a training source for our target classes
- **Next action:** optional. Download only if we need a real sequential stream for the survey-batch demo or adversarial/noise robustness experiments (SWDD-Adversarial variants exist for that).
