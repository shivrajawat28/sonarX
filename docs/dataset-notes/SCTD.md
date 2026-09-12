# Dataset Review: SCTD (Sonar Common Target Detection) 1.0

- **Reviewed by / date:** Buffy (agent) / 2026-09-09
- **Source / URL / license:** https://github.com/MingqiangNing/SCTD (redirects from freepoet/SCTD) · format: **Pascal VOC XML** annotations (`voc2coco.py` provided by authors) · academic use; contact authors per README. No formal OSI license — treat as **request/verify before redistribution**.
- **Access status:** public GitHub (verify the data host in the repo — some SCTD mirrors link to cloud drives)
- **Modality:** ⚠️ **mixed**: 497 high-resolution images from **side-scan sonars, forward-looking sonars, and (interferometric) synthetic aperture sonars** — must filter to the SSS portion or accept domain mix
- **Imaging geometry:** mixed tile crops; heterogeneous resolution; not raw ping streams
- **Format:** VOC XML boxes → needs VOC→YOLO conversion (trivial: read `bndbox`, normalize) into our interim layout. A `voc_to_interim` helper is a small, one-time script.
- **Navigation metadata:** none
- **Label classes present (verbatim from data):** commonly reported as three classes ≈ 363 samples: **57 aircraft, 34 human, 271 shipwreck** (counts vary slightly between papers citing v1.0; confirm from the actual XMLs). Note `human`/`aircraft` are not in our provisional class list — they are config-droppable.
- **Approx instance counts per class:** 596 targets total across 497 images (≈1.2/image)
- **Annotation quality spot-check:** pending download; community usage in many papers suggests reasonable box quality, but class-boundary consistency between SSS/FLS/SAS crops must be eyeballed.
- **Overlap with our target classes:** `shipwreck` overlap (271 samples) is the largest publicly available wreck-box set; others droppable. Mixed modality is the main risk — sonar texture differs between FLS/SAS/SSS.
- **License compatible with SIH demo use?** Academic-use yes; keep out of any redistributed artifact, cite both papers (Zhang et al. TGRS 2022; Ning et al. TGRS 2025).
- **Verdict:** **FALLBACK / AUGMENTATION** — use the SSS+shipwreck subset to augment AI4Shipwrecks if bbox diversity is needed; not as the primary single-domain source.
- **Next action:**
  1. Download; inspect XMLs; tag each image with its sonar modality if identifiable (filename/paper split).
  2. Convert VOC→interim (YOLO txt) keeping only SSS images + `shipwreck` boxes for the merged experiment; keep full set on disk for flexibility.
  3. If merged with AI4Shipwrecks: run one experiment per source dataset first — never train mixed without per-domain eval runs.
