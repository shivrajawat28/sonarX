# Dataset Review: AI4Shipwrecks

> Completed from public documentation (paper IJR 2025 + dataset site). Re-verify
> annotation quality after actual download — see OPEN items.

- **Reviewed by / date:** Buffy (agent) / 2026-09-09
- **Source / URL / license:** https://umfieldrobotics.github.io/ai4shipwrecks/ · download link on site · DeepBlue (Univ. of Michigan) record: https://deepblue.lib.umich.edu/data/concern/data_sets/8623hz41x · cite Sethuraman et al., IJR 44(3) 2025. NOAA-funded; published as an open benchmark. No restrictive custom license found on the landing pages (verify LICENSE file inside the download).
- **Access status:** public download (direct link from dataset homepage)
- **Modality:** ✅ real **side-scan sonar** (Klein-type AUV surveys, Thunder Bay National Marine Sanctuary, Lake Huron, 2022–2023)
- **Imaging geometry:** high-resolution SSS image tiles (PNG); waterfall/sonar imagery processed by survey software; per-image tiles, not raw ping streams
- **Format:** 286 PNG images + **pixel-wise segmentation masks** (binary: shipwreck vs. background/seafloor)
- **Navigation metadata:** site-level (28 wreck sites in a mapped sanctuary); per-image geo not confirmed — OPEN, check manifest after download
- **Label classes present (verbatim from data):** `shipwreck` (binary segmentation; background implicitly seafloor)
- **Approx instance counts per class:** 286 images over 28 distinct wrecks; one wreck region per image typically
- **Annotation quality spot-check:** not yet downloaded — **pending**. Paper reports manual pixel labels + benchmark; spot-check ≥20 images after download.
- **Overlap with our target classes (provisional, config-only):** direct match for `shipwreck`. No pipe/net class. Background = natural seafloor (useful negatives).
- **License compatible with SIH demo use?** Yes — open academic benchmark; attribution via the IJR citation. Confirm no extra terms in the archive.
- **Verdict:** **PRIMARY CANDIDATE** — real SSS, high-quality labels, exactly one provisional class, reputable source.
- **Next action:**
  1. Download; hash the archive; record in `datasets/`.
  2. Converter: **mask→YOLO-bbox** (connected components on binary mask → bounding boxes → `labels/*.txt`) feeding `ml/scripts/convert_dataset.py` interim layout.
  3. Decide `candidate_classes: [shipwreck]` (+ optional `seafloor` negatives as background-only images).
  4. OPEN #4 evidence: supports bbox-detection MVP (masks degrade to boxes losslessly); segmentation remains possible later.
