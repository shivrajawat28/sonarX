# Dataset Review: SeabedObjects-KLSG

- **Reviewed by / date:** Buffy (agent) / 2026-09-09
- **Source / URL / license:** https://github.com/huoguanying/SeabedObjects-Ship-and-Airplane-dataset · paper: "Underwater Object Classification in Sidescan Sonar Images Using Deep Transfer Learning and Semisynthetic Training Data" (2020). "Open for academic use" per README — no formal license file; verify.
- **Access status:** public GitHub (a Kaggle "Side-Scan Sonar Object Detection Challenge" also distributes it)
- **Modality:** ✅ real **side-scan sonar**
- **Imaging geometry:** single-object tile crops (classification-style), heterogeneous resolutions
- **Format:** raw images grouped by class folders; **no bounding boxes** (single object per image, implicitly full-frame) — usable for classification or by synthesizing full-frame boxes (weak labels)
- **Navigation metadata:** none
- **Label classes present (verbatim from data):** full set reported as **385 wreck, 36 drowning victim, 62 airplane, 129 mine, 578 seafloor**; the public GitHub ships the **ship + airplane** subset (385 + 62)
- **Approx instance counts per class:** see above; one object per image
- **Annotation quality spot-check:** pending; widely used in papers, labels are class-level folder names (low annotation risk, but image quality varies)
- **Overlap with our target classes:** `wreck` overlaps shipwreck (385 images — good volume); airplane/victim/mine droppable; 578 seafloor images are **valuable hard negatives** for the FP filter if the full set is obtainable
- **License compatible with SIH demo use?** Academic yes with citation; confirm terms before any redistribution
- **Verdict:** **FALLBACK** — classification-style crops; only becomes a detection source via weak full-frame boxes. Prefer over SCTD only if SCTD's mixed modality proves problematic, or for the seafloor negatives.
- **Next action:** low priority. If used, generate full-frame YOLO boxes, mark them `weak_label: true` in the manifest notes, and never mix weak+strong box supervision in one eval claim.
