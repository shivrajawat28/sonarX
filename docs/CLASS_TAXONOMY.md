# Class Taxonomy — Marine Debris Sonar AI

Generated: 2026-09-11

## Final Class List

Based on dataset audit (docs/DATASET_AUDIT.md), the following classes are used in the trained model:

| ID | Class | Definition | Real/Synthetic | Source | Training Examples | Decision |
|---|---|---|---|---|---|---|
| 0 | `submarine_pipeline` | Exposed or partially buried submarine pipelines/cables on seabed | **Real** | SubPipeMini2 (DRISHTI-SSS `pipe` prefix) | 1,000 instances / 1,000 label files (train) | ✅ Primary class |
| 1 | `shipwreck` | Sunken vessels, wreckage debris, ship structures | **Real** | AI4Shipwrecks (wreckA) + Roboflow SSS (wreckR) | 1,554 instances / ~900 label files (train) | ✅ Primary class |
| 2 | `ghost_net` | Abandoned/lost fishing nets, derelict fishing gear | **⚠️ 100% Synthetic** | Procedural acoustic generator | 900 instances / 900 label files (train) | ⚠️ Experimental — synthetic only |
| 3 | `mine_cylinder` | Cylindrical naval mines, man-made sonar objects | **Real** | Kaggle MILCO contacts (DRISHTI-SSS `mine` prefix) | 843 instances / ~575 label files (train) | ✅ Primary class |

## Excluded Classes

| Class | Reason |
|---|---|
| `crab_pot` | Zero examples in DRISHTI-SSS release (excluded upstream). No other available dataset provides annotated crab pot SSS data. |

## Class Mapping

YOLO training IDs (0–3) map to these class names (verified against
`datasets/processed/drishti-sss/data.yaml` and raw-label class counts):

```
0: submarine_pipeline
1: shipwreck
2: ghost_net
3: mine_cylinder
```

Note: DRISHTI-SSS used IDs 1–4 (with 0=crab_pot excluded). Our training remaps
to 0–3 since we drop crab_pot entirely. Verified train-split instance counts:
`{0: 1000, 1: 1554, 2: 900, 3: 843}` (raw dataset uses ids {1: 1000, 2: 1554,
3: 900, 4: 843} — i.e. raw id N maps to processed id N-1).

## Provenance Details

### submarine_pipeline (Real)
- **Primary source:** SubPipe / SubPipeMini2 (Álvarez-Tuñón et al., OceanScan-MST)
- **License:** CC-BY-4.0
- **Data nature:** Real side-scan sonar survey strips, tiled into 640×640 patches
- **Known limitations:** Single sonar system; generalization to other SSS hardware unproven

### shipwreck (Real)
- **Source A:** AI4Shipwrecks — Sethuraman et al., UM Field Robotics / NOAA Thunder Bay (wreckA prefix)
  - License: CC-BY-4.0
  - Pixel masks converted to bounding boxes
- **Source B:** Side Scan Sonar (Ship, Plane) — Dae Hyeok Lee, Roboflow Universe (wreckR prefix)
  - License: CC-BY-4.0
- **Known limitations:** Test set uses 50%-overlap re-tiling (harder, partial views). Site-disjointness WAS audited and holds — zero train/val↔test site overlap for both sources (`scripts/audit_site_leakage.py`; see `docs/DATASET_AUDIT.md` § Known Caveats). So these are unseen-site numbers, not leaked ones.

### mine_cylinder (Real)
- **Source:** Sonar Imaging Mine Detection — MILCO contacts (Kaggle)
  - License: CC-BY-SA-4.0
- **Data nature:** Real mine contacts on sonar
- **Known limitations:** Smallest real class (~225 training images). Limited diversity.

### ghost_net (Synthetic)
- **Source:** Procedural acoustic generator (`ml/scripts/build_synthetic_data.py`)
  - Background canvases from Roboflow SSS set (CC-BY-4.0)
  - Procedurally modelled objects composited and filtered
  - License: CC-BY-SA-4.0
- **Known limitations:** **100% SYNTHETIC.** No real ghost-net-in-SSS annotated dataset exists publicly. Microsoft AI for Good / WWF effort had 412 real segments and called it a feasibility study. Metrics for this class are synthetic-on-synthetic and NOT representative of real-world performance. Class is retained for pipeline completeness but must be clearly marked as experimental in all outputs.

## Confidence in Each Class

| Class | Confidence Level | Rationale |
|---|---|---|
| submarine_pipeline | **High** | 1,000 real training instances from dedicated SSS pipeline inspection dataset |
| shipwreck | **Medium-High** | 1,554 real instances from two sources; site-disjointness audited (no leakage), but recall on the hard re-tiled test split is only 0.398 |
| mine_cylinder | **Medium** | 843 real instances, smallest real class, limited diversity |
| ghost_net | **Low (Experimental)** | 100% synthetic; no real validation possible |

## Recommendations for SIH Demo

1. **Lead with submarine_pipeline and shipwreck** — these have the strongest real data backing.
2. **Show mine_cylinder as a secondary capability** — real but limited data.
3. **Be transparent about ghost_net** — synthetic only, mark as "experimental" in UI.
4. **Do NOT claim generalization** to sonar systems not seen in training.
5. **Highlight the multi-source nature** of the training data as a strength.
