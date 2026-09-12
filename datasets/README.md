# Datasets

Layout (matches architecture Section 5; everything under `raw/` is immutable):

```
datasets/
├── raw/           # original downloaded archives — NEVER modified; record SHA-256 in review docs
├── interim/       # flat images/ + labels/*.txt per dataset (converter input layout)
├── processed/     # split YOLO-format datasets written by ml/scripts/convert_dataset.py
└── manifests/     # dataset manifest JSONs (hashes, class distribution, split info)
```

## Which datasets (see the shortlist)

The candidate review, licensing, and the recommended MVP pair live in
**[`docs/dataset-notes/SHORTLIST.md`](../docs/dataset-notes/SHORTLIST.md)** with per-dataset
reviews in the same folder.

Recommended primaries: **AI4Shipwrecks** (`shipwreck`) + **SubPipeMini2** (`pipe`) →
proposed initial `candidate_classes: [shipwreck, pipe]` (closes OPEN #1/#9 when downloaded
and inspected).

## Download checklist (per dataset)

1. Download into `raw/` keeping the original archive name; compute `shasum -a 256` and record
   it in the dataset's review doc (`docs/dataset-notes/<name>.md`).
2. Spot-check ≥20 images for annotation quality before converting anything.
3. Convert to `interim/<name>/{images,labels}`:
   - SubPipe: native YOLO labels — copy/organize, no conversion needed.
   - AI4Shipwrecks: masks → bboxes (connected components on binary mask → YOLO txt) — small
     one-off helper script, output stays in interim/.
   - SCTD (fallback): VOC XML → YOLO txt.
4. `PYTHONPATH=ml:. python -m ml.scripts.convert_dataset --interim datasets/interim/<name> \
   --out datasets/processed/<name> --name <name> --classes shipwreck,pipe \
   --group-sep <chunk-prefix> --save-manifest datasets/manifests/<name>.json`
   (use `--group-sep` for waterfall-chunk datasets to prevent train/test leakage).
5. Create the dataset config YAML in `ml/configs/datasets/` (see that folder's README) and run
   the train/evaluate CLIs — one EvaluationRun per dataset, plus one for any union, never merged
   into a single claim.

Nothing in this folder is committed except READMEs; the raw/interim/processed/manifests
contents are gitignored working data.
