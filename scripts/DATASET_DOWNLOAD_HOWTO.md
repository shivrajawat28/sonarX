# Dataset download + verification — how to run

Run these from the repo root:
`C:\Users\kr034\OneDrive\Desktop\SONAR\SONAR`

No Git/Git-Bash needed — everything uses the machine's Python 3.14
(`huggingface_hub` is already installed there).

## 1) Download (resumable — safe to re-run)

```powershell
& "C:\Users\kr034\AppData\Local\Programs\Python\Python314\python.exe" scripts\download_datasets.py
```

What it does:

| Dataset | Source | Into | Size |
|---|---|---|---|
| drishti-sss | HF `rehan9599/drishti-sss` (resume; fills missing `val/` + `train/labels`) | `datasets/raw/drishti-sss/` | ~1.3 GB more |
| subpipe | Zenodo 12666132 → `SubPipeMini2.zip` | `datasets/raw/subpipe/` | 4.9 GB |
| ai4shipwrecks | U-M Deep Blue → `AI4Shipwrecks.zip` | `datasets/raw/ai4shipwrecks/` | 1.13 GB |
| ghostvision | Zenodo 20056679 (open; HF copy is gated) → `GhostVision_DatasetAndModels.zip` | `datasets/raw/ghostvision/` | 857 MB |
| kaggle-sss | Public GitHub SeabedObjects zip (stand-in per your choice) | `datasets/raw/kaggle-sss/` | ~100 MB |

- Downloads are atomic (`.part` → rename) and resume after interruption.
- Archives are kept **as-is** — nothing is extracted, renamed or preprocessed.
- Zenodo files are MD5-verified against official checksums.
- Expected new downloads total ≈ 7.7 GB. Rough times on a normal connection:
  subpipe 15–40 min, ai4shipwrecks 5–15 min, ghostvision 3–10 min.

Download one dataset only (optional):

```powershell
& "C:\Users\kr034\AppData\Local\Programs\Python\Python314\python.exe" scripts\download_datasets.py subpipe
```

Valid names: `drishti-sss`, `subpipe`, `ai4shipwrecks`, `ghostvision`, `kaggle-sss`.

## 2) Verify + auto-write manifests

```powershell
& "C:\Users\kr034\AppData\Local\Programs\Python\Python314\python.exe" scripts\verify_datasets.py
```

- Counts images/labels on disk, validates every zip, prints DRISHTI split totals
  vs the documented 5,205/5,205, and writes `datasets/manifests/<name>.md`
  plus `datasets/manifests/_audit-summary.json`.
- Read-only: nothing under `datasets/raw/` is modified.

## 3) Afterwards

Paste back the terminal output of both scripts — I'll finish the audit and the
final report table.
