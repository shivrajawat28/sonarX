# Demo-Day Runbook

Everything here was executed and verified on 2026-09-12 (see
`docs/ENGINEERING_REPORT.md` §0 and §18). Do not add claims the system cannot
back.

## T-minus checklist

```bash
# 0. from the repo root
.venv/Scripts/python scripts/verify_env.py          # python/torch/disk/env sanity

# 1. backend  (must be running BEFORE the frontend so the browser never 404s)
PYTHONPATH="ml;." .venv/Scripts/python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000

# 2. frontend (separate terminal)
cd frontend && npm run dev                          # http://localhost:5173
```

Confirm the model actually loaded (do this every time — it is the one failure
that kills the whole demo):

```bash
curl -s http://localhost:8000/api/v1/health
# expect: "model": {"loaded": true, "version": "drishti-ss_yolov8n_e30_final"}
```

The header badge in the UI shows the same thing: `model: drishti-ss_yolov8n_e30_final`.
If it says `degraded · no model`, **do not demo** — run `make test` and check the
backend log for the load error.

### Regenerate the survey fixture (only if `e2e/fixtures/geo_survey_demo.zip` is missing)

```bash
.venv/Scripts/python scripts/make_e2e_survey_fixture.py
```

## Demo flow (shortest reliable path, ~4 minutes)

Pick a real tile that reliably detects. **Use a pipeline tile** — it is the
strongest class (test F1 0.988):

```
datasets/processed/drishti-sss/test/images/pipe_1693569383.780_x3500.jpg
```

1. **Workbench** (`/`) — upload that tile. The header shows filename, dimensions
   and sha256.
2. **Run preview** → shows the applied ops (`resize_letterbox`), the config hash,
   and the warning that the source is letterboxed to the model input.
3. **Run detection** → the overlay appears with the box labelled
   `submarine_pipeline NN%`, plus the detection table (model confidence vs final
   confidence, filter status and the reason if any rule fired).
4. **Say the honest part out loud**: the box shown is the model's raw confidence,
   not a probability of correctness; filtering *annotates* (accepted / flagged /
   rejected) and never deletes; click a row to highlight its box and vice versa.
5. **Location**: the panel states *"Location unavailable — no navigation metadata
   provided"*. DRISHTI-SSS ships no navigation data, so coordinates are `null` on
   purpose — the system never invents them.
6. **Exports**: Download CSV then JSON. Open one and point at
   `model_version`, `preprocess_config_hash`, `geo_status`.
7. **Generate report** → download the HTML (and the PDF twin). Point out the bbox
   column and the "coordinates only where real metadata existed" notice.
8. **Survey (batch)** (`/survey`) — the geolocation path. Upload
   `e2e/fixtures/geo_survey_demo.zip`, then **Run batch detection**. Show the job
   progress, then the map with markers and `lat, lon ±Nm`. Be explicit that the
   nav track in this fixture is **synthetic test data** (the UI and archive both
   say so) and that it demonstrates the plumbing, not a real survey.
9. **Models** (`/models`) — the loaded model, the stored TEST-split metrics, the
   per-class table, the confusion matrix and the training provenance. Note that
   ghost_net is 100% synthetic and its 1.000 F1 is not a field capability.
10. **History** (`/history`) — persisted detections with status override.

## Numbers you may quote (measured, test split, held-out)

| Claim | Value |
|---|---|
| Overall (repo pipeline, IoU 0.5) | P 0.758 / R 0.586 / F1 0.661 / mAP50 0.663 |
| Ultralytics val on test | mAP50 0.699 / mAP50-95 0.518 |
| **Site-disjoint** subset (537 tiles) | P 0.733 / R 0.681 / mAP50 **0.702** / mAP50-95 **0.525** |
| submarine_pipeline | F1 0.988 (real, strongest) |
| shipwreck | F1 0.482 — recall 0.398, the weak real class |
| mine_cylinder | F1 0.389 (smallest real class) |
| ghost_net | F1 1.000 — **synthetic-on-synthetic, do not claim** |

Full detail and the earlier interim checkpoints: `docs/EVALUATION.md`.

## Failure recovery

| Symptom | Action |
|---|---|
| Badge says `degraded · no model` | `registry.json` has no `active` entry or the checkpoint is missing. Check the uvicorn log; `models/registry.json` paths are repo-relative now. |
| Port 8000 / 5173 busy | Kill the stale process (another instance may be running from a previous session). |
| `503 MODEL_UNAVAILABLE` on Run detection | Degraded mode by design — no model loaded. Not a crash. |
| Report shows "no PDF artifact" | Only affects reports generated before dual-format support; regenerate. |
| Inference slow | ~40–60 ms/image CPU plus overhead; don't promise live latency on stage. |
| Everything 404s in the browser | The frontend started before the backend, or the backend is on a different port. |

## Never do on stage

- Don't present stub / retired-model output as a real detection.
- Don't fabricate or patch coordinates to make the map look busy.
- Don't quote `model_confidence` as "% accurate" — it is a detector score.
- Don't claim ghost_net, cross-hardware generalization, or operational
  validation. The dataset is one academic-style corpus, already site-disjoint
  and honestly reported — that is the honest strength to lead with.
