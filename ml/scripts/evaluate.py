"""CLI: evaluate a registered model against a dataset manifest split (Step 7/8 gate).

Records a reproducible EvaluationRun (model version, dataset hash, split, seed,
preprocessing hash, metrics, per-class metrics). Plumbing uses the stub detector
for tests; real quality numbers require a real registered model + real dataset.

    PYTHONPATH=ml:. python -m ml.scripts.evaluate \
        --model stub-exp-001 --manifest datasets/manifests/my_ds.json --split test
"""
from __future__ import annotations

import argparse
import sys

from mlpipeline.evaluation.evaluate import evaluate_model
from mlpipeline.registry.eval_runs import get_eval_store


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="registered model version id")
    ap.add_argument("--manifest", required=True, help="dataset manifest JSON produced by inspect/convert tooling")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--filter-config", default="",
                    help="filtering rules YAML — record filter-ON metrics as a separate eval run")
    ap.add_argument("--notes", default="", help="free-text note stored with the run")
    args = ap.parse_args(argv)

    try:
        run = evaluate_model(
            args.model,
            args.manifest,
            split=args.split,
            filter_config_path=args.filter_config or None,
            notes=args.notes or "CLI evaluation",
        )
    except Exception as e:  # noqa: BLE001 — CLI boundary
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    store = get_eval_store()
    m = run.metrics
    print(f"eval run {run.eval_run_id} saved to {store.root}")
    print(f"  model={run.model_version} split={run.split} seed={run.split_seed}")
    print(f"  precision={m.precision:.3f} recall={m.recall:.3f} f1={m.f1:.3f}")
    for cname, pm in run.per_class.items():
        print(f"  [{cname}] P={pm.precision:.3f} R={pm.recall:.3f} F1={pm.f1:.3f} "
              f"AP50={pm.ap50 if pm.ap50 is not None else float('nan'):.3f} support={pm.support}")
    print("NOTE: numbers reflect the model+dataset above only — no generalization claims.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
