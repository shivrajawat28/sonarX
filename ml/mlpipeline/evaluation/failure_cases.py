"""Failure case export (Section 6.5): worst-N cases for the UI/report gallery.

A failure case = an image where detector output diverges most from ground truth
(missed GT with confident predictions, or confident FPs). Honest-limitations
feature per Section 22 risk #7.
"""
from __future__ import annotations

import json
from pathlib import Path

from mlpipeline.evaluation.metrics import EvalCounts


def select_failure_cases(
    per_image_counts: dict[str, EvalCounts],
    top_n: int = 10,
) -> list[dict]:
    """Rank images by (fn + fp) penalty and return the worst N with details."""

    def penalty(item: tuple[str, EvalCounts]) -> float:
        _, c = item
        # weighted: missed objects hurt more than extra proposals
        return c.fn * 2.0 + c.fp * 1.0

    ranked = sorted(per_image_counts.items(), key=penalty, reverse=True)
    out = []
    for rel, c in ranked[:top_n]:
        if c.fn + c.fp == 0:
            break  # stop at first fully-correct image
        out.append(
            {
                "image": rel,
                "missed": c.fn,
                "false_positives": c.fp,
                "true_positives": c.tp,
                "worst_scores": sorted(
                    ({"iou": m.iou, "score": m.score, "gt": m.gt_class, "pred": m.pred_class}
                     for m in c.matches),
                    key=lambda d: d["iou"],
                )[:5],
            }
        )
    return out


def save_failure_cases(cases: list[dict], out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "failures.json"
    out.write_text(json.dumps(cases, indent=2), encoding="utf-8")
    return out
