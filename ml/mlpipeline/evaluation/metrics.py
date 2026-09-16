"""Evaluation metrics (Section 18.2): P/R/F1, AP50, per-class, confusion matrix.

IoU-matching of predicted vs ground-truth boxes at a fixed IoU threshold —
the standard detection evaluation, framework-free and fully testable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from mlpipeline.datatypes.detection import BBox
from mlpipeline.postprocessing.nms import iou as box_iou


# Row/column labels for the confusion matrix. Kept here so producers and the
# dashboard agree on the spelling.
BACKGROUND = "background"  # row: predictions that matched no ground-truth object
MISSED = "missed"  # column: ground-truth objects the model did not detect


@dataclass
class MatchedPair:
    gt_class: str
    pred_class: str
    iou: float
    score: float
    # False for a prediction that claimed no ground-truth box (a false positive).
    # Without this flag an unmatched prediction is indistinguishable from a
    # correct one once it is in the matrix, which is how false positives used to
    # be counted as correct classifications.
    matched: bool = True


@dataclass
class EvalCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    matches: list[MatchedPair] = field(default_factory=list)


def match_detections(
    gt: list[tuple[str, BBox]],  # (class_name, box)
    preds: list[tuple[str, BBox, float]],  # (class_name, box, score)
    iou_threshold: float = 0.5,
) -> EvalCounts:
    """Greedy matching: highest-score preds claim best-overlap unmatched GT of the same class."""
    counts = EvalCounts()
    used = set()
    ordered = sorted(preds, key=lambda p: p[2], reverse=True)
    for pcls, pbox, score in ordered:
        best, best_iou = None, 0.0
        for gi, (gcls, gbox) in enumerate(gt):
            if gi in used or gcls != pcls:
                continue
            ov = box_iou(pbox, gbox)
            if ov > best_iou:
                best, best_iou = gi, ov
        if best is not None and best_iou >= iou_threshold:
            used.add(best)
            counts.tp += 1
            counts.matches.append(MatchedPair(gt[best][0], pcls, best_iou, score))
        else:
            counts.fp += 1
            counts.matches.append(MatchedPair(pcls, pcls, best_iou, score, matched=False))
    counts.fn = len(gt) - len(used)
    return counts


def precision_recall_f1(counts: EvalCounts) -> tuple[float, float, float]:
    p = counts.tp / (counts.tp + counts.fp) if (counts.tp + counts.fp) else 0.0
    r = counts.tp / (counts.tp + counts.fn) if (counts.tp + counts.fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def average_precision_50(
    gt: list[tuple[str, BBox]],
    preds: list[tuple[str, BBox, float]],
    target_class: str,
    iou_threshold: float = 0.5,
) -> float:
    """AP for one class via the precision-recall envelope (11-pt-free, exact)."""
    cls_gt = [b for c, b in gt if c == target_class]
    cls_preds = sorted(
        [(b, s) for c, b, s in preds if c == target_class], key=lambda x: x[1], reverse=True
    )
    if not cls_gt:
        return 0.0 if cls_preds else float("nan")  # no GT & no preds -> undefined; treat via caller
    used = [False] * len(cls_gt)
    tps, fps = [], []
    for box, _score in cls_preds:
        best, best_iou = -1, 0.0
        for gi, gbox in enumerate(cls_gt):
            if used[gi]:
                continue
            ov = box_iou(box, gbox)
            if ov > best_iou:
                best, best_iou = gi, ov
        if best >= 0 and best_iou >= iou_threshold:
            used[best] = True
            tps.append(1)
            fps.append(0)
        else:
            tps.append(0)
            fps.append(1)
    # cumulative PR curve + AP by monotone envelope
    ctp = np_cumsum(tps)
    cfp = np_cumsum(fps)
    recalls = [ctp[i] / len(cls_gt) for i in range(len(tps))]
    precisions = [
        ctp[i] / (ctp[i] + cfp[i]) if (ctp[i] + cfp[i]) else 0.0 for i in range(len(tps))
    ]
    ap = 0.0
    prev_r = 0.0
    for r, p in zip(recalls, precisions):
        ap += p * (r - prev_r)
        prev_r = r
    return ap


def np_cumsum(xs: list[int]) -> list[int]:
    out, total = [], 0
    for x in xs:
        total += x
        out.append(total)
    return out


def confusion_matrix(
    matches: list[MatchedPair],
    class_names: list[str],
    gt_counts: dict[str, int] | None = None,
) -> dict[str, dict[str, int]]:
    """Detection confusion matrix.

    rows  = ground truth, plus a BACKGROUND row for predictions that matched no
            object (a false positive is never a "correct" cell).
    cols  = predicted class, plus a MISSED column for ground-truth objects the
            model failed to detect (populated when `gt_counts` is supplied).

    `gt_counts` holds per-class ground-truth instance counts, so the MISSED
    column can be derived as support - matched. Omit it (e.g. in unit tests) and
    the MISSED column is absent entirely rather than reported as a false zero.
    """
    names = list(class_names)
    cols = [*names, MISSED] if gt_counts is not None else list(names)
    matrix = {t: {p: 0 for p in cols} for t in [*names, BACKGROUND]}
    for m in matches:
        if m.matched:
            if m.gt_class in matrix and m.pred_class in cols:
                matrix[m.gt_class][m.pred_class] += 1
        elif m.pred_class in cols:
            matrix[BACKGROUND][m.pred_class] += 1
    if gt_counts is not None:
        for t in names:
            matched = sum(matrix[t][p] for p in names)
            matrix[t][MISSED] = max(0, int(gt_counts.get(t, 0)) - matched)
    return matrix
