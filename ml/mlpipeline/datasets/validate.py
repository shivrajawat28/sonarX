"""Dataset validation: fail loudly, never silently repair (Section 6.2).

Checks: missing files, corrupt images, label/bbox sanity, class distribution,
train/val/test leakage (duplicate content hashes across splits).
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from pydantic import BaseModel, Field

from mlpipeline.datasets.manifest import DatasetManifest


class ValidationIssue(BaseModel):
    severity: str  # "error" | "warning"
    code: str
    detail: str


class ValidationReport(BaseModel):
    dataset_name: str
    ok: bool
    errors: list[ValidationIssue] = Field(default_factory=list)
    warnings: list[ValidationIssue] = Field(default_factory=list)
    class_distribution: dict[str, int] = Field(default_factory=dict)

    def fail(self, code: str, detail: str) -> None:
        self.errors.append(ValidationIssue(severity="error", code=code, detail=detail))
        self.ok = False

    def warn(self, code: str, detail: str) -> None:
        self.warnings.append(ValidationIssue(severity="warning", code=code, detail=detail))


def validate_dataset(
    manifest: DatasetManifest,
    root: Path | None = None,
    check_images_decodable: bool = True,
) -> ValidationReport:
    """Validate a dataset against its manifest. Errors make ok=False."""
    import cv2
    import numpy as np

    root = Path(root or manifest.root)
    report = ValidationReport(dataset_name=manifest.name, ok=True)
    dist: Counter[str] = Counter()

    class_map = {i: n for i, n in enumerate(manifest.class_names)}
    hash_to_splits: dict[str, set[str]] = {}

    for entry in manifest.images:
        img_path = root / entry.image.path
        split = entry.image.path.split("/")[1] if "/" in entry.image.path else "?"

        # file existence + hash match
        if not img_path.is_file():
            report.fail("missing_file", entry.image.path)
            continue
        from mlpipeline.datasets.manifest import sha256_file

        if sha256_file(img_path) != entry.image.sha256:
            report.fail("hash_mismatch", entry.image.path)

        hash_to_splits.setdefault(entry.image.sha256, set()).add(split)

        # label presence
        if entry.label is None:
            report.warn("no_label", f"{entry.image.path} has no label file (background image)")
        else:
            label_path = root / entry.label.path
            if not label_path.is_file():
                report.fail("missing_label_file", entry.label.path)
            else:
                # bbox sanity: normalized coords in [0,1], w/h > 0
                for lineno, line in enumerate(label_path.read_text().splitlines(), 1):
                    parts = line.split()
                    if not parts:
                        continue
                    if len(parts) < 5:
                        report.fail("malformed_label", f"{entry.label.path}:{lineno}")
                        continue
                    try:
                        cid = int(parts[0])
                        cx, cy, w, h = (float(x) for x in parts[1:5])
                    except ValueError:
                        report.fail("malformed_label", f"{entry.label.path}:{lineno}")
                        continue
                    if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                        report.fail(
                            "bbox_out_of_range",
                            f"{entry.label.path}:{lineno} cxcywh=({cx},{cy},{w},{h})",
                        )
                    cname = class_map.get(cid, f"<id {cid} unmapped>")
                    dist[cname] += 1
                    if cid not in class_map:
                        report.fail("unknown_class_id", f"{entry.label.path}:{lineno} id={cid}")

        # decodability (spot-check cost is fine at SIH dataset sizes)
        if check_images_decodable and img_path.is_file():
            data = np.fromfile(img_path, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
            if img is None:
                report.fail("corrupt_image", entry.image.path)

    # leakage: identical content hashes across different splits
    for h, splits in hash_to_splits.items():
        if len(splits) > 1:
            report.fail("split_leakage", f"identical image content in splits {sorted(splits)} (sha {h[:12]}…)")

    # class distribution
    report.class_distribution = dict(sorted(dist.items()))
    if manifest.class_names:
        missing = [n for n in manifest.class_names if n not in dist]
        if missing:
            report.warn("empty_class", f"classes declared but with zero instances: {missing}")

    return report
