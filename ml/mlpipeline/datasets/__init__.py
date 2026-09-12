"""Dataset subsystem: ingestion, validation, splitting, manifests (Phase 3 / Step 5).

Dataset-first rule: the final class list (OPEN decision #1) emerges from running
these tools against real data — nothing else in the app may hardcode classes.
"""
from mlpipeline.datasets.manifest import DatasetManifest, build_manifest, save_manifest, load_manifest
from mlpipeline.datasets.validate import validate_dataset, ValidationReport
from mlpipeline.datasets.split import split_dataset, SplitResult, write_yolo_split

__all__ = [
    "DatasetManifest",
    "build_manifest",
    "save_manifest",
    "load_manifest",
    "validate_dataset",
    "ValidationReport",
    "split_dataset",
    "SplitResult",
    "write_yolo_split",
]
