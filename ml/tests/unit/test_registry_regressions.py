"""Regression tests (ML layer) for defects fixed in the final release audit.

Covers:
- `resolve_repo_path`: registry entries must survive a repo move instead of
  hard-failing on a stale absolute path (absolute Windows paths baked in during
  registration degrade serving to MODEL_UNAVAILABLE).- `confusion_matrix` artifacts must be real JSON — the eval runner previously
  wrote a Python repr (`.__str__()`), which `json.loads` rejects and which made
the artifact unusable for the evaluation dashboard.
- `portable_ref_path`: EvaluationRun records are Git-tracked, so their
  `dataset_ref.path` must be repo-relative — absolute paths leak the registering
  machine's layout and dangle after a move, the same defect class as the registry.
"""
from __future__ import annotations

import json
from pathlib import Path

from mlpipeline.evaluation.evaluate import portable_ref_path
from mlpipeline.evaluation.metrics import MatchedPair, confusion_matrix
from mlpipeline.registry.models import repo_root, resolve_repo_path


class TestPortableRefPath:
    def test_repo_relative_path_is_left_alone(self):
        rel = "datasets/manifests/drishti-sss.json"
        assert portable_ref_path(rel) == rel

    def test_absolute_path_inside_repo_becomes_relative(self):
        abs_path = repo_root() / "datasets" / "manifests" / "drishti-sss.json"
        assert portable_ref_path(abs_path) == "datasets/manifests/drishti-sss.json"

    def test_stale_machine_path_is_stripped_to_repo_relative(self):
        """The exact defect: a record written on another machine's checkout."""
        stale = (
            "C:/Users/someone/OneDrive/Desktop/SONAR/SONAR/"
            "datasets/manifests/drishti-sss.json"
        )
        out = portable_ref_path(stale)
        assert out == "datasets/manifests/drishti-sss.json"
        assert "someone" not in out

    def test_path_outside_repo_is_not_silently_rewritten(self, tmp_path: Path):
        outside = tmp_path / "elsewhere" / "manifest.json"
        assert portable_ref_path(outside) == str(outside)


class TestResolveRepoPath:
    def test_relative_path_resolves_against_repo_root(self):
        rel = "ml/configs/preprocessing/drishti_preprocessed.yaml"
        resolved = resolve_repo_path(rel)
        assert resolved.is_file()
        assert resolved == (repo_root() / rel).resolve() or resolved == repo_root() / rel

    def test_existing_absolute_path_is_used_as_is(self, tmp_path: Path):
        f = tmp_path / "weights.pt"
        f.write_bytes(b"x")
        assert resolve_repo_path(f) == f

    def test_stale_absolute_path_falls_back_to_repo_relative_suffix(self):
        """A path from another checkout resolves to the same file in this repo."""
        stale = (
            "C:/SomeOtherUser/OneDrive/Desktop/SONAR/SONAR/"
            "ml/configs/preprocessing/drishti_preprocessed.yaml"
        )
        resolved = resolve_repo_path(stale)
        assert resolved.is_file(), f"did not recover a portable path from {stale}"
        assert resolved.name == "drishti_preprocessed.yaml"
        assert "SomeOtherUser" not in str(resolved)

    def test_unrecoverable_absolute_path_is_returned_unchanged(self, tmp_path: Path):
        """Never invent a location: an unresolvable path stays as written so the
        detector raises a clear `checkpoint missing` error."""
        missing = tmp_path / "nope" / "best.pt"
        assert resolve_repo_path(missing) == missing


class TestConfusionMatrixArtifact:
    def test_serializes_to_valid_json(self):
        matches = [
            MatchedPair("shipwreck", "shipwreck", 0.91, 0.8),
            MatchedPair("shipwreck", "mine_cylinder", 0.12, 0.4),
            MatchedPair("mine_cylinder", "mine_cylinder", 0.77, 0.6),
        ]
        matrix = confusion_matrix(matches, ["shipwreck", "mine_cylinder"])
        text = json.dumps(matrix, indent=2)
        parsed = json.loads(text)  # the old `str(dict)` output raised here
        assert parsed == matrix
        assert parsed["shipwreck"]["shipwreck"] == 1
        assert parsed["shipwreck"]["mine_cylinder"] == 1
        assert parsed["mine_cylinder"]["mine_cylinder"] == 1
        assert parsed["mine_cylinder"]["shipwreck"] == 0

    def test_false_positive_never_counts_as_a_correct_cell(self):
        """Regression: unmatched predictions used to land on the diagonal.

        `match_detections` records a false positive as (pred_class, pred_class),
        which the matrix read as a correct classification — inflating the
        diagonal past the class's true-positive count. An unmatched prediction
        must instead be attributed to the BACKGROUND row.
        """
        from mlpipeline.datatypes.detection import BBox
        from mlpipeline.evaluation.metrics import BACKGROUND, MISSED, match_detections

        gt = [("shipwreck", BBox(x=0, y=0, w=10, h=10))]
        # One true positive, one false positive of the same class, one of another.
        preds = [
            ("shipwreck", BBox(x=0, y=0, w=10, h=10), 0.9),
            ("shipwreck", BBox(x=100, y=100, w=10, h=10), 0.8),
            ("mine_cylinder", BBox(x=200, y=200, w=10, h=10), 0.7),
        ]
        counts = match_detections(gt, preds)
        assert (counts.tp, counts.fp) == (1, 2)

        matrix = confusion_matrix(
            counts.matches, ["shipwreck", "mine_cylinder"], gt_counts={"shipwreck": 1, "mine_cylinder": 0}
        )
        # Only the matched pair is a correct shipwreck cell.
        assert matrix["shipwreck"]["shipwreck"] == 1
        assert matrix[BACKGROUND]["shipwreck"] == 1
        assert matrix[BACKGROUND]["mine_cylinder"] == 1
        # A class row sums to its support (nothing missed here).
        assert matrix["shipwreck"][MISSED] == 0
        assert sum(matrix["shipwreck"].values()) == 1

    def test_missed_column_equals_support_minus_matched(self):
        """An undetected object must appear as MISSED, not vanish."""
        from mlpipeline.datatypes.detection import BBox
        from mlpipeline.evaluation.metrics import MISSED, match_detections

        gt = [
            ("shipwreck", BBox(x=0, y=0, w=10, h=10)),
            ("shipwreck", BBox(x=50, y=50, w=10, h=10)),
        ]
        preds = [("shipwreck", BBox(x=0, y=0, w=10, h=10), 0.9)]
        counts = match_detections(gt, preds)
        matrix = confusion_matrix(counts.matches, ["shipwreck"], gt_counts={"shipwreck": 2})
        assert matrix["shipwreck"][MISSED] == 1
        assert sum(matrix["shipwreck"].values()) == 2

    def test_no_missed_column_without_ground_truth_counts(self):
        """Never fabricate a zero: without support data the column is absent."""
        from mlpipeline.evaluation.metrics import MISSED

        matrix = confusion_matrix([MatchedPair("a", "a", 0.9, 0.5)], ["a"])
        assert MISSED not in matrix["a"]

    def test_repr_would_not_parse_as_json(self):
        """Guards the actual defect: the previous encoding was not JSON."""
        matrix = confusion_matrix([MatchedPair("a", "a", 0.9, 0.5)], ["a"])
        import pytest

        with pytest.raises(json.JSONDecodeError):
            json.loads(str(matrix))
