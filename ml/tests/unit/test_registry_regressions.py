"""Regression tests (ML layer) for defects fixed in the final release audit.

Covers:
- `resolve_repo_path`: registry entries must survive a repo move instead of
  hard-failing on a stale absolute path (absolute Windows paths baked in during
  registration degrade serving to MODEL_UNAVAILABLE).
- `confusion_matrix` artifacts must be real JSON — the eval runner previously
  wrote a Python repr (`.__str__()`), which `json.loads` rejects and which made
  the artifact unusable for the evaluation dashboard.
"""
from __future__ import annotations

import json
from pathlib import Path

from mlpipeline.evaluation.metrics import MatchedPair, confusion_matrix
from mlpipeline.registry.models import repo_root, resolve_repo_path


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

    def test_repr_would_not_parse_as_json(self):
        """Guards the actual defect: the previous encoding was not JSON."""
        matrix = confusion_matrix([MatchedPair("a", "a", 0.9, 0.5)], ["a"])
        import pytest

        with pytest.raises(json.JSONDecodeError):
            json.loads(str(matrix))
