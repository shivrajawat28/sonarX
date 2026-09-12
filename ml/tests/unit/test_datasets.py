"""Step 5 tests: dataset tooling — manifests, validation, splitting.

Gate (Section 25, step 5): inspect/convert/validate/split produce manifest +
report + deterministic splits. Synthetic mini-dataset only (clearly test data).
"""
import shutil
from pathlib import Path

import numpy as np
import pytest

from mlpipeline.datasets import build_manifest, load_manifest, save_manifest, split_dataset, validate_dataset
from mlpipeline.datasets.manifest import sha256_file


@pytest.fixture
def mini_yolo_dataset(tmp_path: Path) -> Path:
    """Synthetic 8-image YOLO dataset: images/labels flat + interim layout.

    Classes: 0=alpha (even i, plus one beta box in img00), 1=beta (odd i).
    Two tiles share a survey prefix (leakage test).
    """
    import cv2

    root = tmp_path / "interim"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    rng = np.random.default_rng(3)
    for i in range(8):
        name = f"surveyA_tile{i:02d}" if i < 2 else f"img{i:02d}"
        img = rng.integers(0, 255, (64, 64), dtype=np.uint8)
        cv2.imwrite(str(root / "images" / f"{name}.png"), img)
        if i == 0:
            # two boxes: one alpha + one beta (odd total -> alpha 4 / beta 5)
            (root / "labels" / f"{name}.txt").write_text("0 0.5 0.5 0.2 0.2\n1 0.2 0.2 0.1 0.1\n")
        else:
            (root / "labels" / f"{name}.txt").write_text(f"{i % 2} 0.5 0.5 0.3 0.3\n")
    return root


class TestManifest:
    def test_build_counts_and_hashes(self, mini_yolo_dataset: Path):
        m = build_manifest(mini_yolo_dataset, name="mini", class_names=["alpha", "beta"])
        assert m.total_images == 8
        assert m.total_instances == 9
        assert m.stats["class_counts"] == {"alpha": 4, "beta": 5}
        # hash sanity: stable and matches direct computation
        e0 = m.images[0]
        assert e0.image.sha256 == sha256_file(mini_yolo_dataset / e0.image.path)

    def test_manifest_save_load_roundtrip(self, mini_yolo_dataset: Path, tmp_path: Path):
        m = build_manifest(mini_yolo_dataset, name="mini")
        p = save_manifest(m, tmp_path / "manifests" / "mini.json")
        assert load_manifest(p) == m


class TestValidate:
    def test_clean_dataset_passes(self, mini_yolo_dataset: Path):
        m = build_manifest(mini_yolo_dataset, name="mini", class_names=["alpha", "beta"])
        report = validate_dataset(m, check_images_decodable=True)
        assert report.ok, report.errors
        assert report.class_distribution == {"alpha": 4, "beta": 5}

    def test_corrupt_image_fails_loudly(self, mini_yolo_dataset: Path):
        (mini_yolo_dataset / "images" / "img03.png").write_bytes(b"not an image")
        m = build_manifest(mini_yolo_dataset, name="mini", class_names=["alpha", "beta"])
        report = validate_dataset(m)
        assert not report.ok
        assert any(i.code == "corrupt_image" for i in report.errors)

    def test_bbox_out_of_range_fails(self, mini_yolo_dataset: Path):
        (mini_yolo_dataset / "labels" / "img05.txt").write_text("0 1.5 0.5 0.2 0.2\n")
        m = build_manifest(mini_yolo_dataset, name="mini", class_names=["alpha", "beta"])
        report = validate_dataset(m)
        assert not report.ok
        assert any(i.code == "bbox_out_of_range" for i in report.errors)

    def test_unknown_class_id_fails(self, mini_yolo_dataset: Path):
        (mini_yolo_dataset / "labels" / "img07.txt").write_text("9 0.5 0.5 0.2 0.2\n")
        m = build_manifest(mini_yolo_dataset, name="mini", class_names=["alpha", "beta"])
        report = validate_dataset(m)
        assert not report.ok
        assert any(i.code == "unknown_class_id" for i in report.errors)

    def test_declared_class_with_zero_instances_warns(self, mini_yolo_dataset: Path):
        m = build_manifest(mini_yolo_dataset, name="mini", class_names=["alpha", "beta", "ghost"])
        report = validate_dataset(m, check_images_decodable=False)
        assert report.ok
        assert any(i.code == "empty_class" and "ghost" in i.detail for i in report.warnings)


class TestSplit:
    def _classes(self, paths: list[str]) -> dict[str, list[str]]:
        out = {}
        for p in paths:
            stem = Path(p).stem
            out[p] = ["alpha"] if int(stem[-2:]) % 2 == 0 else ["beta"]
        return out

    def test_deterministic_given_seed(self, mini_yolo_dataset: Path):
        paths = sorted(f"images/{p.name}" for p in (mini_yolo_dataset / "images").iterdir())
        a, ra = split_dataset(paths, self._classes(paths), {"train": 0.7, "val": 0.2, "test": 0.1}, seed=42)
        b, rb = split_dataset(paths, self._classes(paths), {"train": 0.7, "val": 0.2, "test": 0.1}, seed=42)
        assert a == b and ra.counts == rb.counts

    def test_different_seed_can_differ(self, mini_yolo_dataset: Path):
        paths = sorted(f"images/{p.name}" for p in (mini_yolo_dataset / "images").iterdir())
        a, _ = split_dataset(paths, self._classes(paths), {"train": 0.5, "val": 0.25, "test": 0.25}, seed=1)
        b, _ = split_dataset(paths, self._classes(paths), {"train": 0.5, "val": 0.25, "test": 0.25}, seed=2)
        assert a["train"] != b["train"]

    def test_grouping_keeps_survey_tiles_together(self, mini_yolo_dataset: Path):
        """The two surveyA tiles must land in the SAME split (leakage prevention)."""
        paths = sorted(f"images/{p.name}" for p in (mini_yolo_dataset / "images").iterdir())
        assignment, _ = split_dataset(
            paths, self._classes(paths), {"train": 0.7, "val": 0.2, "test": 0.1},
            seed=42, group_by_prefix_sep="_tile",
        )
        homes = {s for s, lst in assignment.items() for p in lst if "surveyA_tile" in p}
        assert len(homes) == 1

    def test_fractions_must_sum_to_one(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            split_dataset(["a.png"], None, {"train": 0.5, "val": 0.2, "test": 0.1})

    def test_write_yolo_split_materializes(self, mini_yolo_dataset: Path, tmp_path: Path):
        from mlpipeline.datasets.split import write_yolo_split

        # assignment paths are relative to interim images/ ("img00.png"-style via name)
        paths = sorted(f"images/{p.name}" for p in (mini_yolo_dataset / "images").iterdir())
        assignment, _ = split_dataset(
            paths, None, {"train": 0.75, "val": 0.25, "test": 0.0}, seed=7
        )
        out = write_yolo_split(mini_yolo_dataset, tmp_path / "processed", assignment)
        n_img = sum(len(list((out / "images" / s).iterdir())) for s in assignment)
        assert n_img == 8
        assert (out / "labels" / "train").is_dir()
