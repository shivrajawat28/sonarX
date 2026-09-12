"""Step 3 acceptance tests: config validation + hash stability.

Gate (Section 25, step 3): invalid config rejected with clear error; hash stable.
"""
from pathlib import Path

import pytest

from mlpipeline.config import (
    DatasetConfig,
    DetectionConfig,
    FilterConfig,
    PreprocessConfig,
    config_sha256,
    load_config_file,
    save_config_yaml,
)
from mlpipeline.config.loader import ConfigError

REPO = Path(__file__).resolve().parents[3]


class TestLoadBundledConfigs:
    def test_baseline_preprocess_loads(self):
        cfg, h = load_config_file(REPO / "ml/configs/preprocessing/baseline_sonar.yaml", PreprocessConfig)
        assert cfg.name == "baseline_sonar"
        assert len(h) == 64
        enabled = [o.op for o in cfg.ops if o.enabled]
        assert enabled == ["normalize_intensity", "clahe", "resize_letterbox"]

    def test_filter_rules_load(self):
        cfg, _ = load_config_file(REPO / "ml/configs/filtering/rules.yaml", FilterConfig)
        names = [r.rule for r in cfg.enabled_rules if r.enabled]
        assert set(names) == {"min_size", "aspect_ratio", "edge_clip", "intensity_outlier"}
        assert cfg.policy.accept_threshold > cfg.policy.flag_threshold

    def test_detection_defaults_load(self):
        cfg, _ = load_config_file(REPO / "ml/configs/detection/serving_defaults.yaml", DetectionConfig)
        assert 0 < cfg.confidence_threshold < 1
        assert 0 < cfg.iou_threshold < 1

    def test_all_bundled_configs_are_valid_yaml_with_stable_hashes(self):
        """Hash stability: loading twice gives identical hashes (deterministic identity)."""
        for rel, cls in [
            ("ml/configs/preprocessing/baseline_sonar.yaml", PreprocessConfig),
            ("ml/configs/filtering/rules.yaml", FilterConfig),
            ("ml/configs/detection/serving_defaults.yaml", DetectionConfig),
        ]:
            _, h1 = load_config_file(REPO / rel, cls)
            _, h2 = load_config_file(REPO / rel, cls)
            assert h1 == h2, rel


class TestValidationErrors:
    def test_missing_file_clear_error(self):
        with pytest.raises(ConfigError, match="not found"):
            load_config_file("/nonexistent/config.yaml", PreprocessConfig)

    def test_invalid_yaml_clear_error(self, tmp_path: Path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("ops: [ {unclosed")
        with pytest.raises(ConfigError, match="invalid YAML"):
            load_config_file(bad, PreprocessConfig)

    def test_schema_violation_lists_field(self, tmp_path: Path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("name: x\nops: []")  # ops must have min_length=1
        with pytest.raises(ConfigError, match="ops"):
            load_config_file(bad, PreprocessConfig)

    def test_duplicate_ops_rejected(self, tmp_path: Path):
        bad = tmp_path / "dup.yaml"
        bad.write_text(
            "name: dup\nops:\n"
            "  - {op: clahe, params: {}}\n"
            "  - {op: clahe, params: {}}\n"
        )
        with pytest.raises(ConfigError, match="duplicate"):
            load_config_file(bad, PreprocessConfig)

    def test_split_fractions_must_sum_to_one(self, tmp_path: Path):
        bad = tmp_path / "ds.yaml"
        bad.write_text(
            "name: d\nroot: /tmp/ds\nformat: yolo\n"
            "split_fractions: {train: 0.5, val: 0.2, test: 0.1}\n"
        )
        with pytest.raises(ConfigError, match="sum to 1.0"):
            load_config_file(bad, DatasetConfig)

    def test_policy_thresholds_ordered(self, tmp_path: Path):
        bad = tmp_path / "f.yaml"
        bad.write_text(
            "name: f\nenabled_rules: []\n"
            "policy: {accept_threshold: 0.3, flag_threshold: 0.6}\n"
        )
        with pytest.raises(ConfigError, match="flag_threshold"):
            load_config_file(bad, FilterConfig)


class TestHashing:
    def test_hash_is_sha256_of_content(self):
        data = b"name: x\n"
        assert config_sha256(data) == __import__("hashlib").sha256(data).hexdigest()

    def test_different_content_different_hash(self):
        assert config_sha256(b"a") != config_sha256(b"b")

    def test_save_then_load_roundtrip(self, tmp_path: Path):
        cfg = FilterConfig(name="t", policy={"accept_threshold": 0.7, "flag_threshold": 0.2})
        out = tmp_path / "sub" / "rules.yaml"
        h = save_config_yaml(cfg, out)
        loaded, h2 = load_config_file(out, FilterConfig)
        assert loaded == cfg
        assert h == h2
