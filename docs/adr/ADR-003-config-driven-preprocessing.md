# ADR-003: Config-Driven Preprocessing (hash-tracked)

**Status:** Accepted (architecture v1.0, Section 7)

## Decision

Preprocessing is an ordered chain of registered ops selected by YAML config (`ml/configs/preprocessing/*.yaml`). Every run records: applied ops with params, config content hash, scale factors/padding, per-op timings. Model versions store their preprocessing config **hash**; inference loads the config by that hash from the model's record — not from the current default file.

## Rationale

- Eliminates train/serve skew structurally.
- Makes preprocessing experiments comparable by evidence (`ml/configs/preprocessing/experiments/` + evaluation runs).
- Deterministic: no random ops at inference (randomness belongs to train-only augmentation).

## Consequences

- Editing a historical config file does not silently change old models' behavior — hash mismatch warns loudly.
- No preprocessing op is applied "by default" unless the baseline config enables it; effectiveness must be measured.
