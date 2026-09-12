# ADR-006: model_confidence vs final_confidence / filtering_status

**Status:** Accepted (architecture v1.0, Section 9)

## Decision

Every detection carries **`model_confidence`** (the raw detector score, untouched — *not* a calibrated probability of correctness) and **`final_confidence`** (model confidence minus explainable rule penalties). The filter assigns `filtering_status` ∈ {`accepted`, `flagged`, `rejected`} plus `filter_reasons[]`. Detections are **never deleted** by filtering; rejected items remain in results (the UI hides them behind a toggle by default).

## Rationale

- Honest semantics: the system's verdict is separable from the model's opinion, auditable, and tunable per config.
- Judges/analysts can inspect *why* anything was flagged or rejected; analyst override (`PATCH /detections/{id}`) stays possible.

## Consequences

- Filter config hash is recorded in every result, exactly like the preprocessing hash.
- Thresholds are applied exactly once, in post-processing, and recorded; the filter never applies hidden cutoffs.
