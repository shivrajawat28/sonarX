# ADR-005: Sync Single-Image, In-Process Jobs for Batch/Reports

**Status:** Accepted (architecture v1.0, Section 11.3)

## Decision

Single-image inference, preprocess preview, and exports run **synchronously**. Survey batch inference and report generation run as **background jobs**: persisted job records (`pending → running → succeeded/failed` with progress and errors) polled via `GET /api/v1/jobs/{job_id}`. No Redis, Celery, Kafka, or external brokers.

## Rationale

- Matches the real latency profile (single image ≈ sub-second to seconds on CPU; surveys can be hundreds of images).
- Persisted job JSON survives page refreshes; single-process assumption is explicit and documented.
- The `JobService` interface is the future seam for a real queue if scale ever demands it (OPEN decision #7).

## Consequences

- Jobs die with the process (documented); restart re-runs surveys explicitly.
- `ASYNC_SURVEY_THRESHOLD` (env) decides upload→job vs sync handling of survey archives.
