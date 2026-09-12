# ADR-004: Local-First Storage with Repository Interface

**Status:** Accepted (architecture v1.0, Section 15)

## Decision

MVP persistence is the local filesystem: immutable originals in `data/uploads/`, derived artifacts under `data/artifacts/`, JSON manifest documents under `data/db/<collection>.json` acting as the "database". All access goes through repository interfaces (`backend/app/persistence/repository.py`); MongoDB is an optional implementation behind `MONGODB_ENABLED` with identical document shapes.

## Rationale

- Zero infrastructure risk at demo time; debuggable with `ls` and any JSON viewer.
- `DATA_ROOT` env var relocates the entire runtime tree.
- Document shapes are the same for file and Mongo implementations, so switching is a repository swap, not a data-model change (OPEN decision #6).

## Consequences

- File repository assumes a single process (documented limitation); multi-user concurrency would require the Mongo path or a locking layer.
- Uploads are never mutated after write; sha256 recorded at ingest for evidence integrity.
