# ADR-001: Modular Monolith (single FastAPI process + SPA)

**Status:** Accepted (architecture v1.0, Sections 2 & 23)

## Decision

One deployable FastAPI backend imports the `mlpipeline` Python package as the ML brain; one React SPA frontend. Internal module boundaries (`ml/mlpipeline/*`) keep preprocessing, detection, filtering, geolocation, and reporting separately testable and replaceable, but nothing is split into separately deployed services.

## Rationale

- SIH student team; single-node demo; ML and API share release cadence.
- Microservice extraction remains possible because every stage boundary is an interface (`Detector`, `NavigationProvider`, repositories, job service).

## Consequences

- No network hops inside the pipeline; jobs are in-process (`BackgroundTasks`) with persisted records.
- The dependency edge **`ml` must not import FastAPI/HTTP** is enforced by `ml/tests/unit/test_boundaries.py`.

## Alternatives rejected

Microservices (ops cost ≫ value at this scale); notebook-centric pipeline (irreproducible, not demoable).
