"""Model info + metrics endpoints (Section 12 / ADR-010): metrics come ONLY from
stored EvaluationRun records — never recomputed ad hoc."""
from __future__ import annotations

from fastapi import APIRouter, Request

from backend.app.core.config import get_settings
from backend.app.core.errors import NotFoundError
from backend.app.services.inference_service import get_inference_service
from mlpipeline.registry import get_eval_store

router = APIRouter(prefix="/models", tags=["models"])


def _registry_at(settings):
    from mlpipeline.registry.models import ModelRegistry

    return ModelRegistry(settings.models_dir / "registry.json")


@router.get("")
def list_models(request: Request) -> dict:
    settings = get_settings()
    service = get_inference_service(settings)
    registry = _registry_at(settings)
    items = []
    for m in registry.list():
        items.append({
            "model_version": m.model_version,
            "architecture_family": m.architecture_family,
            "framework": m.framework,
            "class_map": m.class_map,
            "input_size": m.input_size,
            "status": m.status,
            "notes": m.notes,
            "created_at": m.created_at,
            "is_loaded": service.state.model_version == m.model_version,
        })
    return {"items": items, "loaded_version": service.state.model_version}


@router.get("/{model_version}")
def get_model(request: Request, model_version: str) -> dict:
    settings = get_settings()
    registry = _registry_at(settings)
    m = registry.get(model_version)
    if m is None:
        raise NotFoundError(f"model version '{model_version}' not in registry")
    return m.model_dump()


@router.get("/{model_version}/metrics")
def get_model_metrics(request: Request, model_version: str) -> dict:
    """Latest stored EvaluationRun + the model's training provenance.

    Everything here is read from recorded artifacts: the eval run's immutable
    record, its confusion-matrix artifact, and the registry entry. Nothing is
    recomputed, estimated, or defaulted at request time.
    """
    settings = get_settings()
    registry = _registry_at(settings)
    entry = registry.get(model_version)
    if entry is None:
        raise NotFoundError(f"model version '{model_version}' not in registry")
    store = get_eval_store_at(settings)
    run = store.latest_for_model(model_version)
    if run is None:
        raise NotFoundError(
            f"no evaluation runs recorded for '{model_version}' — metrics appear only "
            f"after a real evaluation (no fabricated numbers)"
        )
    payload = run.model_dump()
    payload["confusion"] = store.read_json_artifact(
        model_version, run.eval_run_id, "confusion.json"
    )
    payload["model"] = {
        "architecture_family": entry.architecture_family,
        "framework": entry.framework,
        "checkpoint_path": entry.checkpoint_path,
        "input_size": entry.input_size,
        "class_map": entry.class_map,
        "status": entry.status,
        "train_dataset_ref": entry.train_dataset_ref.model_dump() if entry.train_dataset_ref else None,
        "train_config": entry.train_config,
        "notes": entry.notes,
        "created_at": entry.created_at,
    }
    return payload


def get_eval_store_at(settings):
    from mlpipeline.registry.eval_runs import EvalRunStore

    return EvalRunStore(settings.models_dir / "eval")
