"""PreprocessingEngine: config -> ordered op chain -> ProcessedSonarImage.

Records (Section 7.1): applied ops with actual params, per-op timings, letterbox
scale factors, config hash. Deterministic given the same config + input.
"""
from __future__ import annotations

import time

import numpy as np
from pydantic import ValidationError

from mlpipeline.config.schemas import PreprocessConfig
from mlpipeline.datatypes.image import AppliedOp, ProcessedSonarImage, ScaleFactors
from mlpipeline.preprocessing.base import OpContext
from mlpipeline.preprocessing.registry import get_op


class PreprocessingError(Exception):
    """Raised when an op fails; message names the failing op (API maps to 500)."""


class PreprocessingEngine:
    """Runs a validated PreprocessConfig against images. Stateless and reusable."""

    def __init__(self) -> None:
        # Built-ins register on package import (mlpipeline.preprocessing.__init__);
        # kept here as belt-and-braces for direct engine construction.
        import mlpipeline.preprocessing.ops  # noqa: F401

    def run(
        self,
        image: np.ndarray,
        config: PreprocessConfig,
        config_hash: str,
        source_image_id: str | None = None,
    ) -> ProcessedSonarImage:
        """Apply the configured chain. Raises PreprocessingError on op failure."""
        if image.ndim not in (2, 3):
            raise PreprocessingError(f"expected 2D or 3D image, got shape {image.shape}")

        ctx = OpContext(source_image_id=source_image_id)
        applied: list[AppliedOp] = []
        work = image

        for idx, spec in enumerate(config.ops):
            if not spec.enabled:
                continue
            try:
                op_cls = get_op(spec.op)
            except KeyError as e:
                raise PreprocessingError(str(e)) from e

            op = op_cls()
            t0 = time.perf_counter()
            try:
                work = op.apply(work, spec.params, ctx)
            except Exception as e:
                raise PreprocessingError(
                    f"preprocessing op '{spec.op}' failed at stage {idx}: {e}"
                ) from e
            duration_ms = (time.perf_counter() - t0) * 1000.0
            applied.append(AppliedOp(op=spec.op, params=spec.params, duration_ms=round(duration_ms, 3)))

        if work.ndim == 3 and work.shape[2] == 1:
            work = work[:, :, 0]

        sf = ScaleFactors(
            scale_x=float(ctx.extras.get("scale_x", 1.0)),
            scale_y=float(ctx.extras.get("scale_y", 1.0)),
            pad_x=float(ctx.extras.get("pad_x", 0.0)),
            pad_y=float(ctx.extras.get("pad_y", 0.0)),
        )

        return ProcessedSonarImage(
            image_id=source_image_id or "unknown",
            width=int(work.shape[1]),
            height=int(work.shape[0]),
            source_width=int(image.shape[1]),
            source_height=int(image.shape[0]),
            scale_factors=sf,
            applied_ops=applied,
            config_hash=config_hash,
            config_name=config.name,
            warnings=list(ctx.warnings),
        )

    def run_with_array(
        self,
        image: np.ndarray,
        config: PreprocessConfig,
        config_hash: str,
        source_image_id: str | None = None,
    ) -> tuple[ProcessedSonarImage, np.ndarray]:
        """Same as run() but also returns the processed array (engine-internal use)."""
        ctx_holder = _ArrayCapture()
        result = self.run(image, config, config_hash, source_image_id)
        processed = ctx_holder.capture or _rerun_for_array(image, config, config_hash, source_image_id, ctx_holder)
        return result, processed


class _ArrayCapture:
    capture: np.ndarray | None = None


def _rerun_for_array(
    image: np.ndarray,
    config: PreprocessConfig,
    config_hash: str,
    source_image_id: str | None,
    holder: _ArrayCapture,
) -> np.ndarray:
    """Internal: re-run the chain capturing the array (keeps run() signature clean)."""
    ctx = OpContext(source_image_id=source_image_id)
    work = image
    for idx, spec in enumerate(config.ops):
        if not spec.enabled:
            continue
        op = get_op(spec.op)()
        work = op.apply(work, spec.params, ctx)
    if work.ndim == 3 and work.shape[2] == 1:
        work = work[:, :, 0]
    holder.capture = work
    return work


def validate_config(config: PreprocessConfig) -> None:
    """Ensure all op names exist in the registry (fail before running anything)."""
    import mlpipeline.preprocessing.ops  # noqa: F401

    for spec in config.ops:
        get_op(spec.op)  # raises KeyError with known-op list

    # Also validate params cheaply on a tiny probe image so bad params fail fast.
    probe = np.zeros((8, 8), dtype=np.uint8)
    ctx = OpContext()
    for spec in config.ops:
        if not spec.enabled:
            continue
        op = get_op(spec.op)()
        try:
            op.apply(probe, spec.params, ctx)
        except ValidationError:
            raise
        except Exception as e:
            raise PreprocessingError(f"op '{spec.op}' rejected params {spec.params}: {e}") from e
