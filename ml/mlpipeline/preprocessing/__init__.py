"""Preprocessing subsystem: registry + protocol + engine.

Contract (Section 7 / ADR-003):
- Ops are selected by config; nothing runs unless enabled.
- Every run records applied ops (with the params actually used), per-op timings,
  letterbox scale factors, and the config content hash.
- Deterministic: no random ops at inference time.
"""
from mlpipeline.preprocessing.base import PreprocessOp, OpContext
from mlpipeline.preprocessing.registry import (
    PreprocessingRegistry,
    register_op,
    get_op,
    known_op_names,
)
from mlpipeline.preprocessing.pipeline import PreprocessingEngine, PreprocessingError

# Importing the preprocessing package registers all built-in ops so that configs
# referencing them validate without extra imports.
import mlpipeline.preprocessing.ops as _builtin_ops  # noqa: F401

__all__ = [
    "PreprocessOp",
    "OpContext",
    "PreprocessingRegistry",
    "register_op",
    "get_op",
    "known_op_names",
    "PreprocessingEngine",
    "PreprocessingError",
]
