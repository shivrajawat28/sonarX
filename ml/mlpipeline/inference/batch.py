"""Survey batch runner: loops the SAME engine per image (Section 8.3).

This is a loop, not a second inference path. Progress callbacks feed the
backend's job manager.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from mlpipeline.datatypes.navigation import NavigationTrack
from mlpipeline.datatypes.results import InferenceResult
from mlpipeline.inference.engine import SonarInferenceEngine
from mlpipeline.io.image_reader import SUPPORTED_EXTENSIONS, ImageReadError


def run_survey_batch(
    engine: SonarInferenceEngine,
    image_paths: list[str | Path],
    track: NavigationTrack | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,  # (done, total, current_id)
    continue_on_error: bool = True,
    fractions_of_image: list[float | None] | None = None,
) -> tuple[list[InferenceResult], list[dict]]:
    """Run inference over every image; returns (results, errors).

    ``fractions_of_image`` optionally supplies each image's relative along-track
    position (0..1) — the OPEN decision #3 MVP fallback derives it from the
    images' ordered position in the survey (real source ordering, not invented
    coordinates). Errors are reported, never silently swallowed: each error dict
    carries the image path and the failure message.
    """
    results: list[InferenceResult] = []
    errors: list[dict] = []
    total = len(image_paths)
    fractions = (
        list(fractions_of_image)
        if fractions_of_image is not None and len(fractions_of_image) == total
        else [None] * total
    )
    for done, path in enumerate(image_paths, start=1):
        path = Path(path)
        image_id = path.stem
        if on_progress is not None:
            on_progress(done, total, image_id)
        try:
            results.append(engine.run_image_file(
                path, image_id=image_id, track=track,
                fraction_of_image=fractions[done - 1],
            ))
        except (ImageReadError, Exception) as e:  # noqa: BLE001 — batch must continue
            errors.append({"image": str(path), "image_id": image_id, "error": str(e)})
            if not continue_on_error:
                raise
    return results, errors


def collect_survey_images(root: str | Path) -> list[Path]:
    """All supported images under a survey directory, sorted for determinism."""
    root = Path(root)
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
