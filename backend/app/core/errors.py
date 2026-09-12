"""Error taxonomy -> HTTP mapping with one uniform envelope (Section 11.2).

Envelope: {"error": {"code", "message", "details?", "request_id"}}.
Metadata problems are RESULT WARNINGS, not HTTP errors (ADR-008).
"""
from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse

from mlpipeline.detection.base import DetectorLoadError, DetectorPredictError
from mlpipeline.io.image_reader import ImageReadError
from mlpipeline.io.navigation import NavigationParseError
from mlpipeline.preprocessing.pipeline import PreprocessingError


class APIError(Exception):
    """Base for backend-raised errors carrying a taxonomy code."""

    status_code = 500
    code = "INTERNAL_ERROR"

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class InvalidImageError(APIError):
    status_code, code = 422, "INVALID_IMAGE"


class UnsupportedFileTypeError(APIError):
    status_code, code = 415, "UNSUPPORTED_FILE_TYPE"


class FileTooLargeError(APIError):
    status_code, code = 413, "FILE_TOO_LARGE"


class CorruptSonarDataError(APIError):
    status_code, code = 422, "CORRUPT_SONAR_DATA"


class ModelUnavailableError(APIError):
    status_code, code = 503, "MODEL_UNAVAILABLE"


class InferenceFailedError(APIError):
    status_code, code = 500, "INFERENCE_FAILED"


class PreprocessingFailedError(APIError):
    status_code, code = 500, "PREPROCESSING_FAILED"


class NotFoundError(APIError):
    status_code, code = 404, "NOT_FOUND"


class StorageUnavailableError(APIError):
    status_code, code = 503, "STORAGE_UNAVAILABLE"


class ReportGenerationFailedError(APIError):
    status_code, code = 500, "REPORT_GENERATION_FAILED"


VALIDATION_CODE = "VALIDATION_ERROR"


def error_envelope(code: str, message: str, request_id: str, details: dict | None = None) -> dict:
    return {
        "error": {
            "code": code,
            "message": message,
            "details": details or {},
            "request_id": request_id,
        }
    }


def ml_exception_to_api_error(exc: Exception) -> APIError:
    """Translate mlpipeline exceptions into the backend taxonomy (thin bridge)."""
    if isinstance(exc, DetectorLoadError):
        return ModelUnavailableError(str(exc))
    if isinstance(exc, DetectorPredictError):
        return InferenceFailedError(str(exc))
    if isinstance(exc, PreprocessingError):
        return PreprocessingFailedError(str(exc))
    if isinstance(exc, ImageReadError):
        return InvalidImageError(str(exc))
    if isinstance(exc, NavigationParseError):
        # Nav problems are warnings on results (ADR-008); an HTTP error only
        # applies when a caller explicitly demands a parseable nav file.
        return CorruptSonarDataError(str(exc))
    return APIError(str(exc))


async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "")
    return JSONResponse(
        status_code=exc.status_code,
        content=error_envelope(exc.code, exc.message, request_id, exc.details),
    )
