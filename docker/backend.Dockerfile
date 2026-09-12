# Backend + ML image (CPU-first; Section 19: GPU optional via base-image swap).
# CPU torch is installed only if needed for a real YOLO model; the stub/ONNX
# path runs on the base deps alone, keeping the demo image small.
FROM python:3.12-slim

WORKDIR /srv

# System deps for opencv-python-headless (libgl not needed with headless build)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

COPY ml/pyproject.toml ml/pyproject.toml
COPY backend/pyproject.toml backend/pyproject.toml
RUN pip install --no-cache-dir \
    numpy opencv-python-headless pyyaml pandas pydantic pydantic-settings \
    fastapi "uvicorn[standard]" python-multipart jinja2 fpdf2
# NOTE: torch + ultralytics are NOT installed here — this image serves the
# clearly-labelled stub path only. For a real-model container, additionally:
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
#   pip install ultralytics
# (weights under models/ are copied below either way; registry decides what loads)

COPY ml/ ml/
COPY backend/ backend/
COPY configs/ configs/
COPY models/ models/

ENV PYTHONPATH=/srv/ml:/srv \
    DATA_ROOT=/srv/data \
    MODELS_DIR=/srv/models

EXPOSE 8000
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
