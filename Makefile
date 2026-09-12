# ============================================================================
# Marine Debris Sonar AI — Makefile (Windows / Git-Bash compatible)
# Paths use .venv/Scripts (Windows venv layout); PYTHONPATH uses the Windows
# path separator (;) because Python-on-Windows splits on os.pathsep.
# ============================================================================

PY      := .venv/Scripts/python
PIP     := .venv/Scripts/pip
PYTEST  := .venv/Scripts/python -m pytest
RUNPY   := PYTHONPATH="ml;." .venv/Scripts/python

.PHONY: help venv dev backend frontend test test-ml test-backend lint typecheck \
        smoke-predict verify-env train evaluate seed clean

help:
	@echo "make venv        - create .venv and install backend/ml dependencies"
	@echo "make dev         - run backend (8000) + frontend (5173) for development"
	@echo "make backend     - run FastAPI only"
	@echo "make frontend    - run Vite dev server only"
	@echo "make test        - run all tests (ml + backend)"
	@echo "make test-ml     - run ml pipeline tests only"
	@echo "make test-backend- run backend tests only"
	@echo "make lint        - boundary lint tests (architecture rules)"
	@echo "make typecheck   - frontend tsc --noEmit"
	@echo "make smoke-predict - CLI inference smoke test on a fixture image"
	@echo "make verify-env  - check python/torch/disk/env"

venv:
	python -m venv .venv
	$(PIP) install --quiet numpy opencv-python-headless pyyaml pandas pydantic \
	    pydantic-settings fastapi "uvicorn[standard]" python-multipart pytest httpx jinja2
	@echo "Optional ML extras (large, install manually if needed):"
	@echo "  $(PIP) install torch torchvision --index-url https://download.pytorch.org/whl/cpu"
	@echo "  $(PIP) install ultralytics"

dev:
	@bash -c 'trap "kill 0" EXIT; $(MAKE) -s backend & $(MAKE) -s frontend; wait'

backend:
	$(RUNPY) -m uvicorn backend.app.main:app --reload --port 8000

frontend:
	cd frontend && npm install --no-fund --no-audit && npm run dev

test:
	$(PYTEST) ml/tests backend/tests -v

test-ml:
	$(PYTEST) ml/tests -v

test-backend:
	$(PYTEST) backend/tests -v

lint:
	$(PYTEST) ml/tests/unit/test_boundaries.py backend/tests/unit/test_arch_boundaries.py -v

typecheck:
	cd frontend && npx tsc --noEmit

smoke-predict:
	$(RUNPY) -m ml.scripts.predict_image --image ml/tests/data/fixture_sonar.png

verify-env:
	$(PY) scripts/verify_env.py

train:
	$(RUNPY) -m ml.scripts.train

evaluate:
	$(RUNPY) -m ml.scripts.evaluate

seed:
	$(PY) scripts/seed_demo.py

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} \; 2>/dev/null; true
