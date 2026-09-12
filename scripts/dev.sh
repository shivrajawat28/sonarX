#!/usr/bin/env bash
# Start backend (8000) + frontend (5173) together for development.
# POSIX/Git-Bash compatible replacement for `make dev` (no make on this box).
set -e
cd "$(dirname "$0")/.."

cleanup() { kill 0 2>/dev/null; }
trap cleanup EXIT

echo "== backend  -> http://localhost:8000 (log: backend_server.log) =="
PYTHONPATH="ml;." .venv/Scripts/python -m uvicorn backend.app.main:app --reload --port 8000 &
BACK_PID=$!

echo "== frontend -> http://localhost:5173 =="
cd frontend
npm run dev &
FRONT_PID=$!

wait
