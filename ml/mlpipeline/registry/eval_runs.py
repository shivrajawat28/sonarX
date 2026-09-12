"""Evaluation run store: immutable EvaluationRun records (Section 6.5 / ADR-010).

Records live in models/eval/<model_version>/<eval_run_id>/ as record.json plus
artifacts (confusion matrix, PR curves, failure cases). The UI reads metrics
from these records only — never recomputed ad hoc.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from mlpipeline.datatypes.model import EvaluationRun


class EvalRunStore:
    def __init__(self, eval_root: str | Path) -> None:
        self.root = Path(eval_root)
        self._lock = threading.Lock()

    def _run_dir(self, model_version: str, eval_run_id: str) -> Path:
        return self.root / model_version / eval_run_id

    def save(self, run: EvaluationRun, artifacts: dict[str, str] | None = None) -> Path:
        """Persist an eval run record. Same eval_run_id twice is refused (immutable)."""
        d = self._run_dir(run.model_version, run.eval_run_id)
        with self._lock:
            record = d / "record.json"
            if record.exists():
                raise FileExistsError(f"eval run '{run.eval_run_id}' already exists (immutable)")
            d.mkdir(parents=True, exist_ok=True)
            record.write_text(json.dumps(run.model_dump(), indent=2), encoding="utf-8")
            for name, content in (artifacts or {}).items():
                p = d / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content, encoding="utf-8")
            return record

    def get(self, model_version: str, eval_run_id: str) -> EvaluationRun | None:
        record = self._run_dir(model_version, eval_run_id) / "record.json"
        if not record.is_file():
            return None
        return EvaluationRun.model_validate(json.loads(record.read_text(encoding="utf-8")))

    def list_for_model(self, model_version: str) -> list[EvaluationRun]:
        base = self.root / model_version
        if not base.is_dir():
            return []
        runs = []
        for d in sorted(base.iterdir()):
            record = d / "record.json"
            if record.is_file():
                runs.append(EvaluationRun.model_validate(json.loads(record.read_text(encoding="utf-8"))))
        return sorted(runs, key=lambda r: r.timestamp)

    def latest_for_model(self, model_version: str) -> EvaluationRun | None:
        runs = self.list_for_model(model_version)
        return runs[-1] if runs else None

    def read_json_artifact(
        self, model_version: str, eval_run_id: str, name: str
    ) -> dict | None:
        """Read a JSON artifact stored beside an eval run's record.

        Returns None when the artifact is absent or unparseable, so callers report
        "unavailable" instead of failing the whole metrics response.
        """
        path = self._run_dir(model_version, eval_run_id) / name
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        return data if isinstance(data, dict) else None


_default_store: EvalRunStore | None = None


def get_eval_store() -> EvalRunStore:
    global _default_store
    if _default_store is None:
        root = Path(__file__).resolve().parents[3]
        _default_store = EvalRunStore(root / "models" / "eval")
    return _default_store
