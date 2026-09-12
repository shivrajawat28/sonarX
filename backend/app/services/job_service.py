"""Job service (ADR-005): persisted in-process background jobs.

- No Redis/Celery/Kafka (Section 24). Job records survive page refreshes.
- Survey batch loops the SAME SonarInferenceEngine per image (no second path).
- Report generation renders a PDF-style HTML artifact.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from fastapi import BackgroundTasks
from fastapi.background import BackgroundTasks as _BT  # noqa: F401 (doc parity)

from backend.app.core.config import Settings, get_settings
from backend.app.services.inference_service import InferenceService, get_inference_service
from backend.app.services.storage_service import get_storage_service

logger = logging.getLogger("api.jobs")


class JobService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.repo = get_inference_service(settings).repo
        self.storage = get_storage_service(settings)
        self._bg: BackgroundTasks | None = None  # set per-request

    def attach_background(self, bg: BackgroundTasks) -> "JobService":
        self._bg = bg
        return self

    # -- job records (persisted; GET /jobs/{id} reads these) -------------------
    def create(self, job_type: str, payload: dict) -> dict:
        from mlpipeline.datatypes.common import new_id

        job = {
            "job_id": new_id("job"),
            "type": job_type,
            "status": "pending",
            "progress_pct": 0,
            "payload": payload,
            "result_ref": None,
            "errors": [],
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        return self.repo.insert("jobs", job)

    def get(self, job_id: str) -> dict | None:
        return self.repo.get("jobs", job_id)

    def _update(self, job_id: str, **patch) -> None:
        self.repo.update("jobs", job_id, patch)

    # -- survey batch ----------------------------------------------------------
    def submit_batch(self, job_id: str, service: InferenceService) -> None:
        if self._bg is None:
            raise RuntimeError("JobService.attach_background() required before submit")
        self._bg.add_task(self._run_batch, job_id)

    def _run_batch(self, job_id: str) -> None:
        from mlpipeline.inference import run_survey_batch
        from mlpipeline.io.image_reader import ImageReadError

        job = self.get(job_id)
        if job is None:
            return
        survey_id = job["payload"]["survey_id"]
        save = job["payload"].get("save", True)
        try:
            survey = self.repo.get("surveys", survey_id)
            image_ids = survey.get("images") or []
            settings = get_settings()
            service = get_inference_service(settings)
            engine = service.require_engine()

            def progress(done: int, total: int, current: str) -> None:
                self._update(job_id, status="running",
                             progress_pct=round(100 * done / max(total, 1)))

            self._update(job_id, status="running", progress_pct=1)
            paths = []
            for iid in image_ids:
                img_doc = self.repo.get("sonar_images", iid)
                if img_doc:
                    from backend.app.core.security import resolve_within

                    paths.append(resolve_within(settings.data_root, img_doc["source_path"]))

            # Navigation (Section 10): load the survey's REAL track if one was
            # parsed at upload. Per-tile along-track fractions come from the
            # tiles' ordered position in the survey (OPEN #3 MVP fallback —
            # real source ordering, not fabricated coordinates).
            track = None
            fractions: list[float | None] | None = None
            nav = (survey.get("navigation") or {})
            track_ref = nav.get("track_ref")
            if nav.get("status") == "present" and track_ref:
                from backend.app.core.security import resolve_within

                track_path = resolve_within(settings.data_root, track_ref)
                if track_path.is_file():
                    from mlpipeline.io.navigation import parse_navigation

                    try:
                        track = parse_navigation(track_path)
                    except Exception as e:  # noqa: BLE001 — missing nav must not kill batch
                        logger.warning("survey %s nav parse failed: %s", survey_id, e)
            if track is not None:
                n = len(paths)
                fractions = [i / max(n - 1, 1) for i in range(n)]

            results, errors = run_survey_batch(
                engine, paths, track=track, on_progress=progress,
                fractions_of_image=fractions,
            )

            run_ids: list[str] = []
            detection_ids: list[str] = []
            for result in results:
                payload = result.model_dump()
                if save:
                    from mlpipeline.datatypes.common import new_id

                    run_doc = {
                        "run_id": new_id("run"),
                        "kind": "survey_batch",
                        "survey_id": survey_id,
                        "image_id": result.image_id,
                        "model_version": result.model_version,
                        "preprocess_config_hash": result.preprocess_config_hash,
                        "filter_config_hash": result.filter_config_hash,
                        "detection_ids": [d["detection_id"] for d in payload["detections"]],
                        "image_ids": [result.image_id],
                        "timings_ms": payload["timings_ms"],
                        "warnings": result.warnings,
                        "created_at": result.created_at,
                    }
                    run_doc = self.repo.insert("detection_runs", run_doc)
                    run_ids.append(run_doc["run_id"])
                    for d in payload["detections"]:
                        d["run_id"] = run_doc["run_id"]
                        d["survey_id"] = survey_id  # history/report filtering (canonical type has no survey field)
                        self.repo.insert("detections", d)
                        detection_ids.append(d["detection_id"])

            self._update(
                job_id, status="succeeded", progress_pct=100,
                result_ref={"run_ids": run_ids, "n_images": len(results),
                            "n_detections": len(detection_ids)},
                errors=errors,
            )
        except Exception as e:  # noqa: BLE001 — job must record failure, not die silently
            logger.exception("batch job %s failed", job_id)
            self._update(job_id, status="failed", errors=[{"error": str(e)}])

    # -- reports ----------------------------------------------------------------
    def submit_report(self, job_id: str, service: InferenceService) -> None:
        if self._bg is None:
            raise RuntimeError("JobService.attach_background() required before submit")
        self._bg.add_task(self._run_report, job_id)

    def _run_report(self, job_id: str) -> None:
        try:
            job = self.get(job_id)
            survey_id = job["payload"].get("survey_id")
            run_id = job["payload"].get("run_id")

            self._update(job_id, status="running", progress_pct=10)

            # list_all (not list): a report is an exported artifact, so the
            # 200-per-page cap must not silently drop detections.
            if survey_id:
                subject = self.repo.get("surveys", survey_id)
                image_ids = subject.get("images") or []
                detections, total = self.repo.list_all("detections",
                                                       filters={"survey_id": survey_id})
                kind = "survey_summary"
            else:
                run = self.repo.get("detection_runs", run_id)
                image_ids = run.get("image_ids") or []
                detections, total = self.repo.list_all("detections",
                                                       filters={"run_id": run_id})
                kind = "run_summary"

            n = {"accepted": 0, "flagged": 0, "rejected": 0}
            for d in detections:
                s = d.get("filtering_status")
                if s in n:
                    n[s] += 1

            # Provenance block: model + preprocessing + filter hashes actually
            # used by these detections (never invented — read from the records).
            model_versions = sorted({d.get("model_version") for d in detections if d.get("model_version")})
            pp_hashes = sorted({(d.get("preprocess_config_hash") or "")[:16] for d in detections if d.get("preprocess_config_hash")})
            filter_hashes = sorted({(d.get("filter_config_hash") or "")[:16] for d in detections if d.get("filter_config_hash")})

            meta_block = {
                "model_versions": model_versions,
                "pp_hashes": pp_hashes,
                "filter_hashes": filter_hashes,
                "kind": kind,
            }
            html = self._render_html(
                title=f"Detection Report — {survey_id or run_id}",
                rows=detections, counts=n, image_count=len(image_ids),
                meta=meta_block,
            )
            artifact = self.storage.save_artifact(
                html.encode("utf-8"), "reports", f"{job_id}.html"
            )
            # Dual-format: HTML (browser-friendly) + PDF (print/archive). Both
            # carry identical content. PDF via fpdf2 (pure-python, no system deps).
            pdf_bytes = self._render_pdf(
                title=f"Detection Report — {survey_id or run_id}",
                rows=detections, counts=n, image_count=len(image_ids), meta=meta_block,
            )
            artifact_pdf = self.storage.save_artifact(
                pdf_bytes, "reports", f"{job_id}.pdf"
            )

            from mlpipeline.datatypes.common import new_id

            report = {
                "report_id": new_id("rpt"),
                "kind": kind,
                "subject_ref": survey_id or run_id,
                "format": "html+pdf",
                "artifact_path": str(artifact.relative_to(self.settings.data_root)),
                "artifact_path_pdf": str(artifact_pdf.relative_to(self.settings.data_root)),
                "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "model_version": (detections[0].get("model_version") if detections else None),
                "content_stats": {
                    "images": len(image_ids),
                    "detections": total,
                    "accepted": n["accepted"],
                    "flagged": n["flagged"],
                    "rejected": n["rejected"],
                },
            }
            report = self.repo.insert("reports", report)
            self._update(job_id, status="succeeded", progress_pct=100,
                         result_ref={"report_id": report["report_id"]})
        except Exception as e:  # noqa: BLE001
            logger.exception("report job %s failed", job_id)
            self._update(job_id, status="failed", errors=[{"error": str(e)}])

    @staticmethod
    def _latin(text: object) -> str:
        """fpdf2 core fonts are latin-1 only; map common unicode punctuation down
        and hard-replace anything outside the charset (never crash a report)."""
        s = str(text)
        for ch, rep in (("\u2014", "-"), ("\u2013", "-"), ("\u00b7", "."), ("\u2019", "'"), ("\u201c", '"'), ("\u201d", '"')):
            s = s.replace(ch, rep)
        return s.encode("latin-1", errors="replace").decode("latin-1")

    def _render_pdf(self, title: str, rows: list[dict], counts: dict, image_count: int,
                    meta: dict | None = None) -> bytes:
        """PDF twin of the HTML report (fpdf2). Content mirrors the HTML template:
        summary, provenance block, and the full detection table with statuses."""
        from fpdf import FPDF

        pdf = FPDF(format="A4")
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("helvetica", "B", 15)
        pdf.cell(0, 10, self._latin(title[:90]), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("helvetica", "", 9)
        pdf.cell(0, 6,
                 f"Generated {datetime.now(UTC).isoformat(timespec='seconds')} | "
                 f"{image_count} images | {len(rows)} detections "
                 f"({counts['accepted']} accepted / {counts['flagged']} flagged / {counts['rejected']} rejected)",
                 new_x="LMARGIN", new_y="NEXT")
        meta = meta or {}
        pdf.set_font("helvetica", "", 8)
        pdf.cell(0, 5, f"Model: {', '.join(meta.get('model_versions') or []) or '-'}",
                 new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 5, f"Preprocessing config (sha256/16): {', '.join(meta.get('pp_hashes') or []) or '-'}",
                 new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 5, f"Filter config (sha256/16): {', '.join(meta.get('filter_hashes') or []) or '-'}",
                 new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

        # Table header
        pdf.set_font("helvetica", "B", 8)
        pdf.set_fill_color(240, 240, 240)
        headers = ["Class", "Model conf", "Final conf", "Status", "BBox x,y,w,h (src px)", "Filter reason", "Lat/Lon"]
        widths = [32, 17, 17, 16, 33, 52, 23]
        for h, w in zip(headers, widths):
            pdf.cell(w, 6, h, border=1, fill=True)
        pdf.ln()

        pdf.set_font("helvetica", "", 7.5)
        for d in rows:
            bbox = d.get("bbox_source_coords") or {}
            lat = d.get("latitude")
            lon = d.get("longitude")
            geo = f"{lat:.5f}, {lon:.5f}" if isinstance(lat, (int, float)) and isinstance(lon, (int, float)) else "unavailable"
            reasons = "; ".join(d.get("filter_reasons") or []) or "-"
            bbox_txt = (
                f"{bbox.get('x', 0):.0f},{bbox.get('y', 0):.0f},"
                f"{bbox.get('w', 0):.0f},{bbox.get('h', 0):.0f}"
                if bbox else "-"
            )
            cells = [
                self._latin(str(d.get("class_name") or "-")[:20]),
                f"{(d.get('model_confidence') or 0):.3f}",
                f"{(d.get('final_confidence') or 0):.3f}",
                self._latin(str(d.get("filtering_status") or "-")),
                bbox_txt,
                self._latin(reasons[:70]),
                geo,
            ]
            h = 6
            for (txt, w) in zip(cells, widths):
                pdf.cell(w, h, str(txt), border=1)
            pdf.ln()

        pdf.ln(2)
        pdf.set_font("helvetica", "I", 7.5)
        pdf.multi_cell(0, 4.5,
                       self._latin("Coordinates are shown only where real navigation metadata was parsed; "
                       "they are null (unavailable) otherwise and are never fabricated. "
                       "Model confidence is the raw detector output; final confidence applies "
                       "deterministic filtering adjustments. Filtering annotates detections "
                       "with reasons - it never silently deletes them."))
        return bytes(pdf.output())

    def _render_html(self, title: str, rows: list[dict], counts: dict, image_count: int,
                     meta: dict | None = None) -> str:
        """Jinja2 template rendering with autoescape (Section 20 input sanitization)."""
        from jinja2 import Environment, select_autoescape

        env = Environment(autoescape=select_autoescape(["html"]))
        template = env.from_string(_REPORT_TEMPLATE)
        return template.render(
            title=title, rows=rows, counts=counts, image_count=image_count,
            generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
            meta=meta or {},
        )


_JOB_BG: dict[str, BackgroundTasks] = {}


def get_job_service(settings: Settings) -> JobService:
    # One service instance per app; BackgroundTasks arrive per request and are
    # attached before submit (FastAPI DI in the routers).
    return JobService(settings)


def job_service_with_bg(settings: Settings, bg: BackgroundTasks) -> JobService:
    return get_job_service(settings).attach_background(bg)


_REPORT_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>{{ title }}</title>
<style>
 body { font-family: -apple-system, sans-serif; margin: 2rem; color: #111 }
 table { border-collapse: collapse; width: 100% }
 th, td { border: 1px solid #ccc; padding: 6px 8px; font-size: 12px; text-align: left }
 th { background: #f0f0f0 }
 .accepted { color: #0a7d24 } .flagged { color: #b07000 } .rejected { color: #a00 }
 .notice { background: #fff8e1; padding: 8px 12px; border-left: 4px solid #e0b400; margin: 1rem 0 }
</style></head>
<body>
<h1>{{ title }}</h1>
<p>Generated {{ generated_at }} · {{ image_count }} images · {{ rows|length }} detections shown
   ({{ counts.accepted }} accepted / {{ counts.flagged }} flagged / {{ counts.rejected }} rejected)</p>
{% if meta %}
<p class="muted" style="font-size:12px">
  Model: {{ meta.model_versions|join(', ') or '—' }} ·
  Preprocessing config: {{ meta.pp_hashes|join(', ') or '—' }}… ·
  Filter config: {{ meta.filter_hashes|join(', ') or '—' }}… ·
  Report kind: {{ meta.kind }}
</p>
{% endif %}
<div class="notice">Coordinates are present ONLY where real navigation metadata existed.
   Detections with null coordinates are shown without location — none are fabricated.</div>
<table>
<tr><th>ID</th><th>Image</th><th>Class</th><th>Model conf.</th><th>Final conf.</th>
    <th>Status</th><th>BBox x,y,w,h (source px)</th><th>Reasons</th>
    <th>Lat</th><th>Lon</th><th>Geo method</th><th>Model</th></tr>
{% for d in rows %}
<tr>
 <td>{{ d.detection_id }}</td><td>{{ d.image_id }}</td><td>{{ d.class_name }}</td>
 <td>{{ '%.3f'|format(d.model_confidence) }}</td><td>{{ '%.3f'|format(d.final_confidence) }}</td>
 <td class="{{ d.filtering_status }}">{{ d.filtering_status }}</td>
 <td>{{ '%.0f,%.0f,%.0f,%.0f'|format(d.bbox_source_coords.x, d.bbox_source_coords.y, d.bbox_source_coords.w, d.bbox_source_coords.h) if d.bbox_source_coords else '—' }}</td>
 <td>{{ d.filter_reasons|join('; ') }}</td>
 <td>{{ d.latitude if d.latitude is not none else '—' }}</td>
 <td>{{ d.longitude if d.longitude is not none else '—' }}</td>
 <td>{{ d.geo_provenance.method if d.geo_provenance and d.geo_provenance.method else (d.geo_status or '—') }}</td>
 <td>{{ d.model_version }}</td>
</tr>
{% endfor %}
</table>
</body></html>
"""
