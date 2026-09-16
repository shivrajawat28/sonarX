/**
 * API client — the frontend's ONLY interaction with the backend.
 *
 * HARD RULE (Section 3, architecture): no ML logic here. This file performs
 * HTTP calls and returns typed data; every classification/confidence/filter/
 * geolocation value is computed server-side and merely displayed.
 */

const rawBase = (import.meta.env.VITE_API_BASE_URL ?? "").trim();
const API_BASE = !rawBase || rawBase.startsWith("http://") || rawBase.startsWith("https://")
  ? rawBase.replace(/\/+$/, "")
  : `https://${rawBase.replace(/\/+$/, "")}`;
const BASE = `${API_BASE}/api/v1`;

export const getApiBase = () => API_BASE;
export const imageUrl = (imageId: string, processed = false) =>
  `${BASE}/images/${encodeURIComponent(imageId)}${processed ? "/processed" : ""}`;

export class ApiError extends Error {
  code: string;
  details: Record<string, unknown>;
  requestId: string | null;

  constructor(code: string, message: string, details: Record<string, unknown> = {}, requestId: string | null = null) {
    super(message);
    this.code = code;
    this.details = details;
    this.requestId = requestId;
  }
}

async function handle<T>(res: Response): Promise<T> {
  if (res.ok) return (await res.json()) as T;
  let code = "HTTP_ERROR";
  let message = `HTTP ${res.status}`;
  let details: Record<string, unknown> = {};
  let requestId: string | null = null;
  try {
    const body = await res.json();
    if (body?.error) {
      code = body.error.code ?? code;
      message = body.error.message ?? message;
      details = body.error.details ?? {};
      requestId = body.error.request_id ?? null;
    }
  } catch {
    /* non-JSON error body — keep defaults */
  }
  throw new ApiError(code, message, details, requestId);
}

function get<T>(path: string): Promise<T> {
  return fetch(`${BASE}${path}`).then((r) => handle<T>(r));
}

function post<T>(path: string, body?: unknown): Promise<T> {
  return fetch(`${BASE}${path}`, {
    method: "POST",
    headers: body !== undefined ? { "Content-Type": "application/json" } : undefined,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  }).then((r) => handle<T>(r));
}

// ---------------------------------------------------------------------------
// Health / models
// ---------------------------------------------------------------------------

export interface HealthInfo {
  status: string;
  app_version: string;
  storage_ok: boolean;
  model: { loaded: boolean; version: string | null; error?: string | null };
}

export const getHealth = () => get<HealthInfo>("/health");

export interface ModelVersionInfo {
  model_version: string;
  architecture_family: string;
  framework: string;
  input_size: [number, number];
  class_map: Record<string, string>;
  status: string;
  notes?: string | null;
  created_at?: string | null;
}

export const listModels = () =>
  get<{ items: ModelVersionInfo[]; loaded_version?: string | null }>("/models");

export const getModel = (version: string) => get<ModelVersionInfo>(`/models/${version}`);

export interface PerClassMetricsInfo {
  precision: number | null;
  recall: number | null;
  f1: number | null;
  ap50: number | null;
  support: number | null;
}

export interface EvalModelProvenance {
  architecture_family: string;
  framework: string;
  checkpoint_path: string;
  input_size: number[];
  class_map: Record<string, string>;
  status: string;
  train_dataset_ref: { path: string; sha256: string } | null;
  train_config: Record<string, unknown>;
  notes?: string | null;
  created_at?: string | null;
}

/** Matches the backend EvaluationRun.model_dump() field names. */
export interface EvalRunInfo {
  eval_run_id: string;
  model_version: string;
  dataset_ref: { path: string; sha256: string } | null;
  split: string | null;
  split_seed?: number | null;
  filter_enabled?: boolean | null;
  metrics: Record<string, number | null>;
  per_class: Record<string, PerClassMetricsInfo>;
  /** Stored confusion-matrix artifact; null when none was recorded. */
  confusion?: Record<string, Record<string, number>> | null;
  /** Training provenance read from the registry entry. */
  model?: EvalModelProvenance | null;
  notes?: string | null;
  timestamp: string;
}

export const getModelMetrics = (version: string) =>
  get<EvalRunInfo>(`/models/${version}/metrics`).catch((e: ApiError) => {
    if (e.code === "NOT_FOUND") return null; // honest "no evaluation yet"
    throw e;
  });

// ---------------------------------------------------------------------------
// Uploads
// ---------------------------------------------------------------------------

export interface UploadedImage {
  image_id: string;
  sha256: string;
  format: string;
  width: number;
  height: number;
  filename: string;
  size_bytes: number;
}

export async function uploadImage(file: File): Promise<UploadedImage> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/uploads/image`, { method: "POST", body: form });
  return handle<UploadedImage>(res);
}

export interface UploadedSurvey {
  survey_id: string;
  name: string;
  image_count: number;
  navigation_status: string;
  images: string[];
  navigation_warnings: string[];
}

export async function uploadSurvey(file: File, name: string): Promise<UploadedSurvey> {
  const form = new FormData();
  form.append("file", file);
  form.append("name", name);
  const res = await fetch(`${BASE}/uploads/survey`, { method: "POST", body: form });
  return handle<UploadedSurvey>(res);
}

// ---------------------------------------------------------------------------
// Preprocessing preview
// ---------------------------------------------------------------------------

export interface AppliedOp {
  op: string;
  params: Record<string, unknown>;
  duration_ms: number;
}

export interface PreprocessPreview {
  image_id: string;
  preview_url: string;
  config_hash: string;
  config_name: string | null;
  applied_ops: AppliedOp[];
  warnings: string[];
}

export const preprocessPreview = (imageId: string) =>
  post<PreprocessPreview>("/previews/preprocess", { image_id: imageId });

// ---------------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------------

export interface Detection {
  detection_id: string;
  run_id?: string | null;
  survey_id?: string | null;
  image_id: string;
  class_name: string;
  model_confidence: number;
  final_confidence: number;
  filtering_status: "accepted" | "flagged" | "rejected";
  filter_reasons: string[];
  /** Source-image pixel space (x,y,w,h) — matches what SonarViewer renders. */
  bbox_source_coords: { x: number; y: number; w: number; h: number };
  bbox_processed_coords?: { x: number; y: number; w: number; h: number };
  latitude: number | null;
  longitude: number | null;
  geo_status: string;
  geo_reason?: string | null;
  /** How coordinates were derived, or why they are absent (backend-computed). */
  geo_provenance?: {
    method?: string | null;
    samples_used?: string[];
    uncertainty_m?: number | null;
    reason?: string | null;
  } | null;
  model_version: string;
  preprocess_config_hash: string;
  analyst_overridden?: boolean;
  analyst_note?: string | null;
  created_at?: string;
}

export interface InferenceResult {
  image_id: string;
  model_version: string;
  preprocess_config_hash: string;
  filter_config_hash: string;
  /** Threshold the pipeline actually ran at (post backend floor) — not merely
   * the requested value. Null when the backend did not report one. */
  applied_confidence_threshold?: number | null;
  detections: Detection[];
  timings_ms: Record<string, number>;
  warnings: string[];
  detection_run_id: string | null;
  processed_image_ref?: string | null;
}

/**
 * Run detection.
 *
 * `confidenceThreshold` is the detector OPERATING POINT: lowering it surfaces
 * more candidates (higher recall, lower precision) instead of missing them.
 * The backend bounds it (floor `MIN_CONFIDENCE_OVERRIDE`) and records the
 * override with the run; filtering still annotates every extra candidate.
 */
export const runDetection = (
  imageId: string,
  save = true,
  confidenceThreshold?: number
) =>
  post<InferenceResult>("/detections/run", {
    image_id: imageId,
    save,
    overrides:
      confidenceThreshold != null ? { confidence_threshold: confidenceThreshold } : undefined,
  });

/**
 * Coordinate space of the image currently on screen.
 *
 * The two spaces genuinely differ whenever the source image is not already
 * 640x640: preprocessing letterboxes it (resize + pad) to the model input, so a
 * box's y in "processed" space is shifted by the letterbox offset. Drawing
 * source-space boxes on the processed image therefore misplaces every overlay.
 */
export type CoordSpace = "source" | "processed";

/**
 * Box in the requested coordinate space.
 *
 * Returns null when that space is unavailable for this detection, so callers
 * can refuse to draw a box rather than silently guess a position.
 */
export function bboxInSpace(
  d: Detection,
  space: CoordSpace
): { x: number; y: number; w: number; h: number } | null {
  if (space === "processed") return d.bbox_processed_coords ?? null;
  return d.bbox_source_coords ?? null;
}

/** bbox (x,y,w,h) -> [x1, y1, x2, y2] for display/overlay. */
export function boxToXyxy(b: { x: number; y: number; w: number; h: number }): [number, number, number, number] {
  return [b.x, b.y, b.x + b.w, b.y + b.h];
}

/** bbox_source_coords (x,y,w,h) -> [x1, y1, x2, y2] for display/overlay. */
export function bboxToXyxy(
  d: Pick<Detection, "bbox_source_coords">
): [number, number, number, number] {
  return boxToXyxy(d.bbox_source_coords);
}

export interface DetectionPage {
  items: Detection[];
  total: number;
  page: number;
}

export const listDetections = (params: Record<string, string | number | undefined> = {}) => {
  const qs = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => {
    if (v !== undefined && v !== "") qs.set(k, String(v));
  });
  return get<DetectionPage>(`/detections?${qs.toString()}`);
};

export const getDetection = (id: string) => get<Detection>(`/detections/${id}`);

export const overrideDetection = (id: string, status: string, note?: string) =>
  fetch(`${BASE}/detections/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ status, note: note ?? null }),
  }).then((r) => handle<Detection>(r));

// ---------------------------------------------------------------------------
// Images (viewer)
// ---------------------------------------------------------------------------

export const originalImageUrl = (imageId: string) => `${BASE}/images/${imageId}`;
export const processedImageUrl = (imageId: string) => `${BASE}/images/${imageId}/processed`;

// ---------------------------------------------------------------------------
// Surveys + jobs
// ---------------------------------------------------------------------------

export interface JobInfo {
  job_id: string;
  type: string;
  status: "pending" | "running" | "succeeded" | "failed";
  progress_pct: number;
  result_ref: Record<string, unknown> | null;
  errors: Array<Record<string, unknown>>;
  created_at: string;
}

export const getJob = (jobId: string) => get<JobInfo>(`/jobs/${jobId}`);

export const runSurveyBatch = (surveyId: string) =>
  post<{ job_id: string; survey_id: string; status: string }>(
    `/surveys/${surveyId}/run`,
    { save: true }
  );

export interface SurveyInfo {
  survey_id: string;
  name: string;
  image_count: number;
  navigation: Record<string, unknown>;
  images?: string[];
}

export const getSurvey = (surveyId: string) =>
  get<SurveyInfo>(`/surveys/${surveyId}`);

// ---------------------------------------------------------------------------
// Reports
// ---------------------------------------------------------------------------

export const createReport = (subject: { run_id?: string | null; survey_id?: string | null }) =>
  post<{ job_id: string }>("/reports", subject);

export const getReport = (reportId: string) => get<{ report_id: string; artifact_path: string; artifact_path_pdf?: string; format?: string }>(`/reports/${reportId}`);
// Backend route: GET /reports/{report_id} returns the artifact with a
// Content-Disposition attachment header (there is no /download suffix route);
// ?format=pdf serves the PDF twin when the report has one.
export const reportDownloadUrl = (reportId: string, format?: "html" | "pdf") =>
  `${BASE}/reports/${reportId}${format === "pdf" ? "?format=pdf" : ""}`;

// ---------------------------------------------------------------------------
// Exports
// ---------------------------------------------------------------------------

const exportQuery = (params: Record<string, string | undefined> = {}) => {
  const qs = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => {
    if (v) qs.set(k, v);
  });
  const s = qs.toString();
  return s ? `?${s}` : "";
};

export const exportCsvUrl = (surveyId?: string) =>
  `${BASE}/exports/detections.csv${exportQuery({ survey_id: surveyId })}`;
export const exportJsonUrl = (surveyId?: string) =>
  `${BASE}/exports/detections.json${exportQuery({ survey_id: surveyId })}`;
