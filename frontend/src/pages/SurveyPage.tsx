import { useState, useEffect, useMemo } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Detection,
  JobInfo,
  exportCsvUrl,
  exportJsonUrl,
  getHealth,
  getJob,
  getSurvey,
  listDetections,
  runSurveyBatch,
  uploadSurvey,
  imageUrl,
} from "../api/client";
import ErrorBox from "../components/ErrorBox";
import MapView from "../components/MapView";
import ConfidenceIndicator from "../components/ConfidenceIndicator";
import SonarViewer from "../components/SonarViewer";

/**
 * SONARX Survey (batch) page — Along-track mission archive processing & geospatial intelligence.
 */
export default function SurveyPage() {
  const qc = useQueryClient();
  const [name, setName] = useState("");
  const [jobId, setJobId] = useState<string | null>(null);
  const [selectedDetectionId, setSelectedDetectionId] = useState<string | null>(null);
  const [isDemoLoading, setIsDemoLoading] = useState(false);
  const [isDemoActive, setIsDemoActive] = useState(false);

  // Health Query for backend and active model
  const healthQ = useQuery({
    queryKey: ["health"],
    queryFn: getHealth,
    refetchInterval: 15000,
  });

  const upload = useMutation({
    mutationFn: (file: File) => uploadSurvey(file, name),
    onSuccess: () => {
      setJobId(null);
      setSelectedDetectionId(null);
      qc.invalidateQueries({ queryKey: ["detections"] });
    },
  });

  const batch = useMutation({
    mutationFn: (surveyId: string) => runSurveyBatch(surveyId),
    onSuccess: (r) => setJobId(r.job_id),
  });

  const surveyId = upload.data?.survey_id ?? null;

  const jobQ = useQuery({
    queryKey: ["job", jobId],
    queryFn: () => getJob(jobId as string),
    enabled: jobId != null,
    refetchInterval: (q) => {
      const s = (q.state.data as JobInfo | undefined)?.status;
      return s === "succeeded" || s === "failed" ? false : 1000;
    },
  });
  const job = jobQ.data;

  const surveyQ = useQuery({
    queryKey: ["survey", surveyId],
    queryFn: () => getSurvey(surveyId as string),
    enabled: surveyId != null,
  });

  const detQ = useQuery({
    queryKey: ["survey-detections", surveyId, job?.status],
    queryFn: () => listDetections({ survey_id: surveyId as string, size: 200 }),
    enabled: surveyId != null && job?.status === "succeeded",
  });

  const detections: Detection[] = detQ.data?.items ?? [];
  const geo = detections.filter((d) => d.latitude != null && d.longitude != null);

  const nav = (surveyQ.data?.navigation ?? {}) as Record<string, unknown>;
  const navStatus = upload.data?.navigation_status ?? (nav.status as string | undefined);

  // Auto-select first detection when detections arrive if none selected
  useEffect(() => {
    if (detections.length > 0 && !selectedDetectionId) {
      setSelectedDetectionId(detections[0].detection_id);
    }
  }, [detections, selectedDetectionId]);

  const selectedDetection = useMemo(() => {
    return detections.find((d) => d.detection_id === selectedDetectionId) ?? (detections[0] || null);
  }, [detections, selectedDetectionId]);

  // Handle 1-Click Load Demo Survey
  const handleLoadDemoSurvey = async () => {
    try {
      setIsDemoLoading(true);
      setName("Demo Survey — Arabian Sea Pipeline Track");
      const resp = await fetch("/demo_samples/geo_survey_demo.zip");
      if (!resp.ok) {
        throw new Error(`Failed to load demo archive: HTTP ${resp.status}`);
      }
      const blob = await resp.blob();
      const file = new File([blob], "geo_survey_demo.zip", { type: "application/zip" });

      const uploadResult = await upload.mutateAsync(file);
      setIsDemoActive(true);
      // Auto-trigger batch execution for instant judge demonstration
      if (uploadResult?.survey_id) {
        batch.mutate(uploadResult.survey_id);
      }
    } catch (err) {
      console.error("Error loading demo survey:", err);
    } finally {
      setIsDemoLoading(false);
    }
  };

  const handleClearSurvey = () => {
    setJobId(null);
    setSelectedDetectionId(null);
    setIsDemoActive(false);
    setName("");
    qc.invalidateQueries({ queryKey: ["survey"] });
    qc.invalidateQueries({ queryKey: ["survey-detections"] });
  };

  // Pipeline step progress determination
  const isArchiveUploaded = !!upload.data;
  const isProcessing = job?.status === "running";
  const isCompleted = job?.status === "succeeded";
  const isNavPresent = navStatus === "present";

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1.25rem" }}>
      {/* ================= HEADER & BRANDING ================= */}
      <div
        className="panel"
        style={{
          background: "linear-gradient(135deg, rgba(7, 16, 30, 0.95) 0%, rgba(13, 31, 56, 0.9) 100%)",
          border: "1px solid rgba(0, 242, 254, 0.25)",
          boxShadow: "0 10px 30px rgba(0, 0, 0, 0.5)",
          display: "flex",
          flexDirection: "column",
          gap: "0.75rem",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: "1rem" }}>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
              <h1 style={{ margin: 0, fontSize: "1.75rem", letterSpacing: "0.02em", color: "#fff" }}>
                <span style={{ color: "#00f2fe" }}>SONARX</span> Survey Mission Batch
              </h1>
              <span className="badge accepted" style={{ fontSize: "0.75rem" }}>
                Survey (batch) selected
              </span>
            </div>
            <p className="muted" style={{ margin: "0.3rem 0 0", fontSize: "0.86rem" }}>
              AI-Powered Underwater Intelligence · Automated Side-Scan Sonar mission batch processing with along-track GPS interpolation.
            </p>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: "0.6rem", flexWrap: "wrap" }}>
            {/* Active Model Indicator */}
            <span
              className="badge"
              style={{
                background: "rgba(0, 242, 254, 0.1)",
                color: "#00f2fe",
                border: "1px solid rgba(0, 242, 254, 0.3)",
                fontSize: "0.78rem",
              }}
            >
              Model: DRISHTI-SS YOLOv8n E30 FINAL
            </span>

            {/* Backend Connection Status */}
            {healthQ.data?.status === "ok" || healthQ.data?.status === "healthy" ? (
              <span
                className="badge accepted"
                style={{
                  background: "rgba(16, 185, 129, 0.12)",
                  color: "#10b981",
                  border: "1px solid rgba(16, 185, 129, 0.3)",
                  fontSize: "0.78rem",
                }}
              >
                ● Backend Connected
              </span>
            ) : (
              <span className="badge rejected" style={{ fontSize: "0.78rem" }}>
                ● Backend Disconnected
              </span>
            )}

            {/* Mode Badge */}
            {isDemoActive ? (
              <span
                className="badge-demo-mode"
                style={{
                  background: "rgba(168, 85, 247, 0.15)",
                  color: "#c084fc",
                  border: "1px solid rgba(168, 85, 247, 0.4)",
                  padding: "0.35rem 0.75rem",
                  fontSize: "0.8rem",
                  fontWeight: 700,
                  borderRadius: "999px",
                }}
              >
                ● DEMO SURVEY — REAL BATCH INFERENCE
              </span>
            ) : upload.data ? (
              <span
                className="badge-live-mode"
                style={{
                  background: "rgba(0, 242, 254, 0.15)",
                  color: "#00f2fe",
                  border: "1px solid rgba(0, 242, 254, 0.4)",
                  padding: "0.35rem 0.75rem",
                  fontSize: "0.8rem",
                  fontWeight: 700,
                  borderRadius: "999px",
                }}
              >
                ● CUSTOM SURVEY ARCHIVE
              </span>
            ) : null}
          </div>
        </div>

        {/* 1-Click Presentation Action Bar */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            flexWrap: "wrap",
            gap: "0.75rem",
            paddingTop: "0.6rem",
            borderTop: "1px solid rgba(255, 255, 255, 0.08)",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", flexWrap: "wrap" }}>
            <button
              type="button"
              onClick={handleLoadDemoSurvey}
              disabled={isDemoLoading || batch.isPending || isProcessing}
              style={{
                background: "linear-gradient(135deg, #0284c7 0%, #00f2fe 100%)",
                color: "#051326",
                fontWeight: 700,
                fontSize: "0.88rem",
                padding: "0.55rem 1.25rem",
                border: "none",
                borderRadius: "var(--radius-md)",
                cursor: "pointer",
                boxShadow: "0 0 20px rgba(0, 242, 254, 0.35)",
                display: "flex",
                alignItems: "center",
                gap: "0.5rem",
              }}
            >
              <span>🚀</span> {isDemoLoading ? "Loading Demo…" : "Load Demo Survey"}
            </button>

            {upload.data && (
              <button
                type="button"
                className="secondary"
                onClick={handleClearSurvey}
                style={{ fontSize: "0.82rem", padding: "0.55rem 0.9rem" }}
              >
                ↺ Clear Survey
              </button>
            )}

            <span className="muted" style={{ fontSize: "0.8rem" }}>
              {isDemoActive
                ? "Loaded bundled Arabian Sea survey archive (3 real tiles + nav.csv). Ready for judge demonstration."
                : "Click 'Load Demo Survey' to run 1-click verified mission batch without manual file selection."}
            </span>
          </div>

          {upload.data && job?.status === "succeeded" && (
            <div style={{ display: "flex", gap: "0.5rem" }}>
              <a href={exportCsvUrl(surveyId ?? undefined)} download>
                <button type="button" className="secondary" style={{ fontSize: "0.8rem", padding: "0.45rem 0.8rem" }}>
                  Export CSV
                </button>
              </a>
              <a href={exportJsonUrl(surveyId ?? undefined)} download>
                <button type="button" className="secondary" style={{ fontSize: "0.8rem", padding: "0.45rem 0.8rem" }}>
                  Export JSON
                </button>
              </a>
            </div>
          )}
        </div>
      </div>

      {/* ================= SURVEY SUMMARY METRIC CARDS ================= */}
      {upload.data && (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
            gap: "0.75rem",
          }}
        >
          <div className="panel" style={{ padding: "0.85rem 1rem", borderLeft: "3px solid #00f2fe" }}>
            <div className="muted" style={{ fontSize: "0.72rem", textTransform: "uppercase", letterSpacing: "0.05em" }}>
              Survey Mission
            </div>
            <div style={{ fontSize: "1rem", fontWeight: 700, color: "#fff", marginTop: "0.2rem", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {upload.data.name || upload.data.survey_id}
            </div>
            <div className="mono muted" style={{ fontSize: "0.72rem" }}>
              {upload.data.survey_id}
            </div>
          </div>

          <div className="panel" style={{ padding: "0.85rem 1rem", borderLeft: "3px solid #38bdf8" }}>
            <div className="muted" style={{ fontSize: "0.72rem", textTransform: "uppercase", letterSpacing: "0.05em" }}>
              Sonar Tiles
            </div>
            <div style={{ fontSize: "1.3rem", fontWeight: 700, color: "#38bdf8", marginTop: "0.1rem" }}>
              {upload.data.image_count} Tiles
            </div>
            <div className="muted" style={{ fontSize: "0.72rem" }}>
              Acoustic backscatter
            </div>
          </div>

          <div className="panel" style={{ padding: "0.85rem 1rem", borderLeft: "3px solid #10b981" }}>
            <div className="muted" style={{ fontSize: "0.72rem", textTransform: "uppercase", letterSpacing: "0.05em" }}>
              Processing Status
            </div>
            <div style={{ fontSize: "1.3rem", fontWeight: 700, color: isCompleted ? "#10b981" : isProcessing ? "#f59e0b" : "#94a3b8", marginTop: "0.1rem" }}>
              {isCompleted ? "Completed (100%)" : isProcessing ? `Running (${job?.progress_pct ?? 0}%)` : "Awaiting Run"}
            </div>
            <div className="muted" style={{ fontSize: "0.72rem" }}>
              {isCompleted ? `${detections.length} images processed` : "YOLOv8n batch loop"}
            </div>
          </div>

          <div className="panel" style={{ padding: "0.85rem 1rem", borderLeft: "3px solid #f59e0b" }}>
            <div className="muted" style={{ fontSize: "0.72rem", textTransform: "uppercase", letterSpacing: "0.05em" }}>
              Detections Found
            </div>
            <div style={{ fontSize: "1.3rem", fontWeight: 700, color: "#f59e0b", marginTop: "0.1rem" }}>
              {detections.length} Objects
            </div>
            <div className="muted" style={{ fontSize: "0.72rem" }}>
              Subsea infrastructure & debris
            </div>
          </div>

          <div className="panel" style={{ padding: "0.85rem 1rem", borderLeft: "3px solid #a855f7" }}>
            <div className="muted" style={{ fontSize: "0.72rem", textTransform: "uppercase", letterSpacing: "0.05em" }}>
              Geolocated Coordinates
            </div>
            <div style={{ fontSize: "1.3rem", fontWeight: 700, color: geo.length > 0 ? "#a855f7" : "#ef4444", marginTop: "0.1rem" }}>
              {geo.length} / {detections.length}
            </div>
            <div className="muted" style={{ fontSize: "0.72rem" }}>
              {isNavPresent ? "Matched via nav.csv track" : "No nav metadata"}
            </div>
          </div>
        </div>
      )}

      {/* ================= BATCH PIPELINE VISUALIZER ================= */}
      {upload.data && (
        <div
          className="panel"
          style={{
            padding: "0.75rem 1rem",
            background: "rgba(10, 24, 46, 0.6)",
            border: "1px solid rgba(0, 242, 254, 0.15)",
          }}
        >
          <div style={{ fontSize: "0.75rem", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--text-secondary)", marginBottom: "0.5rem" }}>
            Survey Mission Execution Pipeline
          </div>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              flexWrap: "wrap",
              gap: "0.4rem",
              fontSize: "0.76rem",
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: "0.3rem", color: isArchiveUploaded ? "#10b981" : "#64748b" }}>
              <span>{isArchiveUploaded ? "✓" : "○"}</span> 1. Survey Archive
            </div>
            <span className="muted">→</span>
            <div style={{ display: "flex", alignItems: "center", gap: "0.3rem", color: isArchiveUploaded ? "#10b981" : "#64748b" }}>
              <span>{isArchiveUploaded ? "✓" : "○"}</span> 2. Sonar Tiles ({upload.data.image_count})
            </div>
            <span className="muted">→</span>
            <div style={{ display: "flex", alignItems: "center", gap: "0.3rem", color: isCompleted ? "#10b981" : isProcessing ? "#f59e0b" : "#64748b" }}>
              <span>{isCompleted ? "✓" : isProcessing ? "⚡" : "○"}</span> 3. AI Detection (YOLOv8n)
            </div>
            <span className="muted">→</span>
            <div style={{ display: "flex", alignItems: "center", gap: "0.3rem", color: isCompleted ? "#10b981" : "#64748b" }}>
              <span>{isCompleted ? "✓" : "○"}</span> 4. Rule Filtering (Edge-clip)
            </div>
            <span className="muted">→</span>
            <div style={{ display: "flex", alignItems: "center", gap: "0.3rem", color: isCompleted && isNavPresent ? "#10b981" : isCompleted ? "#ef4444" : "#64748b" }}>
              <span>{isCompleted && isNavPresent ? "✓" : isCompleted ? "✕" : "○"}</span> 5. Navigation Matching (nav.csv)
            </div>
            <span className="muted">→</span>
            <div style={{ display: "flex", alignItems: "center", gap: "0.3rem", color: isCompleted && geo.length > 0 ? "#10b981" : isCompleted ? "#ef4444" : "#64748b" }}>
              <span>{isCompleted && geo.length > 0 ? "✓" : "○"}</span> 6. Along-Track Interp
            </div>
            <span className="muted">→</span>
            <div style={{ display: "flex", alignItems: "center", gap: "0.3rem", color: isCompleted && geo.length > 0 ? "#10b981" : "#64748b" }}>
              <span>{isCompleted && geo.length > 0 ? "✓" : "○"}</span> 7. Geospatial Mapping
            </div>
          </div>
        </div>
      )}

      {/* ================= MAIN CONTENT: Upload Controls & Map / Results ================= */}
      <div className="row" style={{ alignItems: "flex-start" }}>
        {/* Left Column: Upload & Batch Controls */}
        <section
          className="panel grow"
          style={{ maxWidth: 360, display: "flex", flexDirection: "column", gap: "1rem" }}
        >
          <div>
            <h2>1 · Upload survey archive</h2>
            <p className="muted" style={{ fontSize: "0.8rem", margin: "0 0 0.75rem" }}>
              A .zip of sonar tiles, optionally with a <code>nav.csv</code> /{" "}
              <code>navigation.csv</code> sidecar (columns: timestamp, latitude,
              longitude). Coordinates come only from that file.
            </p>
            <input
              placeholder="survey name (optional)"
              value={name}
              onChange={(e) => setName(e.target.value)}
              style={{
                marginBottom: "0.75rem",
                width: "100%",
                background: "rgba(10, 24, 46, 0.8)",
                color: "var(--text)",
                border: "1px solid var(--border)",
                borderRadius: "var(--radius-md)",
                padding: "0.45rem 0.75rem",
                fontSize: "0.84rem",
              }}
            />
            <input
              type="file"
              accept=".zip"
              disabled={upload.isPending || isDemoLoading}
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) {
                  setIsDemoActive(false);
                  upload.mutate(f);
                }
              }}
              style={{ width: "100%" }}
            />
            <ErrorBox error={upload.error} />
          </div>

          {upload.data && (
            <div style={{ display: "flex", flexDirection: "column", gap: "0.85rem" }}>
              <div
                style={{
                  padding: "0.75rem",
                  background: isNavPresent ? "rgba(0, 242, 254, 0.05)" : "rgba(239, 68, 68, 0.05)",
                  borderRadius: "var(--radius-md)",
                  border: isNavPresent ? "1px solid var(--border)" : "1px solid rgba(239, 68, 68, 0.3)",
                }}
              >
                <div className="muted mono" style={{ fontSize: "0.82rem" }}>
                  {upload.data.survey_id} · {upload.data.image_count} image(s) · navigation: <strong>{navStatus}</strong>
                </div>
                {!isNavPresent && (
                  <p style={{ color: "var(--warn)", fontSize: "0.78rem", margin: "0.4rem 0 0" }}>
                    ⚠ No usable navigation metadata — detections will have null coordinates (never fabricated).
                  </p>
                )}
                {upload.data.navigation_warnings.length > 0 && (
                  <p style={{ color: "var(--warn)", fontSize: "0.78rem", margin: "0.4rem 0 0" }}>
                    ⚠ {upload.data.navigation_warnings.join("; ")}
                  </p>
                )}
              </div>

              <div>
                <h2>2 · Run batch detection</h2>
                <button
                  style={{ width: "100%", marginTop: "0.35rem" }}
                  disabled={batch.isPending || job?.status === "running"}
                  onClick={() => batch.mutate(upload.data.survey_id)}
                >
                  {batch.isPending || job?.status === "running" ? "Running Batch Detection…" : "Run batch detection"}
                </button>
                <ErrorBox error={batch.error} />
              </div>

              {job && (
                <div
                  className="muted mono"
                  style={{
                    fontSize: "0.82rem",
                    padding: "0.6rem 0.8rem",
                    background: "rgba(10, 24, 46, 0.5)",
                    borderRadius: "var(--radius-md)",
                  }}
                >
                  job {job.job_id} · {job.status} · {job.progress_pct}%
                  {job.errors.length > 0 && (
                    <span style={{ color: "var(--warn)" }}>
                      {" "}· {job.errors.length} image error(s)
                    </span>
                  )}
                </div>
              )}

              {job?.status === "failed" && (
                <div className="error-box">{JSON.stringify(job.errors)}</div>
              )}
            </div>
          )}

          {/* Educational Information Card */}
          <div
            style={{
              padding: "0.75rem 0.9rem",
              background: "rgba(10, 24, 46, 0.4)",
              borderRadius: "var(--radius-md)",
              border: "1px solid rgba(255, 255, 255, 0.08)",
              fontSize: "0.78rem",
              lineHeight: 1.45,
            }}
          >
            <div style={{ fontWeight: 700, color: "#38bdf8", marginBottom: "0.35rem" }}>
              🌐 HOW GEOLOCATION WORKS
            </div>
            <ol style={{ margin: 0, paddingLeft: "1.1rem", color: "var(--text-secondary)" }}>
              <li>Sonar tiles are matched with survey navigation logs (<code>nav.csv</code>).</li>
              <li>Navigation timestamps and GPS fixes establish the ship trackline.</li>
              <li>Along-track linear interpolation computes exact detection coordinates.</li>
              <li>Coordinate uncertainty bounds (±X.X m) are assigned to each detection.</li>
              <li><strong>Zero Guessing:</strong> If nav telemetry is absent, coordinates stay null.</li>
            </ol>
          </div>

          {/* Demo Provenance Panel */}
          {isDemoActive && (
            <div
              style={{
                padding: "0.75rem 0.9rem",
                background: "rgba(168, 85, 247, 0.08)",
                borderRadius: "var(--radius-md)",
                border: "1px solid rgba(168, 85, 247, 0.25)",
                fontSize: "0.78rem",
                lineHeight: 1.45,
              }}
            >
              <div style={{ fontWeight: 700, color: "#c084fc", marginBottom: "0.35rem" }}>
                🛡 DEMO SURVEY PROVENANCE
              </div>
              <div style={{ color: "var(--text-secondary)" }}>
                <div><strong>Dataset:</strong> DRISHTI-SSS Arabian Sea Pipeline survey</div>
                <div><strong>Navigation:</strong> nav.csv (along-track sensor telemetry)</div>
                <div><strong>Model:</strong> drishti-ss_yolov8n_e30_final</div>
                <div><strong>Mode:</strong> Real Batch Inference</div>
                <div><strong>Authenticity:</strong> Real YOLOv8n inference on verified tiles</div>
              </div>
            </div>
          )}
        </section>

        {/* Right Column: Geospatial Results, Map & Tile Inspection */}
        <section
          className="grow"
          style={{ flex: 2, minWidth: 0, display: "flex", flexDirection: "column", gap: "1.25rem" }}
        >
          {/* Geospatial Map Panel */}
          <div className="panel">
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: "0.5rem", marginBottom: "0.5rem" }}>
              <h2 style={{ margin: 0 }}>3 · Geospatial results</h2>
              {job?.status === "succeeded" && (
                <div style={{ fontSize: "0.82rem" }}>
                  <span className="badge accepted" style={{ marginRight: "0.4rem" }}>
                    {geo.length} geolocated
                  </span>
                  <span className="muted">
                    {navStatus === "present" ? "method: linear_interp_along_track" : "no navigation metadata"}
                  </span>
                </div>
              )}
            </div>

            {!upload.data && (
              <div style={{ padding: "1.5rem 0", textAlign: "center" }}>
                <p className="muted" style={{ margin: "0 0 1rem" }}>
                  Upload a survey archive or click &ldquo;Load Demo Survey&rdquo; above to run batch detection. Uniquely, this is
                  the path where detections can carry real coordinates — because a survey is the only input that can bring navigation metadata.
                </p>
                <button
                  type="button"
                  onClick={handleLoadDemoSurvey}
                  disabled={isDemoLoading}
                  style={{
                    padding: "0.5rem 1.1rem",
                    fontSize: "0.85rem",
                    background: "rgba(0, 242, 254, 0.12)",
                    color: "#00f2fe",
                    border: "1px solid rgba(0, 242, 254, 0.4)",
                    borderRadius: "var(--radius-md)",
                    cursor: "pointer",
                  }}
                >
                  🚀 Click to Load Verified Demo Survey
                </button>
              </div>
            )}

            {upload.data && job?.status !== "succeeded" && (
              <p className="muted" style={{ padding: "1rem 0" }}>
                Run the batch to populate detections for this survey.
              </p>
            )}

            {job?.status === "succeeded" && (
              <>
                <p className="muted" style={{ fontSize: "0.85rem", marginBottom: "0.85rem" }}>
                  {detections.length} detection(s) · {geo.length} with coordinates
                  {navStatus === "present" ? (
                    <>
                      {" "}· method <code>linear_interp_along_track</code> (uncertainty recorded per detection)
                    </>
                  ) : (
                    <> · no navigation track, so coordinates stay null</>
                  )}
                </p>

                {/* Interactive Map Component */}
                <MapView
                  detections={detections}
                  selectedId={selectedDetectionId}
                  onSelect={(d) => setSelectedDetectionId(d.detection_id)}
                  height={420}
                />

                {/* Survey Trajectory Legend */}
                {geo.length > 0 && (
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: "1.25rem",
                      fontSize: "0.76rem",
                      marginTop: "0.6rem",
                      color: "var(--text-secondary)",
                      flexWrap: "wrap",
                    }}
                  >
                    <div style={{ display: "flex", alignItems: "center", gap: "0.35rem" }}>
                      <span style={{ display: "inline-block", width: 14, height: 3, background: "#00f2fe" }} />
                      <span>Survey Navigation Track (nav.csv)</span>
                    </div>
                    <div style={{ display: "flex", alignItems: "center", gap: "0.35rem" }}>
                      <span style={{ display: "inline-block", width: 10, height: 10, borderRadius: "50%", background: "#10b981" }} />
                      <span>Accepted Detection</span>
                    </div>
                    <div style={{ display: "flex", alignItems: "center", gap: "0.35rem" }}>
                      <span style={{ display: "inline-block", width: 10, height: 10, borderRadius: "50%", background: "#f59e0b" }} />
                      <span>Flagged (Edge-clipping penalty)</span>
                    </div>
                  </div>
                )}
              </>
            )}
            <ErrorBox error={detQ.error} />
          </div>

          {/* ================= TILE INSPECTION PANEL ================= */}
          {job?.status === "succeeded" && selectedDetection && (
            <div
              className="panel"
              style={{
                border: "1px solid rgba(0, 242, 254, 0.3)",
                background: "linear-gradient(135deg, rgba(10, 24, 46, 0.8) 0%, rgba(6, 17, 34, 0.95) 100%)",
              }}
            >
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "0.75rem", flexWrap: "wrap", gap: "0.5rem" }}>
                <div>
                  <h3 style={{ margin: 0, fontSize: "1.05rem", color: "#fff" }}>
                    🔍 Sonar Tile Inspection: <span style={{ color: "#00f2fe" }}>{selectedDetection.class_name}</span>
                  </h3>
                  <span className="muted mono" style={{ fontSize: "0.75rem" }}>
                    Image: {selectedDetection.image_id} · Detection: {selectedDetection.detection_id}
                  </span>
                </div>
                <div style={{ display: "flex", gap: "0.4rem" }}>
                  <span className={`badge ${selectedDetection.filtering_status}`}>
                    {selectedDetection.filtering_status}
                  </span>
                </div>
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: "1rem" }}>
                {/* Sonar Tile Viewer with Bounding Box Overlay */}
                <div style={{ minHeight: 280, display: "flex", flexDirection: "column" }}>
                  <div style={{ fontSize: "0.74rem", fontWeight: 700, textTransform: "uppercase", color: "var(--text-secondary)", marginBottom: "0.3rem" }}>
                    Acoustic Backscatter & Bounding Box Overlay
                  </div>
                  <div style={{ flex: 1, minHeight: 260, borderRadius: "var(--radius-md)", overflow: "hidden", border: "1px solid var(--border)" }}>
                    <SonarViewer
                      imageUrl={imageUrl(selectedDetection.image_id)}
                      detections={[selectedDetection]}
                      coordSpace="source"
                      selectedId={selectedDetection.detection_id}
                    />
                  </div>
                </div>

                {/* Inspection Details Card */}
                <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem", fontSize: "0.82rem" }}>
                  <div style={{ padding: "0.75rem", background: "rgba(10, 24, 46, 0.6)", borderRadius: "var(--radius-md)", border: "1px solid var(--border)" }}>
                    <div style={{ fontWeight: 700, color: "#38bdf8", marginBottom: "0.4rem" }}>
                      DETECTION & CLASSIFICATION
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.3rem" }}>
                      <span className="muted">Target Object:</span>
                      <strong style={{ color: "#fff" }}>{selectedDetection.class_name}</strong>
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.3rem" }}>
                      <span className="muted">Raw Confidence:</span>
                      <span className="mono">{(selectedDetection.model_confidence * 100).toFixed(1)}%</span>
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.3rem" }}>
                      <span className="muted">Final Confidence:</span>
                      <span className="mono" style={{ fontWeight: 700, color: "#10b981" }}>
                        {(selectedDetection.final_confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between" }}>
                      <span className="muted">Status:</span>
                      <span style={{ textTransform: "capitalize", fontWeight: 600 }}>{selectedDetection.filtering_status}</span>
                    </div>
                    {selectedDetection.filter_reasons && selectedDetection.filter_reasons.length > 0 && (
                      <div style={{ marginTop: "0.4rem", color: "#f59e0b", fontSize: "0.76rem" }}>
                        <strong>Filter Reason:</strong> {selectedDetection.filter_reasons.join("; ")}
                      </div>
                    )}
                  </div>

                  <div style={{ padding: "0.75rem", background: "rgba(10, 24, 46, 0.6)", borderRadius: "var(--radius-md)", border: "1px solid var(--border)" }}>
                    <div style={{ fontWeight: 700, color: "#a855f7", marginBottom: "0.4rem" }}>
                      GEOSPATIAL POSITIONING
                    </div>
                    {selectedDetection.latitude != null && selectedDetection.longitude != null ? (
                      <>
                        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.3rem" }}>
                          <span className="muted">Latitude:</span>
                          <span className="mono">{selectedDetection.latitude.toFixed(5)}° N</span>
                        </div>
                        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.3rem" }}>
                          <span className="muted">Longitude:</span>
                          <span className="mono">{selectedDetection.longitude.toFixed(5)}° E</span>
                        </div>
                        {selectedDetection.geo_provenance?.uncertainty_m != null && (
                          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.3rem" }}>
                            <span className="muted">Position Uncertainty:</span>
                            <span className="mono">±{selectedDetection.geo_provenance.uncertainty_m} m</span>
                          </div>
                        )}
                        <div style={{ display: "flex", justifyContent: "space-between" }}>
                          <span className="muted">Navigation Source:</span>
                          <span style={{ color: "#38bdf8" }}>nav.csv sidecar (real track)</span>
                        </div>
                      </>
                    ) : (
                      <div style={{ color: "var(--warn)" }}>
                        Location unavailable — no navigation metadata provided.
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* ================= DETECTIONS TABLE ================= */}
          {job?.status === "succeeded" && detections.length > 0 && (
            <div className="panel">
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "0.75rem" }}>
                <h2 style={{ margin: 0 }}>Detections ({detections.length})</h2>
                <span className="muted" style={{ fontSize: "0.78rem" }}>
                  Click any row to inspect acoustic backscatter tile and bounding box
                </span>
              </div>
              <div style={{ overflowX: "auto" }}>
                <table>
                  <thead>
                    <tr>
                      <th>Image</th>
                      <th>Class</th>
                      <th>Confidence</th>
                      <th>Location</th>
                      <th>Geo status</th>
                      <th>Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {detections.map((d) => {
                      const isRowSelected = selectedDetectionId === d.detection_id;
                      return (
                        <tr
                          key={d.detection_id}
                          onClick={() => setSelectedDetectionId(d.detection_id)}
                          style={{
                            cursor: "pointer",
                            background: isRowSelected ? "rgba(0, 242, 254, 0.08)" : undefined,
                            borderLeft: isRowSelected ? "3px solid #00f2fe" : "3px solid transparent",
                          }}
                        >
                          <td className="muted mono" title={d.image_id} style={{ fontSize: "0.78rem" }}>
                            {d.image_id.slice(0, 16)}…
                          </td>
                          <td>
                            <strong style={{ color: "#fff" }}>{d.class_name}</strong>
                          </td>
                          <td>
                            <ConfidenceIndicator detection={d} />
                          </td>
                          <td>
                            {d.latitude != null && d.longitude != null ? (
                              <span
                                className="mono"
                                style={{ fontSize: "0.8rem" }}
                                title={d.geo_provenance?.method ?? "unknown method"}
                              >
                                {d.latitude.toFixed(5)}, {d.longitude.toFixed(5)}
                                {d.geo_provenance?.uncertainty_m != null && (
                                  <span className="muted"> ±{d.geo_provenance.uncertainty_m}m</span>
                                )}
                              </span>
                            ) : (
                              <span className="muted">unavailable</span>
                            )}
                          </td>
                          <td className="muted mono" style={{ fontSize: "0.78rem" }}>
                            {d.geo_status}
                          </td>
                          <td>
                            <button
                              type="button"
                              className={isRowSelected ? "" : "secondary"}
                              style={{ padding: "0.25rem 0.6rem", fontSize: "0.75rem" }}
                              onClick={(e) => {
                                e.stopPropagation();
                                setSelectedDetectionId(d.detection_id);
                              }}
                            >
                              {isRowSelected ? "Selected" : "Inspect"}
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
