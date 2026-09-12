import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  CoordSpace,
  Detection,
  InferenceResult,
  PreprocessPreview,
  UploadedImage,
  originalImageUrl,
  preprocessPreview,
  processedImageUrl,
  runDetection,
  uploadImage,
  exportCsvUrl,
  exportJsonUrl,
} from "../api/client";
import ErrorBox from "../components/ErrorBox";
import SonarViewer from "../components/SonarViewer";
import DetectionDetails from "../components/DetectionDetails";
import MapView from "../components/MapView";
import ReportActions from "../components/ReportActions";

type Phase = "idle" | "uploaded" | "previewed" | "detected";

/**
 * Core demo flow (Section 14): upload → preprocessing preview → run detection
 * → viewer + details + map → export. All intelligence lives in the backend.
 */
export default function WorkbenchPage() {
  const qc = useQueryClient();
  const [, setPhase] = useState<Phase>("idle");
  const [upload, setUpload] = useState<UploadedImage | null>(null);
  const [preview, setPreview] = useState<PreprocessPreview | null>(null);
  const [result, setResult] = useState<InferenceResult | null>(null);
  const [viewProcessed, setViewProcessed] = useState(true);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const uploadMut = useMutation<UploadedImage, unknown, File>({
    mutationFn: (file) => uploadImage(file),
    onSuccess: (u) => {
      setUpload(u);
      setPreview(null);
      setResult(null);
      setSelectedId(null);
      setPhase("uploaded");
    },
  });

  const previewMut = useMutation<PreprocessPreview, unknown, string>({
    mutationFn: (imageId) => preprocessPreview(imageId),
    onSuccess: (p) => {
      setPreview(p);
      setPhase("previewed");
    },
  });

  const detectMut = useMutation<InferenceResult, unknown, string>({
    mutationFn: (imageId) => runDetection(imageId, true),
    onSuccess: (r) => {
      setResult(r);
      setSelectedId(null);
      setPhase("detected");
      qc.invalidateQueries({ queryKey: ["detections"] });
    },
  });

  const busy =
    uploadMut.isPending || previewMut.isPending || detectMut.isPending;

  const detections: Detection[] = result?.detections ?? [];
  const geoCount = detections.filter(
    (d) => d.latitude != null && d.longitude != null
  ).length;
  // The processed image only exists after a run; before that we show the
  // original upload. The overlay must use the space of whatever is displayed.
  const showProcessed = viewProcessed && result != null;
  const coordSpace: CoordSpace = showProcessed ? "processed" : "source";
  const statusCounts = detections.reduce<Record<string, number>>((acc, d) => {
    acc[d.filtering_status] = (acc[d.filtering_status] ?? 0) + 1;
    return acc;
  }, {});

  return (
    <div className="row">
      {/* ---- left: upload + actions ---- */}
      <section className="panel grow" style={{ maxWidth: 340 }}>
        <h2>1 · Upload sonar image</h2>
        <input
          type="file"
          accept=".png,.jpg,.jpeg,.tif,.tiff"
          disabled={busy}
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) uploadMut.mutate(f);
          }}
        />
        <ErrorBox error={uploadMut.error} />

        {upload && (
          <p className="muted">
            {upload.filename} · {upload.width}×{upload.height} ·{" "}
            {upload.format.toUpperCase()} · sha {upload.sha256.slice(0, 12)}…
          </p>
        )}

        {upload && (
          <>
            <h2 style={{ marginTop: "1rem" }}>2 · Preprocessing preview</h2>
            <button
              className="secondary"
              disabled={busy}
              onClick={() => previewMut.mutate(upload.image_id)}
            >
              {previewMut.isPending ? "Preprocessing…" : "Run preview"}
            </button>
            <ErrorBox error={previewMut.error} />
            {preview && (
              <p className="muted">
                config {preview.config_hash.slice(0, 12)}… · ops:{" "}
                {preview.applied_ops.map((o) => o.op).join(" → ")}
                {preview.warnings.length > 0 && (
                  <span style={{ color: "var(--warn)" }}>
                    {" "}
                    ⚠ {preview.warnings.join("; ")}
                  </span>
                )}
              </p>
            )}

            <h2 style={{ marginTop: "1rem" }}>3 · Run AI detection</h2>
            <button
              disabled={busy}
              onClick={() => detectMut.mutate(upload.image_id)}
            >
              {detectMut.isPending ? "Detecting…" : "Run detection"}
            </button>
            <ErrorBox error={detectMut.error} />
          </>
        )}
      </section>

      {/* ---- right: viewer + results ---- */}
      <section className="grow" style={{ flex: 2, minWidth: 0 }}>
        {upload ? (
          <div className="panel">
            <div className="row" style={{ alignItems: "center" }}>
              <h2 style={{ marginRight: "auto" }}>Sonar viewer</h2>
              <label className="muted">
                <input
                  type="checkbox"
                  checked={viewProcessed}
                  onChange={(e) => setViewProcessed(e.target.checked)}
                />{" "}
                preprocessed
              </label>
            </div>
            <SonarViewer
              imageUrl={
                showProcessed
                  ? processedImageUrl(upload.image_id)
                  : originalImageUrl(upload.image_id)
              }
              detections={detections}
              coordSpace={coordSpace}
              selectedId={selectedId}
              onSelect={setSelectedId}
            />
            {result && (
              <p className="muted" style={{ fontSize: "0.8rem" }}>
                <span style={{ color: "#2ea043" }}>■ accepted</span>{" "}
                <span style={{ color: "#d29922" }}>■ flagged</span>{" "}
                <span style={{ color: "#f85149" }}>■ rejected (kept, dashed)</span>{" "}
                · click a box or a detection row to highlight it
              </p>
            )}
            {result && (
              <p className="muted">
                model {result.model_version} · preprocess{" "}
                {result.preprocess_config_hash.slice(0, 12)}… · timings{" "}
                {Object.entries(result.timings_ms)
                  .map(([k, v]) => `${k} ${Math.round(v as number)}ms`)
                  .join(", ")}
              </p>
            )}
            {result && result.warnings.length > 0 && (
              <p style={{ color: "var(--warn)", fontSize: "0.8rem" }}>
                {result.warnings.join(" · ")}
              </p>
            )}
          </div>
        ) : (
          <div className="panel muted">
            Upload a sonar image to begin. The system will preprocess it, run the
            registered detector, apply rule-based false-positive filtering, and
            show results here.
          </div>
        )}

        {result && (
          <>
            <div className="panel" style={{ marginTop: "1rem" }}>
              <h2>Detections ({detections.length})</h2>
              {detections.length > 0 && (
                <p className="muted" style={{ fontSize: "0.8rem" }}>
                  {statusCounts.accepted ?? 0} accepted ·{" "}
                  {statusCounts.flagged ?? 0} flagged · {statusCounts.rejected ?? 0}{" "}
                  rejected — filtering annotates, it never deletes a detection.
                </p>
              )}
              <DetectionDetails
                detections={detections}
                coordSpace={coordSpace}
                selectedId={selectedId}
                onSelect={setSelectedId}
              />
            </div>

            <div className="panel" style={{ marginTop: "1rem" }}>
              <h2>Map {geoCount > 0 ? `(${geoCount} geolocated)` : ""}</h2>
              <MapView detections={detections} />
            </div>

            <div className="panel" style={{ marginTop: "1rem" }}>
              <h2>Exports & report</h2>
              <a href={exportCsvUrl()} download>
                <button className="secondary">Download CSV</button>
              </a>{" "}
              <a href={exportJsonUrl()} download>
                <button className="secondary">Download JSON</button>
              </a>
              <div style={{ marginTop: "0.75rem" }}>
                <ReportActions runId={result.detection_run_id} />
              </div>
            </div>
          </>
        )}
      </section>
    </div>
  );
}
