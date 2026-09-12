import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Detection,
  JobInfo,
  exportCsvUrl,
  exportJsonUrl,
  getJob,
  getSurvey,
  listDetections,
  runSurveyBatch,
  uploadSurvey,
} from "../api/client";
import ErrorBox from "../components/ErrorBox";
import MapView from "../components/MapView";
import ConfidenceIndicator from "../components/ConfidenceIndicator";

/**
 * Survey (batch) mode — the geolocation path.
 *
 * A single-image upload carries no navigation data, so its detections
 * legitimately have null coordinates. Geolocation is only possible for a SURVEY
 * where a navigation sidecar (nav.csv / navigation.csv) was parsed at upload:
 * the backend interpolates each tile's along-track position from the real track.
 *
 * Everything on this page is read from the API. When navigation is absent or
 * unparseable the page says so and the map stays empty — coordinates are never
 * invented (Section 10).
 */
export default function SurveyPage() {
  const qc = useQueryClient();
  const [name, setName] = useState("");
  const [jobId, setJobId] = useState<string | null>(null);

  const upload = useMutation({
    mutationFn: (file: File) => uploadSurvey(file, name),
    onSuccess: () => {
      setJobId(null);
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

  return (
    <div className="row">
      <section className="panel grow" style={{ maxWidth: 360 }}>
        <h2>1 · Upload survey archive</h2>
        <p className="muted" style={{ fontSize: "0.8rem" }}>
          A .zip of sonar tiles, optionally with a <code>nav.csv</code> /
          <code>navigation.csv</code> sidecar (columns: timestamp, latitude,
          longitude). Coordinates come only from that file.
        </p>
        <input
          placeholder="survey name (optional)"
          value={name}
          onChange={(e) => setName(e.target.value)}
          style={{ marginBottom: "0.5rem", width: "100%" }}
        />
        <input
          type="file"
          accept=".zip"
          disabled={upload.isPending}
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) upload.mutate(f);
          }}
        />
        <ErrorBox error={upload.error} />

        {upload.data && (
          <div>
            <p className="muted" style={{ fontSize: "0.85rem" }}>
              {upload.data.survey_id} · {upload.data.image_count} image(s) ·
              navigation: <strong>{navStatus}</strong>
            </p>
            {navStatus !== "present" && (
              <p style={{ color: "var(--warn)", fontSize: "0.8rem" }}>
                ⚠ No usable navigation metadata — detections will have null
                coordinates (never fabricated).
              </p>
            )}
            {upload.data.navigation_warnings.length > 0 && (
              <p style={{ color: "var(--warn)", fontSize: "0.8rem" }}>
                ⚠ {upload.data.navigation_warnings.join("; ")}
              </p>
            )}

            <h2 style={{ marginTop: "1rem" }}>2 · Run batch detection</h2>
            <button
              disabled={batch.isPending || job?.status === "running"}
              onClick={() => batch.mutate(upload.data.survey_id)}
            >
              {batch.isPending ? "Starting…" : "Run batch detection"}
            </button>
            <ErrorBox error={batch.error} />

            {job && (
              <p className="muted" style={{ fontSize: "0.85rem" }}>
                job {job.job_id} · {job.status} · {job.progress_pct}%
                {job.errors.length > 0 && (
                  <span style={{ color: "var(--warn)" }}>
                    {" "}
                    · {job.errors.length} image error(s)
                  </span>
                )}
              </p>
            )}
            {job?.status === "failed" && (
              <div className="error-box">{JSON.stringify(job.errors)}</div>
            )}
          </div>
        )}
      </section>

      <section className="grow" style={{ flex: 2, minWidth: 0 }}>
        <div className="panel">
          <h2>3 · Geospatial results</h2>
          {!upload.data && (
            <p className="muted">
              Upload a survey archive to run batch detection. Uniquely, this is
              the path where detections can carry real coordinates — because a
              survey is the only input that can bring navigation metadata.
            </p>
          )}
          {upload.data && job?.status !== "succeeded" && (
            <p className="muted">
              Run the batch to populate detections for this survey.
            </p>
          )}
          {job?.status === "succeeded" && (
            <>
              <p className="muted" style={{ fontSize: "0.85rem" }}>
                {detections.length} detection(s) · {geo.length} with coordinates
                {navStatus === "present" ? (
                  <>
                    {" "}
                    · method{" "}
                    <code>linear_interp_along_track</code> (uncertainty recorded
                    per detection)
                  </>
                ) : (
                  <> · no navigation track, so coordinates stay null</>
                )}
              </p>

              <MapView detections={detections} />

              <div style={{ marginTop: "0.75rem" }}>
                <a href={exportCsvUrl(surveyId ?? undefined)} download>
                  <button className="secondary">Download CSV (this survey)</button>
                </a>{" "}
                <a href={exportJsonUrl(surveyId ?? undefined)} download>
                  <button className="secondary">Download JSON (this survey)</button>
                </a>
              </div>
            </>
          )}
          <ErrorBox error={detQ.error} />
        </div>

        {job?.status === "succeeded" && detections.length > 0 && (
          <div className="panel" style={{ marginTop: "1rem" }}>
            <h2>Detections ({detections.length})</h2>
            <table>
              <thead>
                <tr>
                  <th>Image</th>
                  <th>Class</th>
                  <th>Confidence</th>
                  <th>Location</th>
                  <th>Geo status</th>
                </tr>
              </thead>
              <tbody>
                {detections.map((d) => (
                  <tr key={d.detection_id}>
                    <td className="muted" title={d.image_id}>
                      {d.image_id.slice(0, 14)}…
                    </td>
                    <td>{d.class_name}</td>
                    <td>
                      <ConfidenceIndicator detection={d} />
                    </td>
                    <td>
                      {d.latitude != null && d.longitude != null ? (
                        <span
                          title={d.geo_provenance?.method ?? "unknown method"}
                        >
                          {d.latitude.toFixed(5)}, {d.longitude.toFixed(5)}
                          {d.geo_provenance?.uncertainty_m != null && (
                            <span className="muted">
                              {" "}
                              ±{d.geo_provenance.uncertainty_m}m
                            </span>
                          )}
                        </span>
                      ) : (
                        <span className="muted">unavailable</span>
                      )}
                    </td>
                    <td className="muted">{d.geo_status}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
