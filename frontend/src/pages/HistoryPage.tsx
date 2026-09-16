import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Detection,
  getJob,
  listDetections,
  overrideDetection,
} from "../api/client";
import ErrorBox from "../components/ErrorBox";

/**
 * Detection history with filters and analyst status override (display-only).
 */
export default function HistoryPage() {
  const [surveyId, setSurveyId] = useState("");
  const [status, setStatus] = useState("");

  const { data, isLoading, error } = useQuery({
    queryKey: ["detections", surveyId, status],
    queryFn: () => listDetections({ survey_id: surveyId || undefined, status: status || undefined, size: 100 }),
    refetchInterval: 10000,
  });

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1.25rem" }}>
      <div>
        <h1 style={{ margin: 0, fontSize: "1.6rem" }}>Acoustic Detection History</h1>
        <p className="muted" style={{ margin: "0.25rem 0 0", fontSize: "0.88rem" }}>
          Full audit trail of AI detections, confidence filtering, and analyst reviews.
        </p>
      </div>

      <div className="panel">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "1rem", marginBottom: "1rem" }}>
          <h2 style={{ margin: 0 }}>Detection history</h2>
          <div style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap" }}>
            <input
              placeholder="filter by survey_id"
              value={surveyId}
              onChange={(e) => setSurveyId(e.target.value)}
              style={{
                background: "rgba(10, 24, 46, 0.8)",
                color: "var(--text)",
                border: "1px solid var(--border)",
                borderRadius: "var(--radius-md)",
                padding: "0.4rem 0.75rem",
                fontSize: "0.84rem",
              }}
            />
            <select
              value={status}
              onChange={(e) => setStatus(e.target.value)}
              style={{
                background: "rgba(10, 24, 46, 0.8)",
                color: "var(--text)",
                border: "1px solid var(--border)",
                borderRadius: "var(--radius-md)",
                padding: "0.4rem 0.75rem",
                fontSize: "0.84rem",
              }}
            >
              <option value="">any status</option>
              <option value="accepted">accepted</option>
              <option value="flagged">flagged</option>
              <option value="rejected">rejected</option>
            </select>
          </div>
        </div>

        {isLoading && <p className="muted">loading…</p>}
        <ErrorBox error={error} />

        {data && (
          <>
            <div className="muted" style={{ fontSize: "0.82rem", marginBottom: "0.85rem" }}>
              <strong>{data.total}</strong> detections recorded in persistence store
            </div>
            <HistoryTable items={data.items} />
          </>
        )}
      </div>
    </div>
  );
}

function HistoryTable({ items }: { items: Detection[] }) {
  const qc = useQueryClient();
  const override = useMutation({
    mutationFn: (vars: { id: string; status: string }) => overrideDetection(vars.id, vars.status),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["detections"] }),
  });

  if (items.length === 0) return <p className="muted" style={{ padding: "1rem 0" }}>No detections match.</p>;

  return (
    <div style={{ overflowX: "auto" }}>
      <table>
        <thead>
          <tr>
            <th>When</th>
            <th>Image</th>
            <th>Class</th>
            <th>Confidence</th>
            <th>Location</th>
            <th>Status</th>
            <th>Override</th>
          </tr>
        </thead>
        <tbody>
          {items.map((d) => (
            <tr key={d.detection_id}>
              <td className="muted mono" style={{ fontSize: "0.78rem" }}>{d.created_at?.slice(0, 16).replace("T", " ") ?? "—"}</td>
              <td className="muted mono" title={d.image_id} style={{ fontSize: "0.78rem" }}>
                {d.image_id.slice(0, 14)}…
              </td>
              <td><strong style={{ color: "#fff" }}>{d.class_name}</strong></td>
              <td className="muted mono">
                {(d.final_confidence * 100).toFixed(0)}% final
              </td>
              <td className="mono" style={{ fontSize: "0.8rem" }}>
                {d.latitude != null && d.longitude != null
                  ? `${d.latitude.toFixed(4)}, ${d.longitude.toFixed(4)}`
                  : "—"}
              </td>
              <td>
                <span className={`badge ${d.filtering_status}`}>{d.filtering_status}</span>
                {d.analyst_overridden && <span className="muted" style={{ fontSize: "0.72rem" }}> (reviewed)</span>}
              </td>
              <td>
                <select
                  value={d.filtering_status}
                  disabled={override.isPending}
                  onChange={(e) => override.mutate({ id: d.detection_id, status: e.target.value })}
                  style={{
                    background: "rgba(10, 24, 46, 0.8)",
                    color: "var(--text)",
                    border: "1px solid var(--border)",
                    borderRadius: "var(--radius-sm)",
                    padding: "0.2rem 0.4rem",
                    fontSize: "0.78rem",
                  }}
                >
                  <option value="accepted">accepted</option>
                  <option value="flagged">flagged</option>
                  <option value="rejected">rejected</option>
                </select>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function JobPoller({ jobId }: { jobId: string }) {
  const { data: job } = useQuery({
    queryKey: ["job", jobId],
    queryFn: () => getJob(jobId),
    refetchInterval: (query) => {
      const s = query.state.data?.status;
      return s === "succeeded" || s === "failed" ? false : 500;
    },
  });
  if (!job) return null;
  return (
    <div className="muted" style={{ fontSize: "0.82rem", marginTop: "0.5rem" }}>
      Job {job.job_id.slice(0, 12)}… · <strong>{job.status}</strong> · {job.progress_pct}%
      {job.errors?.length ? ` (${job.errors.length} errors)` : ""}
    </div>
  );
}
