import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  ApiError,
  Detection,
  getJob,
  listDetections,
  overrideDetection,
} from "../api/client";
import ErrorBox from "../components/ErrorBox";

/**
 * Detection history with filters and analyst status override (display-only —
 * the backend owns all status semantics).
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
    <div className="panel">
      <div className="row" style={{ alignItems: "center", marginBottom: "0.75rem" }}>
        <h2 style={{ marginRight: "auto" }}>Detection history</h2>
        <input
          placeholder="filter by survey_id"
          value={surveyId}
          onChange={(e) => setSurveyId(e.target.value)}
        />
        <select value={status} onChange={(e) => setStatus(e.target.value)}>
          <option value="">any status</option>
          <option value="accepted">accepted</option>
          <option value="flagged">flagged</option>
          <option value="rejected">rejected</option>
        </select>
      </div>

      {isLoading && <p className="muted">loading…</p>}
      <ErrorBox error={error} />

      {data && (
        <>
          <p className="muted">{data.total} detections</p>
          <HistoryTable items={data.items} />
        </>
      )}
    </div>
  );
}

function HistoryTable({ items }: { items: Detection[] }) {
  const qc = useQueryClient();
  const override = useMutation({
    mutationFn: (vars: { id: string; status: string }) => overrideDetection(vars.id, vars.status),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["detections"] }),
  });

  if (items.length === 0) return <p className="muted">No detections match.</p>;

  return (
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
            <td className="muted">{d.created_at?.slice(0, 16).replace("T", " ") ?? "—"}</td>
            <td className="muted" title={d.image_id}>
              {d.image_id.slice(0, 14)}…
            </td>
            <td>{d.class_name}</td>
            <td className="muted">
              {(d.final_confidence * 100).toFixed(0)}% final
            </td>
            <td>
              {d.latitude != null && d.longitude != null
                ? `${d.latitude.toFixed(4)}, ${d.longitude.toFixed(4)}`
                : "—"}
            </td>
            <td>
              <span className={`badge ${d.filtering_status}`}>{d.filtering_status}</span>
              {d.analyst_overridden && <span className="muted"> (reviewed)</span>}
            </td>
            <td>
              <select
                value={d.filtering_status}
                onChange={(e) => override.mutate({ id: d.detection_id, status: e.target.value })}
                disabled={override.isPending}
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
  );
}

export function JobPoller({ jobId }: { jobId: string | null }) {
  const { data: job, error } = useQuery({
    queryKey: ["job", jobId],
    queryFn: () => getJob(jobId as string),
    enabled: jobId != null,
    refetchInterval: (q) =>
      q.state.data && ["succeeded", "failed"].includes(q.state.data.status) ? false : 1000,
  });
  if (!jobId) return null;
  if (error) return <ErrorBox error={error instanceof ApiError ? error : error} />;
  if (!job) return <p className="muted">creating job…</p>;
  return (
    <div className="muted">
      job {job.job_id} · {job.status} · {job.progress_pct}%
      {job.status === "failed" && job.errors.length > 0 && (
        <div className="error-box">{JSON.stringify(job.errors)}</div>
      )}
    </div>
  );
}
