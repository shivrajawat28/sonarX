import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { createReport, getReport, reportDownloadUrl } from "../api/client";
import { JobPoller } from "../pages/HistoryPage";
import ErrorBox from "../components/ErrorBox";

/**
 * Report generation: POST /reports (background job) → poll → download the
 * stored HTML/PDF artifact. The report content is produced server-side from
 * persisted detections; the frontend only triggers and fetches it.
 */
export default function ReportActions({ runId }: { runId: string | null }) {
  const [jobId, setJobId] = useState<string | null>(null);
  const [reportId, setReportId] = useState<string | null>(null);

  const create = useMutation({
    mutationFn: () => createReport({ run_id: runId }),
    onSuccess: (r) => {
      setJobId(r.job_id);
      // Kick off completion polling explicitly — the mutation below is never
      // invoked automatically (bugfix: the download link previously never
      // appeared even though the report job succeeded server-side).
      poll.mutate(r.job_id);
    },
  });

  // When the job finishes, fetch the report id from the job result_ref.
  const poll = useMutation({
    // Takes the job id as an argument to avoid a stale-closure read of state.
    mutationFn: async (id: string) => {
      // JobPoller below shows progress; this polling fetches the final id.
      for (let i = 0; i < 60; i++) {
        await new Promise((r) => setTimeout(r, 500));
        const res = await fetch(`/api/v1/jobs/${id}`);
        const job = (await res.json()) as {
          status: string;
          result_ref: { report_id?: string } | null;
        };
        if (job.status === "succeeded" && job.result_ref?.report_id) {
          return job.result_ref.report_id;
        }
        if (job.status === "failed") throw new Error("report job failed");
      }
      throw new Error("report job timed out");
    },
    onSuccess: (rid) => setReportId(rid),
  });

  if (!runId) {
    return (
      <p className="muted">Run a saved detection first — reports summarize a persisted run.</p>
    );
  }

  return (
    <div>
      <button
        className="secondary"
        disabled={create.isPending || poll.isPending}
        onClick={() => {
          setReportId(null);
          create.mutate();
        }}
      >
        {create.isPending ? "Creating…" : "Generate report"}
      </button>
      <ErrorBox error={create.error ?? poll.error} />
      {jobId && !reportId && <JobPoller jobId={jobId} />}
      {reportId && (
        <p>
          Report ready —{" "}
          <a href={reportDownloadUrl(reportId, "html")} download>
            download HTML
          </a>{" "}
          ·{" "}
          <a href={reportDownloadUrl(reportId, "pdf")} download>
            download PDF
          </a>
        </p>
      )}
    </div>
  );
}

export async function fetchReport(reportId: string) {
  return getReport(reportId);
}
