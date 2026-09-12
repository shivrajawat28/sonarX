import { ApiError } from "../api/client";

/** Structured error display: code + message (+ request id for traceability). */
export default function ErrorBox({ error }: { error: unknown }) {
  if (!error) return null;
  const isApi = error instanceof ApiError;
  return (
    <div className="error-box" role="alert">
      <strong>{isApi ? (error as ApiError).code : "CLIENT_ERROR"}</strong>{" "}
      {(error as Error).message}
      {isApi && (error as ApiError).requestId && (
        <div className="muted">request_id: {(error as ApiError).requestId}</div>
      )}
    </div>
  );
}
