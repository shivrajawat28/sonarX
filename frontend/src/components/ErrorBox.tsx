import { ApiError } from "../api/client";

/** Structured error display: code + message (+ request id for traceability). */
export default function ErrorBox({ error, fallbackMessage }: { error: unknown; fallbackMessage?: string }) {
  if (!error) return null;
  const isApi = error instanceof ApiError;
  const rawMsg = (error as Error)?.message || "";
  const code = isApi ? (error as ApiError).code : "ERROR";

  let friendlyMessage = fallbackMessage || rawMsg;
  if (/failed to fetch|network|econnrefused/i.test(rawMsg)) {
    friendlyMessage = "Backend connection unavailable.";
  } else if (code === "UNSUPPORTED_FILE_TYPE" || code === "INVALID_IMAGE" || /unsupported|undecodable|format/i.test(rawMsg)) {
    friendlyMessage = "Unsupported sonar image format.";
  } else if (code === "MODEL_UNAVAILABLE" || code === "DETECTOR_PREDICT_ERROR" || /inference failed|model/i.test(rawMsg)) {
    friendlyMessage = "AI inference could not be completed.";
  }

  return (
    <div className="error-box" role="alert" style={{ marginTop: "0.5rem" }}>
      <div style={{ fontWeight: 600 }}>
        <strong>{code}</strong>: {friendlyMessage}
      </div>
      {rawMsg && rawMsg !== friendlyMessage && (
        <div className="muted" style={{ fontSize: "0.74rem", marginTop: "0.2rem" }}>
          {rawMsg}
        </div>
      )}
      {isApi && (error as ApiError).requestId && (
        <div className="muted" style={{ fontSize: "0.72rem", marginTop: "0.15rem" }}>
          request_id: {(error as ApiError).requestId}
        </div>
      )}
    </div>
  );
}
