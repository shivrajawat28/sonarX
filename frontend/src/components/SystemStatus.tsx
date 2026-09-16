import { useQuery } from "@tanstack/react-query";
import { getHealth } from "../api/client";

/**
 * Compact live telemetry area showing actual backend health & service state.
 * Values come directly from GET /api/v1/health.
 */
export default function SystemStatus() {
  const { data, isError, isLoading } = useQuery({
    queryKey: ["health"],
    queryFn: getHealth,
    refetchInterval: 10000,
  });

  const isOnline = !isError && !isLoading && data?.status === "ok";
  const isModelLoaded = isOnline && !!data?.model?.loaded;
  const isStorageOk = isOnline && !!data?.storage_ok;

  return (
    <div className="system-status-grid">
      <div className="system-status-card">
        <div>
          <div className="label">Backend API</div>
          <div className="value" style={{ color: isOnline ? "var(--ok)" : "var(--bad)" }}>
            <span className="status-dot" style={{ background: isOnline ? "#10b981" : "#ef4444", boxShadow: isOnline ? "0 0 8px #10b981" : "0 0 8px #ef4444" }} />
            {isOnline ? "Online" : isLoading ? "Connecting…" : "Offline"}
          </div>
        </div>
        <span className="muted" style={{ fontSize: "0.75rem" }}>:8000</span>
      </div>

      <div className="system-status-card">
        <div>
          <div className="label">AI Model</div>
          <div className="value" style={{ color: isModelLoaded ? "var(--ok)" : "var(--warn)" }}>
            <span className="status-dot" style={{ background: isModelLoaded ? "#10b981" : "#f59e0b", boxShadow: isModelLoaded ? "0 0 8px #10b981" : "0 0 8px #f59e0b" }} />
            {isModelLoaded ? "Loaded" : "Unloaded"}
          </div>
        </div>
        <span className="muted mono" style={{ fontSize: "0.72rem" }} title={data?.model?.version ?? ""}>
          {data?.model?.version ? "YOLOv8n" : "—"}
        </span>
      </div>

      <div className="system-status-card">
        <div>
          <div className="label">Inference Service</div>
          <div className="value" style={{ color: isModelLoaded ? "#38bdf8" : "var(--warn)" }}>
            <span className="status-dot" style={{ background: isModelLoaded ? "#38bdf8" : "#f59e0b", boxShadow: isModelLoaded ? "0 0 8px #38bdf8" : "0 0 8px #f59e0b" }} />
            {isModelLoaded ? "Ready" : "Degraded"}
          </div>
        </div>
        <span className="muted" style={{ fontSize: "0.75rem" }}>CPU</span>
      </div>

      <div className="system-status-card">
        <div>
          <div className="label">Storage System</div>
          <div className="value" style={{ color: isStorageOk ? "var(--ok)" : "var(--bad)" }}>
            <span className="status-dot" style={{ background: isStorageOk ? "#10b981" : "#ef4444", boxShadow: isStorageOk ? "0 0 8px #10b981" : "0 0 8px #ef4444" }} />
            {isStorageOk ? "Available" : "Unavailable"}
          </div>
        </div>
        <span className="muted" style={{ fontSize: "0.75rem" }}>data/</span>
      </div>
    </div>
  );
}
