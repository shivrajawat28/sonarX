import { useQuery } from "@tanstack/react-query";
import { getHealth } from "../api/client";

/** Backend/model health from GET /health — displayed, never inferred. */
export default function HealthBadge() {
  const { data, isLoading, isError } = useQuery({
    queryKey: ["health"],
    queryFn: getHealth,
    refetchInterval: 15000,
  });

  if (isLoading) return <span className="muted">checking…</span>;
  if (isError || !data) return <span className="badge rejected">backend unreachable</span>;

  if (!data.model.loaded) {
    return (
      <span className="badge flagged" title={data.model.error ?? "no active model"}>
        <span className="status-dot" style={{ background: "#f59e0b", boxShadow: "0 0 8px #f59e0b" }} />
        degraded · no model
      </span>
    );
  }
  return (
    <span
      className="badge accepted"
      title="Backend Connected and Healthy"
      style={{ border: "1px solid rgba(16, 185, 129, 0.4)", background: "rgba(16, 185, 129, 0.12)", color: "#34d399", display: "inline-flex", alignItems: "center", gap: "0.35rem" }}
    >
      <span className="status-dot" />
      <span>● Backend Connected</span>
      <span style={{ opacity: 0.7 }}>·</span>
      <span>model: {data.model.version}</span>
    </span>
  );
}
