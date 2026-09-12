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
        degraded · no model
      </span>
    );
  }
  return <span className="badge accepted">model: {data.model.version}</span>;
}
