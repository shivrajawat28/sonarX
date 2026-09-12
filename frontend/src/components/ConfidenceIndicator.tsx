import { Detection } from "../api/client";

const STATUS_CLASS: Record<string, string> = {
  accepted: "accepted",
  flagged: "flagged",
  rejected: "rejected",
};

/**
 * Model confidence (raw detector output, NOT a probability of correctness) vs
 * final confidence (after rule-based filtering) — distinction from ADR-006.
 */
export default function ConfidenceIndicator({ detection }: { detection: Detection }) {
  const pct = (v: number) => `${Math.round(v * 100)}%`;
  return (
    <span title={`filter reasons: ${detection.filter_reasons.join("; ") || "none"}`}>
      <span className={`badge ${STATUS_CLASS[detection.filtering_status] ?? ""}`}>
        {detection.filtering_status}
      </span>{" "}
      <span className="muted">model</span> {pct(detection.model_confidence)} ·{" "}
      <span className="muted">final</span> {pct(detection.final_confidence)}
    </span>
  );
}
