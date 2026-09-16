import { Detection } from "../api/client";

const STATUS_CLASS: Record<string, string> = {
  accepted: "accepted",
  flagged: "flagged",
  rejected: "rejected",
};

/**
 * Model confidence (raw detector output) vs final confidence (after rule-based filtering).
 * Dual progress line visualization with distinct status badges.
 */
export default function ConfidenceIndicator({ detection }: { detection: Detection }) {
  const modelPct = detection.model_confidence * 100;
  const finalPct = detection.final_confidence * 100;

  const getBarColor = (pct: number) => {
    if (pct >= 70) return "#10b981";
    if (pct >= 40) return "#f59e0b";
    return "#ef4444";
  };

  return (
    <div
      className="confidence-bar-wrapper"
      title={`Filter reasons: ${detection.filter_reasons.join("; ") || "none"}`}
    >
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "0.2rem" }}>
        <span className={`badge ${STATUS_CLASS[detection.filtering_status] ?? ""}`}>
          {detection.filtering_status}
        </span>
      </div>

      {/* Model Confidence Progress Line */}
      <div className="confidence-row">
        <span className="muted" style={{ fontSize: "0.72rem", width: 44 }}>raw</span>
        <div className="progress-track">
          <div
            className="progress-fill"
            style={{ width: `${Math.min(100, modelPct)}%`, background: getBarColor(modelPct) }}
          />
        </div>
        <span className="mono" style={{ fontSize: "0.76rem", fontWeight: 600, width: 48, textAlign: "right" }}>
          {modelPct.toFixed(1)}%
        </span>
      </div>

      {/* Final Confidence Progress Line */}
      <div className="confidence-row">
        <span className="muted" style={{ fontSize: "0.72rem", width: 44 }}>final</span>
        <div className="progress-track">
          <div
            className="progress-fill"
            style={{ width: `${Math.min(100, finalPct)}%`, background: getBarColor(finalPct) }}
          />
        </div>
        <span className="mono" style={{ fontSize: "0.76rem", fontWeight: 600, width: 48, textAlign: "right" }}>
          {finalPct.toFixed(1)}%
        </span>
      </div>
    </div>
  );
}
