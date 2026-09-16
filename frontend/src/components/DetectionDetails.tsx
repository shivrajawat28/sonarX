import { useState } from "react";
import { Detection, bboxInSpace } from "../api/client";
import ConfidenceIndicator from "./ConfidenceIndicator";
import { CLASS_COLORS } from "./SonarViewer";

interface Props {
  detections: Detection[];
  selectedId?: string | null;
  onSelect?: (detectionId: string | null) => void;
  coordSpace?: "source" | "processed";
}

type FilterStatus = "all" | "accepted" | "flagged" | "rejected";

/**
 * Detail list & table for the selected run's detections.
 * Preserves exact table semantics (tbody tr) and test hooks for Playwright e2e,
 * with status filter tabs and clean marine aesthetic.
 */
export default function DetectionDetails({
  detections,
  selectedId = null,
  onSelect,
  coordSpace = "source",
}: Props) {
  const [filter, setFilter] = useState<FilterStatus>("all");

  if (detections.length === 0) {
    return <p className="muted" style={{ padding: "1rem 0" }}>No detections returned for this run.</p>;
  }

  const filteredDetections = detections.filter((d) => {
    if (filter === "all") return true;
    return d.filtering_status === filter;
  });

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "0.85rem" }}>
      {/* Filter Tabs */}
      <div style={{ display: "flex", gap: "0.4rem", flexWrap: "wrap" }}>
        {(["all", "accepted", "flagged", "rejected"] as FilterStatus[]).map((f) => {
          const count = f === "all" ? detections.length : detections.filter((d) => d.filtering_status === f).length;
          const isActive = filter === f;
          return (
            <button
              key={f}
              type="button"
              className={isActive ? "" : "secondary"}
              style={{
                padding: "0.25rem 0.65rem",
                fontSize: "0.75rem",
                borderRadius: "var(--radius-full)",
                textTransform: "capitalize",
              }}
              onClick={() => setFilter(f)}
            >
              {f} ({count})
            </button>
          );
        })}
      </div>

      {/* Detections Table */}
      <div style={{ overflowX: "auto", borderRadius: "var(--radius-md)", border: "1px solid var(--border-subtle)" }}>
        <table>
          <thead>
            <tr>
              <th>Class</th>
              <th>Confidence</th>
              <th>BBox (px, {coordSpace})</th>
              <th>Why filtered</th>
              <th>Location</th>
              <th>Model</th>
            </tr>
          </thead>
          <tbody>
            {filteredDetections.map((d) => {
              const box = bboxInSpace(d, coordSpace);
              const isSelected = d.detection_id === selectedId;
              const classColor = CLASS_COLORS[d.class_name] ?? "var(--accent)";

              return (
                <tr
                  key={d.detection_id}
                  onClick={() => onSelect?.(isSelected ? null : d.detection_id)}
                  style={{
                    cursor: onSelect ? "pointer" : undefined,
                    background: isSelected ? "rgba(0, 242, 254, 0.12)" : undefined,
                    borderLeft: isSelected ? "3px solid #00f2fe" : "3px solid transparent",
                    transition: "background 0.15s ease",
                  }}
                >
                  <td style={{ whiteSpace: "nowrap" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                      <span
                        style={{
                          width: 8,
                          height: 8,
                          borderRadius: "50%",
                          background: classColor,
                          boxShadow: `0 0 6px ${classColor}`,
                        }}
                      />
                      <strong style={{ color: "#fff" }}>{d.class_name}</strong>
                    </div>
                  </td>
                  <td>
                    <ConfidenceIndicator detection={d} />
                  </td>
                  <td className="muted mono" style={{ fontSize: "0.76rem" }}>
                    {box
                      ? `[${[box.x, box.y, box.x + box.w, box.y + box.h]
                          .map((v) => Math.round(v))
                          .join(", ")}]`
                      : `— (no ${coordSpace} coords)`}
                  </td>
                  <td className="muted" style={{ fontSize: "0.78rem" }}>
                    {d.filter_reasons.length ? d.filter_reasons.join("; ") : "no rule triggered"}
                  </td>
                  <td>
                    {d.latitude != null && d.longitude != null ? (
                      <span
                        className="mono"
                        style={{ fontSize: "0.8rem" }}
                        title={
                          `method: ${d.geo_provenance?.method ?? "unknown"}` +
                          (d.geo_provenance?.uncertainty_m != null
                            ? ` · ±${d.geo_provenance.uncertainty_m} m`
                            : "")
                        }
                      >
                        {d.latitude.toFixed(5)}, {d.longitude.toFixed(5)}
                        {d.geo_provenance?.uncertainty_m != null && (
                          <span className="muted"> ±{d.geo_provenance.uncertainty_m}m</span>
                        )}
                      </span>
                    ) : (
                      <span
                        className="muted"
                        style={{ fontSize: "0.76rem" }}
                        title={d.geo_provenance?.reason ?? d.geo_reason ?? d.geo_status}
                      >
                        Location unavailable — no navigation metadata provided (
                        {d.geo_status})
                      </span>
                    )}
                  </td>
                  <td className="muted mono" style={{ fontSize: "0.74rem" }}>
                    {d.model_version}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
