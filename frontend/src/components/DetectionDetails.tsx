import { Detection, bboxInSpace } from "../api/client";
import ConfidenceIndicator from "./ConfidenceIndicator";

interface Props {
  detections: Detection[];
  selectedId?: string | null;
  onSelect?: (detectionId: string | null) => void;
  /** Space of the image shown next to this list — bbox column follows it. */
  coordSpace?: "source" | "processed";
}

/**
 * Detail list for the selected run's detections — values come from the API.
 * Rows are selectable so the list and the overlay highlight stay in exact
 * correspondence (clicking a row marks the same box on the image).
 */
export default function DetectionDetails({
  detections,
  selectedId = null,
  onSelect,
  coordSpace = "source",
}: Props) {
  if (detections.length === 0) {
    return <p className="muted">No detections returned for this run.</p>;
  }
  return (
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
        {detections.map((d) => {
          const box = bboxInSpace(d, coordSpace);
          const isSelected = d.detection_id === selectedId;
          return (
            <tr
              key={d.detection_id}
              onClick={() => onSelect?.(isSelected ? null : d.detection_id)}
              style={{
                cursor: onSelect ? "pointer" : undefined,
                background: isSelected ? "rgba(47,129,247,0.14)" : undefined,
              }}
            >
              <td>{d.class_name}</td>
              <td>
                <ConfidenceIndicator detection={d} />
              </td>
              <td className="muted">
                {box
                  ? `[${[box.x, box.y, box.x + box.w, box.y + box.h]
                      .map((v) => Math.round(v))
                      .join(", ")}]`
                  : `— (no ${coordSpace} coords)`}
              </td>
              <td className="muted" style={{ fontSize: "0.8rem" }}>
                {d.filter_reasons.length ? d.filter_reasons.join("; ") : "no rule triggered"}
              </td>
              <td>
                {d.latitude != null && d.longitude != null ? (
                  <span
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
                    title={d.geo_provenance?.reason ?? d.geo_reason ?? d.geo_status}
                  >
                    Location unavailable — no navigation metadata provided (
                    {d.geo_status})
                  </span>
                )}
              </td>
              <td className="muted">{d.model_version}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
