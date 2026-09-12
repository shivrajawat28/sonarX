import { useState } from "react";
import { CoordSpace, Detection, bboxInSpace, boxToXyxy } from "../api/client";

interface Props {
  imageUrl: string;
  detections: Detection[];
  alt?: string;
  /**
   * Which pixel space `imageUrl` is rendered in. MUST match the image actually
   * shown, otherwise the overlay is drawn at the wrong offset (the preprocessing
   * letterbox shifts every box when the source is not 640x640).
   */
  coordSpace: CoordSpace;
  /** Currently selected detection id (linked to the details list). */
  selectedId?: string | null;
  onSelect?: (detectionId: string | null) => void;
}

export const STATUS_COLOR: Record<string, string> = {
  accepted: "#2ea043",
  flagged: "#d29922",
  rejected: "#f85149",
};

/**
 * Sonar viewer with bbox overlay. The backend supplies pixel coordinates in
 * BOTH spaces (bbox_source_coords for the original upload, bbox_processed_coords
 * for the preprocessed artifact); this component draws whichever space matches
 * the image on screen. All values displayed come from the API — no client-side
 * scoring or class logic (architecture hard rule). Rejected detections are drawn
 * dashed (annotated, never hidden).
 */
export default function SonarViewer({
  imageUrl,
  detections,
  alt = "sonar image",
  coordSpace,
  selectedId = null,
  onSelect,
}: Props) {
  const [size, setSize] = useState<{ w: number; h: number } | null>(null);

  // Overlay opacity: when a detection is selected, fade the others so the
  // selected box is unmistakable; the list and the image stay in sync.
  const dimOthers = selectedId != null;
  const unplaceable = detections.filter((d) => bboxInSpace(d, coordSpace) == null);

  return (
    <div style={{ position: "relative", display: "inline-block", maxWidth: "100%" }}>
      <img
        src={imageUrl}
        alt={alt}
        onLoad={(e) => {
          const img = e.currentTarget;
          setSize({ w: img.naturalWidth, h: img.naturalHeight });
        }}
        style={{ maxWidth: "100%", display: "block", borderRadius: 6 }}
      />
      {size && size.w > 0 && size.h > 0 && (
        <svg
          viewBox={`0 0 ${size.w} ${size.h}`}
          preserveAspectRatio="none"
          style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }}
        >
          {detections.map((d) => {
            const box = bboxInSpace(d, coordSpace);
            if (box == null) return null; // never guess a box position
            const [x1, y1, x2, y2] = boxToXyxy(box);
            const color = STATUS_COLOR[d.filtering_status] ?? "#d29922";
            const rejected = d.filtering_status === "rejected";
            const isSelected = d.detection_id === selectedId;
            const opacity = dimOthers && !isSelected ? 0.25 : 1;
            return (
              <g
                key={d.detection_id}
                opacity={opacity}
                style={{ cursor: onSelect ? "pointer" : undefined }}
                onClick={() => onSelect?.(isSelected ? null : d.detection_id)}
              >
                <rect
                  x={x1}
                  y={y1}
                  width={Math.max(x2 - x1, 1)}
                  height={Math.max(y2 - y1, 1)}
                  fill="none"
                  stroke={color}
                  strokeWidth={Math.max(size.w / 400, 1) * (isSelected ? 2.5 : 1)}
                  strokeDasharray={rejected ? "6 4" : undefined}
                />
                <text
                  x={x1}
                  y={Math.max(y1 - 4, 12)}
                  fill={color}
                  fontSize={Math.max(size.w / 45, 10)}
                  style={{ paintOrder: "stroke", stroke: "#000", strokeWidth: 2 }}
                >
                  {d.class_name} {(d.model_confidence * 100).toFixed(0)}%
                  {rejected ? " (rejected)" : ""}
                </text>
              </g>
            );
          })}
        </svg>
      )}
      {detections.length === 0 && (
        <div className="muted" style={{ position: "absolute", top: 8, left: 8 }}>
          no detections in this run
        </div>
      )}
      {unplaceable.length > 0 && (
        <div
          style={{
            position: "absolute",
            top: 8,
            left: 8,
            color: "var(--warn)",
            fontSize: "0.8rem",
          }}
        >
          ⚠ {unplaceable.length} detection(s) have no coordinates in{" "}
          {coordSpace} space — not drawn (never guessed)
        </div>
      )}
      <div className="muted" style={{ fontSize: "0.72rem", marginTop: "0.3rem" }}>
        boxes in {coordSpace}-image pixel space · label shows raw model confidence
      </div>
    </div>
  );
}
