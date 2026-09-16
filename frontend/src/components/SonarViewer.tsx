import { useState, useRef } from "react";
import { CoordSpace, Detection, bboxInSpace, boxToXyxy } from "../api/client";

interface Props {
  imageUrl?: string | null;
  detections: Detection[];
  alt?: string;
  coordSpace: CoordSpace;
  selectedId?: string | null;
  onSelect?: (detectionId: string | null) => void;
  isScanning?: boolean;
}

export const STATUS_COLOR: Record<string, string> = {
  accepted: "#10b981",
  flagged: "#f59e0b",
  rejected: "#ef4444",
};

export const CLASS_COLORS: Record<string, string> = {
  submarine_pipeline: "#00f2fe",
  shipwreck: "#f59e0b",
  ghost_net: "#c084fc",
  mine_cylinder: "#fb7185",
};

/**
 * Sonar viewer with interactive bounding box overlay, zoom controls,
 * empty state radar graphic, and live scanning animation.
 */
export default function SonarViewer({
  imageUrl,
  detections,
  alt = "sonar image",
  coordSpace,
  selectedId = null,
  onSelect,
  isScanning = false,
}: Props) {
  const [size, setSize] = useState<{ w: number; h: number } | null>(null);
  const [zoom, setZoom] = useState<number>(1);
  const containerRef = useRef<HTMLDivElement>(null);

  // Overlay opacity: when a detection is selected, dim other boxes
  const dimOthers = selectedId != null;
  const unplaceable = detections.filter((d) => bboxInSpace(d, coordSpace) == null);

  const handleZoom = (delta: number) => {
    setZoom((prev) => Math.min(Math.max(Number((prev + delta).toFixed(2)), 0.5), 3));
  };

  const handleResetZoom = (level: number = 1) => {
    setZoom(level);
  };

  // 1. Empty State when no image has been uploaded
  if (!imageUrl) {
    return (
      <div className="sonar-empty-state">
        <div className="radar-rings" />
        <div style={{ position: "relative", zIndex: 2, textAlign: "center", padding: "1.5rem" }}>
          <div style={{
            width: 56,
            height: 56,
            borderRadius: "50%",
            background: "rgba(0, 242, 254, 0.1)",
            border: "1px solid rgba(0, 242, 254, 0.3)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            margin: "0 auto 1rem",
            boxShadow: "0 0 20px rgba(0, 242, 254, 0.2)",
          }}>
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#00f2fe" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="12" x2="19" y2="5" />
              <circle cx="12" cy="12" r="2" fill="#00f2fe" />
            </svg>
          </div>
          <h3 style={{ fontSize: "1.2rem", fontWeight: 700, margin: "0 0 0.4rem", color: "#fff" }}>
            Sonar Analysis Workspace
          </h3>
          <p className="muted" style={{ maxWidth: 380, margin: "0 auto", fontSize: "0.86rem" }}>
            Upload a Side-Scan Sonar image to begin AI-powered detection, false-positive filtering, and spatial geolocation.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
      {/* Zoom / Viewport Toolbar */}
      <div style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        background: "rgba(10, 24, 46, 0.5)",
        border: "1px solid var(--border-subtle)",
        borderRadius: "var(--radius-md)",
        padding: "0.35rem 0.75rem",
        fontSize: "0.8rem",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
          <span className="muted">Zoom: <strong>{Math.round(zoom * 100)}%</strong></span>
          <button
            type="button"
            className="secondary"
            style={{ padding: "0.2rem 0.55rem", fontSize: "0.78rem" }}
            onClick={() => handleZoom(0.25)}
            title="Zoom In"
          >
            + In
          </button>
          <button
            type="button"
            className="secondary"
            style={{ padding: "0.2rem 0.55rem", fontSize: "0.78rem" }}
            onClick={() => handleZoom(-0.25)}
            title="Zoom Out"
          >
            - Out
          </button>
          <button
            type="button"
            className="secondary"
            style={{ padding: "0.2rem 0.55rem", fontSize: "0.78rem" }}
            onClick={() => handleResetZoom(1)}
            title="Fit / 1:1"
          >
            Fit (1:1)
          </button>
        </div>

        {isScanning && (
          <div style={{ display: "flex", alignItems: "center", gap: "0.4rem", color: "var(--accent)" }}>
            <span className="status-dot" style={{ background: "#00f2fe", boxShadow: "0 0 8px #00f2fe" }} />
            <span style={{ fontWeight: 600, fontSize: "0.76rem" }}>Scanning sonar target…</span>
          </div>
        )}
      </div>

      {/* Main Sonar Image + SVG Bounding Box Canvas Container */}
      <div
        ref={containerRef}
        style={{
          position: "relative",
          overflow: "auto",
          maxWidth: "100%",
          maxHeight: 640,
          background: "#020710",
          border: "1px solid var(--border)",
          borderRadius: "var(--radius-md)",
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
        }}
      >
        <div
          style={{
            position: "relative",
            display: "inline-block",
            transform: `scale(${zoom})`,
            transformOrigin: "center center",
            transition: "transform 0.15s ease",
          }}
        >
          {/* Main Sonar Image */}
          <img
            src={imageUrl}
            alt={alt}
            onLoad={(e) => {
              const img = e.currentTarget;
              setSize({ w: img.naturalWidth, h: img.naturalHeight });
            }}
            style={{
              maxWidth: "100%",
              display: "block",
              borderRadius: 6,
              filter: isScanning ? "brightness(1.08)" : undefined,
            }}
          />

          {/* Radar Scan Light Sweep Overlay during inference */}
          {isScanning && <div className="radar-scan-overlay" />}

          {/* SVG Overlay: Exact coordinate matching for displayed space */}
          {size && size.w > 0 && size.h > 0 && (
            <svg
              viewBox={`0 0 ${size.w} ${size.h}`}
              preserveAspectRatio="none"
              style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }}
            >
              {detections.map((d) => {
                const box = bboxInSpace(d, coordSpace);
                if (box == null) return null;
                const [x1, y1, x2, y2] = boxToXyxy(box);
                const color = STATUS_COLOR[d.filtering_status] ?? "#d29922";
                const rejected = d.filtering_status === "rejected";
                const isSelected = d.detection_id === selectedId;
                const opacity = dimOthers && !isSelected ? 0.25 : 1;
                const strokeWidth = Math.max(size.w / 400, 1.2) * (isSelected ? 2.5 : 1);

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
                      strokeWidth={strokeWidth}
                      strokeDasharray={rejected ? "6 4" : undefined}
                      style={{
                        filter: isSelected ? `drop-shadow(0 0 6px ${color})` : undefined,
                        transition: "stroke-width 0.15s ease",
                      }}
                    />
                    <text
                      x={x1}
                      y={Math.max(y1 - 4, 12)}
                      fill={color}
                      fontSize={Math.max(size.w / 45, 11)}
                      fontWeight="bold"
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

          {detections.length === 0 && !isScanning && (
            <div className="muted" style={{ position: "absolute", top: 8, left: 8, background: "rgba(0,0,0,0.6)", padding: "2px 6px", borderRadius: 4 }}>
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
                background: "rgba(0,0,0,0.7)",
                padding: "2px 6px",
                borderRadius: 4,
              }}
            >
              ⚠ {unplaceable.length} detection(s) have no coordinates in{" "}
              {coordSpace} space — not drawn (never guessed)
            </div>
          )}
        </div>
      </div>

      {/* Required provenance caption matching e2e contracts */}
      <div className="muted" style={{ fontSize: "0.74rem", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <span>boxes in {coordSpace}-image pixel space · label shows raw model confidence</span>
        {size && <span>{size.w}×{size.h} px</span>}
      </div>
    </div>
  );
}
