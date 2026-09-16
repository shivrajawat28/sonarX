import { MapContainer, TileLayer, CircleMarker, Tooltip, Popup, Polyline } from "react-leaflet";
import { Detection } from "../api/client";
import "leaflet/dist/leaflet.css";

interface MapViewProps {
  detections: Detection[];
  trackPoints?: [number, number][];
  selectedId?: string | null;
  onSelect?: (d: Detection) => void;
  height?: string | number;
}

/**
 * Map of geolocated detections and survey trajectory track.
 * Renders ONLY detections whose coordinates came from real navigation metadata
 * (lat/lon non-null) — the system never fabricates coordinates (Section 10),
 * and this map never guesses them.
 */
export default function MapView({
  detections,
  trackPoints,
  selectedId = null,
  onSelect,
  height = 420,
}: MapViewProps) {
  const geo = detections.filter((d) => d.latitude != null && d.longitude != null);

  if (geo.length === 0) {
    return (
      <p className="muted">
        No geolocated detections. Coordinates appear only when real navigation
        metadata was available for the source data.
      </p>
    );
  }

  // Derive all coordinate points for bounding box center
  const lats = geo.map((d) => d.latitude as number);
  const lons = geo.map((d) => d.longitude as number);

  // If track points exist, include them in the bounding box
  const allLats = trackPoints && trackPoints.length > 0 ? [...lats, ...trackPoints.map((p) => p[0])] : lats;
  const allLons = trackPoints && trackPoints.length > 0 ? [...lons, ...trackPoints.map((p) => p[1])] : lons;

  const center: [number, number] = [
    (Math.min(...allLats) + Math.max(...allLats)) / 2,
    (Math.min(...allLons) + Math.max(...allLons)) / 2,
  ];

  // Derive survey trajectory track if not explicitly passed:
  // Sort geolocated detections along track to create a trajectory line
  const effectiveTrack: [number, number][] =
    trackPoints && trackPoints.length > 1
      ? trackPoints
      : geo.length > 1
        ? (geo.map((d) => [d.latitude as number, d.longitude as number]) as [number, number][])
        : [];

  return (
    <div
      style={{
        borderRadius: "var(--radius-md)",
        overflow: "hidden",
        border: "1px solid var(--border)",
        boxShadow: "0 8px 30px rgba(0, 0, 0, 0.4)",
        height,
        position: "relative",
      }}
    >
      <MapContainer
        center={center}
        zoom={14}
        scrollWheelZoom
        style={{ height: "100%", width: "100%", background: "#0b1526" }}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        {/* Survey Trajectory / Trackline */}
        {effectiveTrack.length >= 2 && (
          <>
            <Polyline
              positions={effectiveTrack}
              pathOptions={{
                color: "#00f2fe",
                weight: 3,
                dashArray: "6, 8",
                opacity: 0.85,
              }}
            />
            {/* Start of survey track */}
            <CircleMarker
              center={effectiveTrack[0]}
              radius={5}
              pathOptions={{ color: "#38bdf8", fillColor: "#0284c7", fillOpacity: 0.9 }}
            >
              <Tooltip>Start of Survey Track (nav.csv)</Tooltip>
            </CircleMarker>
            {/* End of survey track */}
            <CircleMarker
              center={effectiveTrack[effectiveTrack.length - 1]}
              radius={5}
              pathOptions={{ color: "#a855f7", fillColor: "#7e22ce", fillOpacity: 0.9 }}
            >
              <Tooltip>End of Survey Track (nav.csv)</Tooltip>
            </CircleMarker>
          </>
        )}

        {/* Detection Markers */}
        {geo.map((d) => {
          const isSelected = selectedId === d.detection_id;
          const statusColor =
            d.filtering_status === "accepted"
              ? "#10b981"
              : d.filtering_status === "flagged"
                ? "#f59e0b"
                : "#ef4444";

          const rawPct = (d.model_confidence * 100).toFixed(1);
          const finalPct = (d.final_confidence * 100).toFixed(1);
          const uncertainty = d.geo_provenance?.uncertainty_m != null ? ` ±${d.geo_provenance.uncertainty_m}m` : "";

          return (
            <CircleMarker
              key={d.detection_id}
              center={[d.latitude as number, d.longitude as number]}
              radius={isSelected ? 11 : 8}
              pathOptions={{
                color: isSelected ? "#00f2fe" : statusColor,
                fillColor: statusColor,
                fillOpacity: isSelected ? 0.95 : 0.8,
                weight: isSelected ? 3 : 1.5,
              }}
              eventHandlers={{
                click: () => {
                  if (onSelect) onSelect(d);
                },
              }}
            >
              <Tooltip>
                <div style={{ fontSize: "0.82rem", fontWeight: 600 }}>
                  {d.class_name} · final {finalPct}% · {d.filtering_status}
                </div>
                <div style={{ fontSize: "0.74rem", color: "#64748b" }}>
                  Lat: {(d.latitude as number).toFixed(4)}, Lon: {(d.longitude as number).toFixed(4)}
                </div>
              </Tooltip>

              <Popup>
                <div style={{ color: "#0f172a", minWidth: 200, fontSize: "0.82rem" }}>
                  <div style={{ fontWeight: 700, fontSize: "0.95rem", color: "#0284c7", marginBottom: "0.25rem" }}>
                    {d.class_name}
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.2rem" }}>
                    <span>Raw Confidence:</span>
                    <strong>{rawPct}%</strong>
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.2rem" }}>
                    <span>Final Confidence:</span>
                    <strong>{finalPct}%</strong>
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.2rem" }}>
                    <span>Status:</span>
                    <span
                      style={{
                        textTransform: "capitalize",
                        fontWeight: 600,
                        color: statusColor,
                      }}
                    >
                      {d.filtering_status}
                    </span>
                  </div>
                  {d.filter_reasons && d.filter_reasons.length > 0 && (
                    <div style={{ color: "#b45309", fontSize: "0.74rem", marginTop: "0.2rem" }}>
                      Reason: {d.filter_reasons.join(", ")}
                    </div>
                  )}
                  <hr style={{ margin: "0.45rem 0", border: 0, borderTop: "1px solid #e2e8f0" }} />
                  <div style={{ fontSize: "0.75rem", color: "#475569" }}>
                    <div>
                      <strong>Lat:</strong> {(d.latitude as number).toFixed(5)}° N
                    </div>
                    <div>
                      <strong>Lon:</strong> {(d.longitude as number).toFixed(5)}° E
                    </div>
                    {uncertainty && (
                      <div>
                        <strong>Uncertainty:</strong> {uncertainty}
                      </div>
                    )}
                    <div>
                      <strong>Source:</strong> nav.csv (survey navigation metadata)
                    </div>
                  </div>

                  {onSelect && (
                    <button
                      type="button"
                      onClick={() => onSelect(d)}
                      style={{
                        marginTop: "0.6rem",
                        width: "100%",
                        padding: "0.3rem 0.6rem",
                        fontSize: "0.78rem",
                        background: "#0284c7",
                        color: "#fff",
                        border: 0,
                        borderRadius: "4px",
                        cursor: "pointer",
                      }}
                    >
                      🔍 Inspect Sonar Tile
                    </button>
                  )}
                </div>
              </Popup>
            </CircleMarker>
          );
        })}
      </MapContainer>
    </div>
  );
}
