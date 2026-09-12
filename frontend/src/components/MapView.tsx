import { MapContainer, TileLayer, CircleMarker, Tooltip } from "react-leaflet";
import { Detection } from "../api/client";
import "leaflet/dist/leaflet.css";

/**
 * Map of geolocated detections. Renders ONLY detections whose coordinates came
 * from real navigation metadata (lat/lon non-null) — the system never
 * fabricates coordinates (Section 10), and this map never guesses them.
 */
export default function MapView({ detections }: { detections: Detection[] }) {
  const geo = detections.filter((d) => d.latitude != null && d.longitude != null);

  if (geo.length === 0) {
    return (
      <p className="muted">
        No geolocated detections. Coordinates appear only when real navigation
        metadata was available for the source data.
      </p>
    );
  }

  const lats = geo.map((d) => d.latitude as number);
  const lons = geo.map((d) => d.longitude as number);
  const center: [number, number] = [
    (Math.min(...lats) + Math.max(...lats)) / 2,
    (Math.min(...lons) + Math.max(...lons)) / 2,
  ];

  return (
    <MapContainer center={center} zoom={13} scrollWheelZoom>
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      {geo.map((d) => {
        const color =
          d.filtering_status === "accepted"
            ? "#2ea043"
            : d.filtering_status === "flagged"
              ? "#d29922"
              : "#f85149";
        return (
          <CircleMarker
            key={d.detection_id}
            center={[d.latitude as number, d.longitude as number]}
            radius={7}
            pathOptions={{ color, fillOpacity: 0.7 }}
          >
            <Tooltip>
              {d.class_name} · final {(d.final_confidence * 100).toFixed(0)}% ·{" "}
              {d.filtering_status}
            </Tooltip>
          </CircleMarker>
        );
      })}
    </MapContainer>
  );
}
