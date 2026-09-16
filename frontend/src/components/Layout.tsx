import { ReactNode, useState } from "react";
import { NavLink } from "react-router-dom";
import HealthBadge from "./HealthBadge";

/** App shell: modern marine-tech nav + health indicator + routed content + footer. */
export default function Layout({ children }: { children: ReactNode }) {
  const [showAbout, setShowAbout] = useState(false);

  return (
    <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column" }}>
      {/* Sticky Top Navigation Bar */}
      <header className="app-header">
        <div className="app-header-inner">
          <div className="brand-wrapper">
            <div className="brand-icon">
              {/* Custom SVG Sonar Radar Wave Icon */}
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#00f2fe" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="10" strokeOpacity="0.25" />
                <path d="M12 2a10 10 0 0 1 10 10" />
                <path d="M12 6a6 6 0 0 1 6 6" />
                <circle cx="12" cy="12" r="2" fill="#00f2fe" />
                <line x1="12" y1="12" x2="19" y2="5" stroke="#38bdf8" />
              </svg>
            </div>
            <div className="brand-text">
              <h1>SONARX</h1>
              <div className="brand-subtitle">AI-Powered Underwater Intelligence</div>
            </div>
          </div>

          <nav className="nav-links">
            <NavLink to="/" className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}>
              Workbench
            </NavLink>
            <NavLink to="/survey" className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}>
              Survey (batch)
            </NavLink>
            <NavLink to="/history" className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}>
              History
            </NavLink>
            <NavLink to="/models" className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}>
              Models
            </NavLink>
            <button
              type="button"
              className="nav-link secondary"
              style={{ background: "transparent", border: "none", cursor: "pointer" }}
              onClick={() => setShowAbout(true)}
            >
              About
            </button>
          </nav>

          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
            <HealthBadge />
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main style={{ maxWidth: 1520, width: "100%", margin: "0 auto", padding: "0 1.5rem", flex: 1 }}>
        {children}
      </main>

      {/* About Modal */}
      {showAbout && (
        <div
          style={{
            position: "fixed",
            inset: 0,
            zIndex: 1000,
            background: "rgba(3, 7, 18, 0.8)",
            backdropFilter: "blur(10px)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            padding: "1rem",
          }}
          onClick={() => setShowAbout(false)}
        >
          <div
            className="panel"
            style={{ maxWidth: 600, width: "100%", position: "relative" }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
              <h2 style={{ margin: 0, display: "flex", alignItems: "center", gap: "0.5rem" }}>
                <span style={{ color: "var(--accent)" }}>●</span> About SONARX
              </h2>
              <button
                className="secondary"
                style={{ padding: "0.3rem 0.6rem", fontSize: "0.8rem" }}
                onClick={() => setShowAbout(false)}
              >
                ✕
              </button>
            </div>
            <p className="muted" style={{ lineHeight: 1.6 }}>
              <strong>SIH26057</strong>: SONARX — AI-Powered Automated Underwater Marine Debris & Anomaly Detection.
            </p>
            <p className="muted" style={{ lineHeight: 1.6 }}>
              This system processes Side-Scan Sonar (SSS) imagery through an end-to-end pipeline: speckle reduction & CLAHE letterboxing → YOLOv8n object detection → deterministic false-positive filtering → along-track navigation survey interpolation → verifiable reporting.
            </p>
            <div style={{ marginTop: "1rem", padding: "0.85rem", background: "rgba(0, 242, 254, 0.05)", borderRadius: "var(--radius-md)", border: "1px solid var(--border)" }}>
              <div style={{ fontWeight: 600, color: "#fff", marginBottom: "0.25rem" }}>Active Real Model</div>
              <div className="mono" style={{ fontSize: "0.82rem", color: "var(--accent-secondary)" }}>
                drishti-ss_yolov8n_e30_final (30 epochs on DRISHTI-SSS, 3.0M params)
              </div>
            </div>
            <div style={{ marginTop: "1.25rem", textAlign: "right" }}>
              <button onClick={() => setShowAbout(false)}>Close</button>
            </div>
          </div>
        </div>
      )}

      {/* Professional Footer */}
      <footer className="app-footer">
        <div className="app-footer-inner">
          <div>
            <div style={{ fontWeight: 700, fontSize: "1rem", color: "#fff", marginBottom: "0.25rem" }}>
              SONARX
            </div>
            <div className="muted" style={{ fontSize: "0.8rem" }}>AI-Powered Underwater Intelligence</div>
          </div>

          <div style={{ textAlign: "center" }}>
            <div style={{ fontWeight: 600, color: "#fff", fontSize: "0.85rem" }}>SIH26057</div>
            <div className="muted" style={{ fontSize: "0.8rem", maxWidth: 450 }}>
              AI-Powered Underwater Marine Debris & Anomaly Detection
            </div>
          </div>

          <div style={{ textAlign: "right" }}>
            <div className="muted" style={{ fontSize: "0.8rem" }}>
              Detections show raw model confidence & final filtered confidence.
            </div>
            <div className="muted" style={{ fontSize: "0.75rem", marginTop: "0.2rem" }}>
              Coordinates appear ONLY when real navigation metadata was parsed — never invented.
            </div>
          </div>
        </div>
        <div style={{ maxWidth: 1520, margin: "1.5rem auto 0", borderTop: "1px solid var(--border-subtle)", paddingTop: "1rem", textAlign: "center", fontSize: "0.75rem", color: "var(--muted)" }}>
          © SONARX · DRISHTI-SSS Dataset Benchmark · Verified Real Inference Engine
        </div>
      </footer>
    </div>
  );
}
