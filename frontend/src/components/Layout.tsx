import { ReactNode } from "react";
import { NavLink } from "react-router-dom";
import HealthBadge from "./HealthBadge";

/** App shell: nav + health indicator + routed content. */
export default function Layout({ children }: { children: ReactNode }) {
  return (
    <div style={{ maxWidth: 1200, margin: "0 auto", padding: "1rem" }}>
      <header className="row" style={{ alignItems: "center", marginBottom: "1rem" }}>
        <h1 style={{ marginRight: "auto" }}>Sonar Debris AI</h1>
        <nav className="row" style={{ gap: "0.75rem" }}>
          <NavLink to="/">Workbench</NavLink>
          <NavLink to="/survey">Survey (batch)</NavLink>
          <NavLink to="/history">History</NavLink>
          <NavLink to="/models">Models</NavLink>
        </nav>
        <HealthBadge />
      </header>
      <main>{children}</main>
      <footer className="muted" style={{ marginTop: "2rem", fontSize: "0.75rem" }}>
        Detections show model confidence (raw) and final confidence (after rule-based
        filtering). Coordinates appear ONLY when real navigation metadata existed
        (survey mode with a nav sidecar) — never invented.
      </footer>
    </div>
  );
}
