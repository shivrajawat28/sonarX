/**
 * E2E Verification Suite for SIH26057:
 * Tests both LIVE AI ANALYSIS and DEMO MODE (Precomputed Real Samples).
 */
import { chromium } from "playwright";
import path from "path";

const BASE = "http://localhost:5173";
const ROOT = path.resolve(import.meta.dirname, "..");
const TEST_IMAGE = path.join(
  ROOT,
  "datasets/processed/drishti-sss/test/images/pipe_1693569383.780_x3500.jpg"
);

const failures = [];
const note = (ok, label, extra = "") => {
  console.log(`${ok ? "PASS" : "FAIL"}  ${label}${extra ? " :: " + extra : ""}`);
  if (!ok) failures.push(label + (extra ? ` (${extra})` : ""));
};

const errors = [];
const browser = await chromium.launch({ channel: "chrome", headless: true });
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });

page.on("pageerror", (e) => errors.push("PAGEERROR: " + e.message));
page.on("console", (m) => {
  if (m.type() === "error") errors.push("CONSOLE: " + m.text());
});

console.log("==================================================");
console.log("STARTING LIVE + DEMO MODE VERIFICATION");
console.log("==================================================");

// 1. Initial Page Load & Default Mode Check
await page.goto(`${BASE}/`, { waitUntil: "networkidle", timeout: 30000 });
const initialBody = await page.locator("body").innerText();
note(/LIVE AI ANALYSIS/.test(initialBody), "Default mode is LIVE AI ANALYSIS");
note(/Choose Verified Sonar Sample/.test(initialBody) === false, "Demo sample selector hidden in Live Mode");

// 2. Switch to DEMO MODE
console.log("\n[TEST: Switching to DEMO MODE]");
await page.getByRole("button", { name: /DEMO MODE/i }).click();
await page.waitForTimeout(600);
const demoBody = await page.locator("body").innerText();
note(/DEMO • PRECOMPUTED REAL SAMPLE/.test(demoBody), "DEMO • PRECOMPUTED REAL SAMPLE badge visible");
note(/Choose Verified Sonar Sample/i.test(demoBody), "Demo sample selector visible");

// 3. Verify Demo Sample 1: Submarine Pipeline
console.log("\n[TEST: Demo Sample 1 — Submarine Pipeline]");
await page.getByText("Submarine Pipeline").first().click();
await page.waitForTimeout(600);
const pipeBody = await page.locator("body").innerText();
note(/Detections \(1\)/.test(pipeBody), "Pipeline sample has 1 detection");
note(/submarine_pipeline/.test(pipeBody), "Class is submarine_pipeline");
note(/accepted/.test(pipeBody), "Status is accepted");
note(/Location unavailable — no navigation metadata provided/.test(pipeBody), "Honest geolocation: Location unavailable");

// Verify bounding box on screen
const pipeRectCount = await page.locator("svg rect").count();
note(pipeRectCount === 1, "Pipeline bounding box drawn in SVG overlay", `drawn=${pipeRectCount}`);

// 4. Verify Demo Sample 2: Shipwreck / Reef
console.log("\n[TEST: Demo Sample 2 — Shipwreck / Reef]");
await page.getByText("Shipwreck / Reef").first().click();
await page.waitForTimeout(600);
const wreckBody = await page.locator("body").innerText();
note(/Detections \(2\)/.test(wreckBody), "Shipwreck sample has 2 detections");
note(/shipwreck/.test(wreckBody), "Class is shipwreck");
note(/flagged/.test(wreckBody), "Flagged status present (due to border edge_clip)");
note(/accepted/.test(wreckBody), "Accepted status present");
note(/edge_clip/.test(wreckBody), "Filter reason 'edge_clip' displayed");

const wreckRectCount = await page.locator("svg rect").count();
note(wreckRectCount === 2, "Shipwreck drawn 2 bounding boxes", `drawn=${wreckRectCount}`);

// 5. Verify Demo Sample 3: Background Seafloor (Negative Test)
console.log("\n[TEST: Demo Sample 3 — Background Seafloor (Negative Sample)]");
await page.getByText("Background / Seafloor").first().click();
await page.waitForTimeout(600);
const bgBody = await page.locator("body").innerText();
note(/Detections \(0\)/.test(bgBody), "Background sample has 0 detections");
note(/No detections returned for this run/.test(bgBody), "Empty state rendered cleanly for negative sample");

const bgRectCount = await page.locator("svg rect").count();
note(bgRectCount === 0, "No bounding boxes drawn for background tile", `drawn=${bgRectCount}`);

// 6. Test Mode Switching back to LIVE AI ANALYSIS
console.log("\n[TEST: Mode Switching — Returning to LIVE AI ANALYSIS]");
await page.getByRole("button", { name: /LIVE AI ANALYSIS/i }).click();
await page.waitForTimeout(600);
const switchedBackBody = await page.locator("body").innerText();
note(/LIVE AI ANALYSIS/.test(switchedBackBody), "Switched back to LIVE AI ANALYSIS mode");
note(/Choose Verified Sonar Sample/i.test(switchedBackBody) === false, "Demo selector disappeared");
note(/Drop sonar image now|Drag & drop sonar image here/.test(switchedBackBody), "Live upload dropzone active");

// 7. Verify Real Live Pipeline on user-uploaded file
console.log("\n[TEST: Real Live Pipeline Execution]");
await page.locator('input[type="file"]').first().setInputFiles(TEST_IMAGE);
await page.waitForTimeout(2000);

const [liveRunResp] = await Promise.all([
  page.waitForResponse(
    (r) => r.url().includes("/detections/run") && r.request().method() === "POST",
    { timeout: 60000 }
  ),
  page.getByRole("button", { name: /run detection/i }).click(),
]);
const liveResult = await liveRunResp.json();
note(liveRunResp.status() === 201, "Real live inference POST /detections/run returned 201", `status=${liveRunResp.status()}`);
note(liveResult.model_version === "drishti-ss_yolov8n_e30_final", "Live inference model is drishti-ss_yolov8n_e30_final");
note(liveResult.detections.length === 1, "Live inference returned 1 real detection", `count=${liveResult.detections.length}`);

// 8. Switch to DEMO MODE again to verify no cross-contamination
console.log("\n[TEST: Switch to DEMO MODE with active live state]");
await page.getByRole("button", { name: /DEMO MODE/i }).click();
await page.waitForTimeout(500);
const demoRecheck = await page.locator("body").innerText();
note(/DEMO • PRECOMPUTED REAL SAMPLE/.test(demoRecheck), "Re-entered DEMO MODE cleanly");

await browser.close();

console.log("\n==================================================");
console.log(`VERIFICATION COMPLETE. Failures: ${failures.length}, Console Errors: ${errors.length}`);
console.log("==================================================");
if (failures.length > 0) {
  console.error("FAILURES:", failures);
  process.exit(1);
}
if (errors.length > 0) {
  console.error("CONSOLE ERRORS:", errors);
  process.exit(1);
}
console.log("ALL LIVE AND DEMO MODE TESTS PASSED!");
process.exit(0);
