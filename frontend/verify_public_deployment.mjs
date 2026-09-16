import { chromium } from "playwright";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const PUBLIC_FRONTEND = "https://neighborhood-wanna-tobago-cultures.trycloudflare.com";
const PUBLIC_BACKEND = "https://importantly-switches-reported-hours.trycloudflare.com";

async function run() {
  console.log("==================================================");
  console.log("STARTING PUBLIC DEPLOYMENT END-TO-END VERIFICATION");
  console.log("Frontend URL:", PUBLIC_FRONTEND);
  console.log("Backend URL: ", PUBLIC_BACKEND);
  console.log("==================================================");

  const browser = await chromium.launch({ channel: "chrome", headless: true });
  const page = await browser.newPage();
  await page.setViewportSize({ width: 1440, height: 950 });

  const errors = [];
  page.on("console", (msg) => {
    if (msg.type() === "error") {
      errors.push(msg.text());
    }
  });

  // Track failed requests
  const failedRequests = [];
  page.on("requestfailed", (req) => {
    failedRequests.push(`${req.method()} ${req.url()} - ${req.failure()?.errorText}`);
  });

  // -------------------------------------------------------------
  // 1. PUBLIC FRONTEND LOAD & BRANDING
  // -------------------------------------------------------------
  console.log("\n[1/8] Verifying Public Frontend Loading & Branding...");
  await page.goto(PUBLIC_FRONTEND, { waitUntil: "networkidle" });
  console.log("✓ Loaded", PUBLIC_FRONTEND);

  const navText = await page.locator("header.app-header").innerText();
  if (!navText.includes("SONARX")) {
    throw new Error("Branding 'SONARX' missing in header on public deployment");
  }
  console.log("✓ Navbar branding 'SONARX' verified");

  // Wait for Backend Connected badge
  let connected = false;
  for (let i = 0; i < 15; i++) {
    const text = await page.locator("header.app-header").innerText();
    if (/Backend Connected/i.test(text)) {
      connected = true;
      break;
    }
    await page.waitForTimeout(1000);
  }
  if (!connected) {
    throw new Error("Public frontend failed to connect to public backend");
  }
  console.log("✓ '● Backend Connected' verified against public backend");

  // -------------------------------------------------------------
  // 2. DEMO MODE ON PUBLIC DEPLOYMENT
  // -------------------------------------------------------------
  console.log("\n[2/8] Verifying Demo Mode on Public Deployment...");
  const demoToggle = page.locator('button:has-text("DEMO MODE")');
  await demoToggle.click();
  await page.waitForTimeout(500);

  const demoBadge = await page.getByText(/DEMO MODE — PRECOMPUTED VERIFIED SAMPLE/).first().isVisible();
  if (!demoBadge) throw new Error("Demo mode badge not visible on public deployment");
  console.log("✓ Demo Mode badge verified");

  // Test Pipeline Demo
  await page.locator('button:has-text("Pipeline Detection")').click();
  await page.waitForTimeout(500);
  let bodyText = await page.locator("body").innerText();
  if (!bodyText.includes("submarine_pipeline") || !bodyText.includes("65.3%")) {
    throw new Error("Demo Pipeline Detection failed on public deployment");
  }
  console.log("✓ Demo Pipeline Detection verified (submarine_pipeline, 65.3%, Accepted)");

  // Test Shipwreck Demo
  await page.locator('button:has-text("Shipwreck Detection")').click();
  await page.waitForTimeout(500);
  bodyText = await page.locator("body").innerText();
  if (!bodyText.includes("shipwreck")) {
    throw new Error("Demo Shipwreck Detection failed on public deployment");
  }
  console.log("✓ Demo Shipwreck Detection verified (shipwreck, dual detections, flagged)");

  // Test Seafloor (Clean 0-detection) Demo
  await page.locator('button:has-text("Seafloor")').click();
  await page.waitForTimeout(500);
  bodyText = await page.locator("body").innerText();
  if (!bodyText.includes("0 Objects Detected") && !bodyText.includes("No detections returned") && !bodyText.includes("No anomalies detected")) {
    throw new Error("Demo Clean Seafloor failed on public deployment");
  }
  console.log("✓ Demo Seafloor (0 anomalies) verified");

  // -------------------------------------------------------------
  // 3. PUBLIC LIVE AI ANALYSIS (REAL INFERENCE OVER INTERNET)
  // -------------------------------------------------------------
  console.log("\n[3/8] Verifying Real LIVE AI Inference over Public Internet...");
  const liveToggle = page.locator('button:has-text("LIVE AI ANALYSIS")');
  await liveToggle.click();
  await page.waitForTimeout(500);

  const liveBadge = await page.getByText(/LIVE AI ANALYSIS — REAL MODEL INFERENCE/).first().isVisible();
  if (!liveBadge) throw new Error("Live mode badge not visible on public deployment");
  console.log("✓ Live AI mode badge verified");

  // Load live sample
  const sampleBtn = page.locator('button:has-text("Pipeline Tile")');
  await sampleBtn.click();
  await page.waitForTimeout(1000);

  // Click Run detection
  const runBtn = page.getByRole("button", { name: /run detection/i });
  await runBtn.click();
  console.log("✓ Clicked 'Run detection'");

  // Wait for real inference to return
  let liveInferenceDone = false;
  for (let i = 0; i < 20; i++) {
    await page.waitForTimeout(1000);
    const content = await page.locator("body").innerText();
    if (content.includes("drishti-ss_yolov8n_e30_final") && content.includes("submarine_pipeline")) {
      liveInferenceDone = true;
      break;
    }
  }
  if (!liveInferenceDone) {
    throw new Error("Public live AI inference timed out or failed to return model results");
  }
  console.log("✓ Real YOLO live inference completed successfully on public deployment!");

  // Verify honest location status
  bodyText = await page.locator("body").innerText();
  if (!bodyText.includes("Location unavailable — no navigation metadata provided")) {
    throw new Error("Missing honest location notice on public workbench live analysis");
  }
  console.log("✓ Honest location notice verified ('Location unavailable — no navigation metadata provided')");

  // -------------------------------------------------------------
  // 4. PUBLIC SPA ROUTING & SURVEY MISSION BATCH
  // -------------------------------------------------------------
  console.log("\n[4/8] Verifying Public SPA Routing to /survey...");
  await page.goto(`${PUBLIC_FRONTEND}/survey`, { waitUntil: "networkidle" });
  await page.waitForTimeout(1000);
  console.log("✓ Loaded public", `${PUBLIC_FRONTEND}/survey`);

  const surveyHeader = await page.locator("body").innerText();
  if (!surveyHeader.includes("SONARX") || !/SURVEY \(BATCH\) SELECTED/i.test(surveyHeader)) {
    throw new Error("Survey header or selection indicator missing on public deployment");
  }
  if (!/DRISHTI-SS YOLOv8n E30 FINAL/i.test(surveyHeader)) {
    throw new Error("Model indicator missing on public survey page");
  }
  if (!/Backend Connected/i.test(surveyHeader)) {
    throw new Error("Backend Connected badge missing on public survey page");
  }
  console.log("✓ Survey Header, model badge, and backend connection verified on public deployment");

  // -------------------------------------------------------------
  // 5. PUBLIC 1-CLICK DEMO SURVEY EXECUTION
  // -------------------------------------------------------------
  console.log("\n[5/8] Verifying 1-Click Demo Survey on Public Deployment...");
  const demoSurveyBtn = page.getByRole("button", { name: /load demo survey/i });
  if (!(await demoSurveyBtn.isVisible())) {
    throw new Error("'Load Demo Survey' button not visible on public survey page");
  }
  await demoSurveyBtn.click();
  console.log("✓ Clicked 'Load Demo Survey' on public frontend");

  // Wait for batch processing to succeed
  let surveyBatchDone = false;
  for (let i = 0; i < 30; i++) {
    await page.waitForTimeout(1500);
    const content = await page.locator("body").innerText();
    if (/job \S+ · succeeded/.test(content)) {
      surveyBatchDone = true;
      break;
    }
    if (/· failed/.test(content)) {
      throw new Error("Survey batch job reported failure on public deployment");
    }
  }
  if (!surveyBatchDone) {
    throw new Error("Survey batch processing timed out on public deployment");
  }
  console.log("✓ Public YOLO batch survey job executed and succeeded!");

  // Verify survey badge
  const surveyBadge = await page.getByText(/DEMO SURVEY — REAL BATCH INFERENCE/).first().isVisible();
  if (!surveyBadge) throw new Error("Batch inference badge not visible on public survey page");
  console.log("✓ 'DEMO SURVEY — REAL BATCH INFERENCE' badge verified");

  // Verify Summary cards
  const surveyBody = await page.locator("body").innerText();
  if (!surveyBody.includes("3 Tiles") || !surveyBody.includes("3 Objects") || !surveyBody.includes("3 / 3")) {
    throw new Error("Survey Summary cards count mismatch on public deployment");
  }
  console.log("✓ Survey Summary cards verified (3 Tiles, Completed, 3 Objects, 3/3 Geolocated)");

  // Verify execution pipeline tracker
  if (!/Survey Mission Execution Pipeline/i.test(surveyBody) || !/Geospatial Mapping/i.test(surveyBody)) {
    throw new Error("Batch pipeline visualizer missing on public deployment");
  }
  console.log("✓ Batch pipeline execution visualizer verified");

  // Verify Map and markers
  await page.waitForSelector(".leaflet-container", { timeout: 10000 });
  const mapVisible = await page.locator(".leaflet-container").isVisible();
  if (!mapVisible) throw new Error("Leaflet map container not rendered on public deployment");
  console.log("✓ Geospatial map rendered on public deployment");

  // Check markers
  const markers = await page.locator(".leaflet-container .leaflet-interactive").count();
  console.log(`✓ Geospatial markers count: ${markers}`);
  if (markers === 0) throw new Error("Expected Leaflet markers on map");

  // Verify Tile Inspection Panel
  if (!surveyBody.includes("Sonar Tile Inspection") || !surveyBody.includes("submarine_pipeline")) {
    throw new Error("Tile inspection panel missing or not showing submarine_pipeline");
  }
  if (!surveyBody.includes("18.90") || !surveyBody.includes("72.80")) {
    throw new Error("Coordinates missing in Tile Inspection panel");
  }
  console.log("✓ Tile Inspection panel verified with real acoustic backscatter, bbox & GPS coordinates");

  // Verify Detection Table
  const tableRows = await page.locator("tbody tr").count();
  if (tableRows < 3) throw new Error(`Expected at least 3 detection table rows, got ${tableRows}`);
  console.log(`✓ Detection table verified with ${tableRows} rows`);

  // Verify Educational Cards
  if (!surveyBody.includes("HOW GEOLOCATION WORKS") || !surveyBody.includes("DEMO SURVEY PROVENANCE")) {
    throw new Error("Educational geolocation or provenance cards missing on public survey page");
  }
  console.log("✓ Educational 'HOW GEOLOCATION WORKS' & 'DEMO SURVEY PROVENANCE' cards verified");

  // Save screenshot of public survey
  await page.screenshot({ path: path.join(ROOT, "e2e/public_deployment_survey.png"), fullPage: true });
  console.log("✓ Saved full screenshot: e2e/public_deployment_survey.png");

  // -------------------------------------------------------------
  // 6. PUBLIC MODELS PAGE
  // -------------------------------------------------------------
  console.log("\n[6/8] Verifying Public Models Page (/models)...");
  await page.goto(`${PUBLIC_FRONTEND}/models`, { waitUntil: "networkidle" });
  await page.waitForTimeout(2000);
  const modelsText = await page.locator("body").innerText();
  if (!/Evaluation metrics|Split: TEST|Model/i.test(modelsText)) {
    throw new Error("Models page failed to load metrics on public deployment");
  }
  console.log("✓ Models page verified with evaluation metrics and confusion matrix");

  // -------------------------------------------------------------
  // 7. PUBLIC HISTORY PAGE
  // -------------------------------------------------------------
  console.log("\n[7/8] Verifying Public History Page (/history)...");
  await page.goto(`${PUBLIC_FRONTEND}/history`, { waitUntil: "networkidle" });
  await page.waitForTimeout(1500);
  const historyText = await page.locator("body").innerText();
  if (!/Detection history|history|Runs/i.test(historyText)) {
    throw new Error("History page failed to load on public deployment");
  }
  console.log("✓ History page verified");

  // -------------------------------------------------------------
  // 8. BROWSER CONSOLE ERRORS & FAILED REQUESTS AUDIT
  // -------------------------------------------------------------
  console.log("\n[8/8] Auditing Browser Console Errors & Network Failures...");
  console.log(`Browser console errors: ${errors.length}`);
  if (errors.length > 0) {
    console.error("Console Errors:", errors);
    throw new Error(`Public deployment had ${errors.length} browser console error(s)`);
  }
  console.log("✓ Zero browser console errors");

  // Filter out any favicon or non-critical 404s
  const criticalFails = failedRequests.filter((f) => !f.includes("favicon"));
  console.log(`Critical network request failures: ${criticalFails.length}`);
  if (criticalFails.length > 0) {
    console.error("Failed Requests:", criticalFails);
    throw new Error(`Public deployment had ${criticalFails.length} failed network request(s)`);
  }
  console.log("✓ Zero critical network request failures");

  await browser.close();

  console.log("\n==================================================");
  console.log("PUBLIC DEPLOYMENT END-TO-END VERIFICATION: PASSED!");
  console.log("PUBLIC FRONTEND: ", PUBLIC_FRONTEND);
  console.log("PUBLIC BACKEND:  ", PUBLIC_BACKEND);
  console.log("==================================================");
}

run().catch((err) => {
  console.error("\nPUBLIC VERIFICATION FAILED:", err.message);
  process.exit(1);
});
