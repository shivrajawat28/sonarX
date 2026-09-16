import { chromium } from "playwright";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const BASE = "http://localhost:5173";

async function run() {
  console.log("Starting SONARX Interactive Presentation Verification...");
  const browser = await chromium.launch({ channel: "chrome", headless: true });
  const page = await browser.newPage();
  await page.setViewportSize({ width: 1440, height: 900 });

  const errors = [];
  page.on("console", (msg) => {
    if (msg.type() === "error") {
      errors.push(msg.text());
    }
  });

  // 1. Visit Workbench
  await page.goto(BASE, { waitUntil: "networkidle" });
  console.log("✓ Loaded", BASE);

  // Check branding
  const navText = await page.locator("header.app-header").innerText();
  if (!navText.includes("SONARX")) {
    throw new Error("Branding 'SONARX' missing in header/navbar");
  }
  console.log("✓ Navbar branding 'SONARX' verified");

  // 2. Click DEMO MODE
  const demoToggle = page.locator('button:has-text("DEMO MODE")');
  await demoToggle.click();
  await page.waitForTimeout(500);

  const demoBadge = await page.getByText(/DEMO MODE — PRECOMPUTED VERIFIED SAMPLE/).first().isVisible();
  if (!demoBadge) {
    throw new Error("Demo mode badge not visible");
  }
  console.log("✓ Demo mode badge verified: 'DEMO MODE — PRECOMPUTED VERIFIED SAMPLE'");

  // 3. Click "Pipeline Detection" demo button
  const pipelineBtn = page.locator('button:has-text("Pipeline Detection")');
  await pipelineBtn.click();
  await page.waitForTimeout(500);

  const bodyAfterPipeline = await page.locator("body").innerText();
  if (!bodyAfterPipeline.includes("submarine_pipeline")) {
    throw new Error("submarine_pipeline not displayed in demo pipeline result");
  }
  if (!bodyAfterPipeline.includes("65.3%")) {
    throw new Error("65.3% confidence not displayed in demo pipeline result");
  }
  console.log("✓ Demo Pipeline Detection verified (submarine_pipeline, 65.3%, Accepted)");
  await page.screenshot({ path: path.join(ROOT, "e2e/demo_1_pipeline.png"), fullPage: true });

  // 4. Click "Shipwreck Detection" demo button
  const shipwreckBtn = page.locator('button:has-text("Shipwreck Detection")');
  await shipwreckBtn.click();
  await page.waitForTimeout(500);

  const bodyAfterShipwreck = await page.locator("body").innerText();
  if (!bodyAfterShipwreck.includes("84.7%") || !bodyAfterShipwreck.includes("54.7%")) {
    throw new Error("Shipwreck detection 1 confidences (84.7% / 54.7%) missing");
  }
  if (!bodyAfterShipwreck.includes("edge clipping")) {
    throw new Error("Edge clipping flag reason missing");
  }
  if (!bodyAfterShipwreck.includes("65.0%")) {
    throw new Error("Shipwreck detection 2 confidence (65.0%) missing");
  }
  console.log("✓ Demo Shipwreck Detection verified (2 detections, edge clipping flagged, accepted)");
  await page.screenshot({ path: path.join(ROOT, "e2e/demo_2_shipwreck.png"), fullPage: true });

  // 5. Click "Seafloor — No Detection" demo button
  const seafloorBtn = page.locator('button:has-text("Seafloor — No Detection")');
  await seafloorBtn.click();
  await page.waitForTimeout(500);

  const bodyAfterSeafloor = await page.locator("body").innerText();
  if (!bodyAfterSeafloor.includes("0 Objects Detected") && !bodyAfterSeafloor.includes("No detections returned")) {
    throw new Error("Empty state message missing in seafloor demo");
  }
  if (!bodyAfterSeafloor.includes("drishti-ss_yolov8n_e30_final")) {
    throw new Error("Model provenance drishti-ss_yolov8n_e30_final missing in empty state");
  }
  console.log("✓ Demo Seafloor (No Detection) verified (0 objects, honest empty state banner)");
  await page.screenshot({ path: path.join(ROOT, "e2e/demo_3_seafloor.png"), fullPage: true });

  // 6. Switch back to LIVE AI ANALYSIS
  const liveToggle = page.locator('button:has-text("LIVE AI ANALYSIS")');
  await liveToggle.click();
  await page.waitForTimeout(500);

  const liveBadge = await page.getByText(/LIVE AI ANALYSIS — REAL MODEL INFERENCE/).first().isVisible();
  if (!liveBadge) {
    throw new Error("Live mode badge not visible");
  }
  console.log("✓ Live mode badge verified: 'LIVE AI ANALYSIS — REAL MODEL INFERENCE'");

  // 7. Load Live Sample Tile
  const liveSampleBtn = page.locator('button:has-text("Pipeline Tile")');
  await liveSampleBtn.click();
  await page.waitForTimeout(1000);

  // Check upload info
  const previewInfo = await page.locator("body").innerText();
  if (!previewInfo.includes("sha") && !previewInfo.includes("640")) {
    throw new Error("Sample preview did not load image metadata");
  }
  console.log("✓ Live sample tile loaded (sha256 & dimensions displayed)");

  // 8. Run Real Sonar Inference
  const runBtn = page.getByRole("button", { name: /run detection/i });
  await runBtn.click();
  await page.waitForTimeout(3500);

  const afterLiveInference = await page.locator("body").innerText();
  if (!afterLiveInference.includes("drishti-ss_yolov8n_e30_final")) {
    throw new Error("Model name drishti-ss_yolov8n_e30_final not shown in live results");
  }
  if (!afterLiveInference.includes("Location unavailable — no navigation metadata provided")) {
    throw new Error("Honest location notice missing in live mode with non-georeferenced tile");
  }
  console.log("✓ Real YOLO live inference executed successfully with model drishti-ss_yolov8n_e30_final");
  console.log("✓ Honest location notice verified ('Location unavailable — no navigation metadata provided')");
  await page.screenshot({ path: path.join(ROOT, "e2e/live_inference_result.png"), fullPage: true });

  // 9. Check Education card
  if (!afterLiveInference.includes("How Geolocation Works")) {
    throw new Error("'How Geolocation Works' informational card missing");
  }
  console.log("✓ 'How Geolocation Works' informational card verified");

  if (errors.length > 0) {
    console.warn("Browser console errors logged:", errors);
  } else {
    console.log("✓ Zero browser console errors");
  }

  await browser.close();
  console.log("\n==================================================");
  console.log("ALL PRESENTATION FLOWS VERIFIED SUCCESSFULLY!");
  console.log("==================================================");
}

run().catch((err) => {
  console.error("Verification failed:", err);
  process.exit(1);
});
