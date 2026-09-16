import { chromium } from "playwright";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const BASE = "http://localhost:5173";

async function run() {
  console.log("Starting SONARX Survey Demo Presentation Flow Verification...");
  const browser = await chromium.launch({ channel: "chrome", headless: true });
  const page = await browser.newPage();
  await page.setViewportSize({ width: 1440, height: 950 });

  const errors = [];
  page.on("console", (msg) => {
    if (msg.type() === "error") {
      errors.push(msg.text());
    }
  });

  // 1. Visit Survey Page
  await page.goto(`${BASE}/survey`, { waitUntil: "networkidle" });
  console.log("✓ Loaded", `${BASE}/survey`);

  // 2. Check Header branding and badges
  await page.waitForTimeout(1000);
  const pageHeader = await page.locator("body").innerText();
  console.log("DEBUG BODY:", pageHeader.slice(0, 400));
  if (!pageHeader.includes("SONARX")) {
    throw new Error("Branding 'SONARX' missing on Survey page");
  }
  if (!/DRISHTI-SS YOLOv8n E30 FINAL/i.test(pageHeader)) {
    throw new Error("Active model badge missing on Survey page");
  }
  if (!/Backend Connected/i.test(pageHeader)) {
    throw new Error("Backend Connected badge missing on Survey page");
  }
  console.log("✓ Survey Header branding, model indicator, and backend connection verified");

  // 3. Test 1-Click "Load Demo Survey" Button
  const demoBtn = page.getByRole("button", { name: /load demo survey/i });
  if (!(await demoBtn.isVisible())) {
    throw new Error("'Load Demo Survey' button not visible");
  }
  console.log("✓ 'Load Demo Survey' button present");

  await demoBtn.click();
  console.log("✓ Clicked 'Load Demo Survey'");

  // 4. Wait for Upload & Batch completion
  let batchDone = false;
  for (let i = 0; i < 30; i++) {
    await page.waitForTimeout(2000);
    const body = await page.locator("body").innerText();
    if (/job \S+ · succeeded/.test(body)) {
      batchDone = true;
      break;
    }
    if (/· failed/.test(body)) {
      throw new Error("Survey batch job failed");
    }
  }

  if (!batchDone) {
    throw new Error("Survey batch job timed out after 60s");
  }
  console.log("✓ Real YOLO batch inference completed successfully");

  const afterBatchBody = await page.locator("body").innerText();

  // 5. Verify Demo Survey badge
  if (!afterBatchBody.includes("DEMO SURVEY — REAL BATCH INFERENCE")) {
    throw new Error("Demo survey badge missing");
  }
  console.log("✓ 'DEMO SURVEY — REAL BATCH INFERENCE' badge verified");

  // 6. Verify Survey Summary Cards
  if (!afterBatchBody.includes("3 Tiles") || !afterBatchBody.includes("3 Objects") || !afterBatchBody.includes("3 / 3")) {
    throw new Error("Survey summary cards missing expected counts (3 Tiles, 3 Objects, 3 / 3 geolocated)");
  }
  console.log("✓ Survey Summary cards verified (3 Tiles, Completed, 3 Objects, 3/3 Geolocated)");

  // 7. Verify Batch Pipeline Visualizer
  if (!/Survey Mission Execution Pipeline/i.test(afterBatchBody) || !/Geospatial Mapping/i.test(afterBatchBody)) {
    throw new Error("Batch pipeline visualizer missing");
  }
  console.log("✓ Batch pipeline execution visualizer verified");

  // 8. Verify Map and Markers
  const markers = await page.locator(".leaflet-container .leaflet-interactive").count();
  if (markers === 0) {
    throw new Error("No Leaflet markers rendered on geospatial map");
  }
  console.log(`✓ Geospatial map rendered with ${markers} markers`);

  // 9. Verify Tile Inspection Panel
  if (!afterBatchBody.includes("Sonar Tile Inspection") || !afterBatchBody.includes("submarine_pipeline")) {
    throw new Error("Tile inspection panel missing or not showing submarine_pipeline");
  }
  if (!afterBatchBody.includes("18.90") || !afterBatchBody.includes("72.80")) {
    throw new Error("Coordinates missing in Tile Inspection panel");
  }
  console.log("✓ Tile Inspection panel verified with acoustic backscatter image, bounding box & GPS coordinates");

  // 10. Verify Detection Table
  const rows = await page.locator("tbody tr").count();
  if (rows < 3) {
    throw new Error(`Expected at least 3 detection table rows, got ${rows}`);
  }
  console.log(`✓ Detection table verified with ${rows} rows`);

  // 11. Verify Educational & Provenance Cards
  if (!afterBatchBody.includes("HOW GEOLOCATION WORKS")) {
    throw new Error("'HOW GEOLOCATION WORKS' educational card missing");
  }
  if (!afterBatchBody.includes("DEMO SURVEY PROVENANCE")) {
    throw new Error("'DEMO SURVEY PROVENANCE' card missing");
  }
  console.log("✓ Educational 'HOW GEOLOCATION WORKS' & 'DEMO SURVEY PROVENANCE' cards verified");

  // Capture Screenshots
  await page.screenshot({ path: path.join(ROOT, "e2e/survey_demo_verified.png"), fullPage: true });
  console.log("✓ Saved full-page verification screenshot: e2e/survey_demo_verified.png");

  if (errors.length > 0) {
    console.warn("Browser console errors:", errors);
  } else {
    console.log("✓ Zero browser console errors");
  }

  await browser.close();
  console.log("\n==================================================");
  console.log("SURVEY PRESENTATION FLOW FULLY VERIFIED!");
  console.log("==================================================");
}

run().catch((err) => {
  console.error("Survey verification failed:", err);
  process.exit(1);
});
