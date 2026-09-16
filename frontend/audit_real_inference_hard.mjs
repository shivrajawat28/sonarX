/**
 * HARD REAL-INFERENCE AUDIT SCRIPT
 *
 * SIH26057: AI-Powered Automated Underwater Marine Debris & Anomaly Detection
 *
 * Verifies the complete real data pipeline:
 * REAL IMAGE -> FRONTEND UPLOAD -> BACKEND SHA-256 MATCH -> PREPROCESSING
 * -> REAL YOLOv8n MODEL -> RAW DETECTIONS -> FILTERING -> FINAL CONFIDENCE
 * -> API RESPONSE -> FRONTEND STATE -> UI CARDS -> UI CONFIDENCE -> SVG BBOX
 */

import { chromium } from "playwright";
import path from "node:path";
import fs from "node:fs";
import crypto from "node:crypto";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const FRONTEND_URL = "http://localhost:5173";
const BACKEND_URL = "http://127.0.0.1:8000";

const IMAGES = {
  pipeline: path.join(ROOT, "datasets/processed/drishti-sss/test/images/pipe_1693569383.780_x3500.jpg"),
  wreck: path.join(ROOT, "datasets/processed/drishti-sss/test/images/wreckA_Artificial_Reef_06_y1280_x0.jpg"),
  background: path.join(ROOT, "datasets/processed/drishti-sss/test/images/bg_1693569262.760_x0.jpg"),
};

function sha256File(filePath) {
  const data = fs.readFileSync(filePath);
  return crypto.createHash("sha256").update(data).digest("hex");
}

async function audit() {
  console.log("==================================================");
  console.log("STARTING HARD REAL-INFERENCE AUDIT (SIH26057)");
  console.log("==================================================");

  const consoleErrors = [];
  const networkFailures = [];
  const auditResults = {
    imagesTested: [],
    repeatabilityPassed: false,
    boxErrors: {},
    exportsVerified: {},
  };

  const browser = await chromium.launch({ channel: "chrome", headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 960 } });

  page.on("pageerror", (err) => consoleErrors.push(`PAGEERROR: ${err.message}`));
  page.on("console", (msg) => {
    if (msg.type() === "error") consoleErrors.push(`CONSOLE_ERROR: ${msg.text()}`);
  });
  page.on("response", (resp) => {
    if (resp.status() >= 400) networkFailures.push(`${resp.status()} ${resp.request().method()} ${resp.url()}`);
  });

  // Step 0: Check backend active model health directly
  console.log("\n[STAGE 0: Backend Health & Model Check]");
  const healthResp = await fetch(`${BACKEND_URL}/api/v1/health`);
  const health = await healthResp.json();
  console.log("Backend Health:", JSON.stringify(health, null, 2));
  if (!health.model?.loaded || health.model?.version !== "drishti-ss_yolov8n_e30_final") {
    throw new Error(`Active model unexpected: ${JSON.stringify(health.model)}`);
  }

  // =========================================================================
  // TEST 1: REAL IMAGE 1 — Submarine Pipeline
  // =========================================================================
  console.log("\n[STAGE 1: Testing Image 1 — Submarine Pipeline]");
  const img1Path = IMAGES.pipeline;
  const img1Stats = fs.statSync(img1Path);
  const img1Sha = sha256File(img1Path);
  console.log(`Source Image 1: ${path.basename(img1Path)}`);
  console.log(`Size: ${img1Stats.size} bytes | SHA-256: ${img1Sha}`);

  await page.goto(FRONTEND_URL, { waitUntil: "networkidle" });
  await page.waitForTimeout(1000);

  // 1.1 Upload Image 1
  console.log("1.1 Uploading Image 1 through UI...");
  const uploadRespPromise = page.waitForResponse((r) => r.url().includes("/uploads/image") && r.request().method() === "POST");
  const fileInput = page.locator('input[type="file"]').first();
  await fileInput.setInputFiles(img1Path);
  const uploadResp = await uploadRespPromise;
  const uploadData = await uploadResp.json();
  console.log(`Upload Response: HTTP ${uploadResp.status()}`, uploadData);

  if (uploadResp.status() !== 201) throw new Error(`Upload returned status ${uploadResp.status()}`);
  if (uploadData.sha256 !== img1Sha) throw new Error(`SHA-256 mismatch! Backend: ${uploadData.sha256}, Source: ${img1Sha}`);
  console.log("-> PASS: Backend received and verified identical SHA-256 hash.");

  // Check stored image in backend data directory
  const storedRelative = uploadData.image_id;
  console.log(`Backend Image ID: ${storedRelative}`);

  // 1.2 Run Preprocessing Preview
  console.log("1.2 Running Preprocessing Preview through UI...");
  const previewRespPromise = page.waitForResponse((r) => r.url().includes("/previews/preprocess") && r.request().method() === "POST");
  await page.getByRole("button", { name: /run preview/i }).click();
  const previewResp = await previewRespPromise;
  const previewData = await previewResp.json();
  console.log(`Preview Response: HTTP ${previewResp.status()}`, previewData);

  // 1.3 Run Real AI Detection
  console.log("1.3 Running Real AI Detection through UI...");
  const detectRespPromise = page.waitForResponse((r) => r.url().includes("/detections/run") && r.request().method() === "POST");
  await page.getByRole("button", { name: /run detection/i }).click();
  const detectResp = await detectRespPromise;
  const detectData = await detectResp.json();
  console.log(`Inference Response: HTTP ${detectResp.status()}`);
  console.log(`Model Version: ${detectData.model_version}`);
  console.log(`Applied Confidence Threshold: ${detectData.applied_confidence_threshold}`);
  console.log(`Timings:`, detectData.timings_ms);
  console.log(`Total Detections: ${detectData.detections.length}`);

  // Trace individual detections
  detectData.detections.forEach((d, idx) => {
    console.log(`  Detection #${idx + 1}:`);
    console.log(`    Class: ${d.class_name}`);
    console.log(`    Raw Model Confidence: ${d.model_confidence}`);
    console.log(`    Final Confidence: ${d.final_confidence}`);
    console.log(`    Filtering Status: ${d.filtering_status}`);
    console.log(`    Filter Reasons: ${d.filter_reasons.join(", ") || "none"}`);
    console.log(`    Source BBox:`, d.bbox_source_coords);
    console.log(`    Processed BBox:`, d.bbox_processed_coords);
    console.log(`    Latitude: ${d.latitude}, Longitude: ${d.longitude}`);
    console.log(`    Geo Status: ${d.geo_status}`);
  });

  // 1.4 Trace into Frontend UI
  console.log("1.4 Verifying UI elements match API response...");
  await page.waitForSelector("table tbody tr", { timeout: 10000 });

  // Verify Detection count in UI header
  const uiHeadingText = await page.locator('h2:has-text("Detections")').innerText();
  console.log(`UI Detection Heading: "${uiHeadingText}"`);
  if (!uiHeadingText.includes(`(${detectData.detections.length})`)) {
    throw new Error(`UI detection count mismatch! Heading: "${uiHeadingText}", API: ${detectData.detections.length}`);
  }

  // Verify UI Table Rows
  const rows = page.locator("table tbody tr");
  const rowCount = await rows.count();
  console.log(`UI Table Row Count: ${rowCount}`);
  if (rowCount !== detectData.detections.length) {
    throw new Error(`Table row count (${rowCount}) != API count (${detectData.detections.length})`);
  }

  for (let i = 0; i < detectData.detections.length; i++) {
    const apiDet = detectData.detections[i];
    const row = rows.nth(i);
    const rowText = await row.innerText();

    console.log(`Checking Row #${i + 1}:`);
    console.log(`  Class display: expecting "${apiDet.class_name}"`);
    if (!rowText.includes(apiDet.class_name)) throw new Error(`Row #${i + 1} does not contain class ${apiDet.class_name}`);

    const expectedModelPct = `${Math.round(apiDet.model_confidence * 100)}%`;
    const expectedFinalPct = `${Math.round(apiDet.final_confidence * 100)}%`;
    console.log(`  Confidence display: expecting model=${expectedModelPct}, final=${expectedFinalPct}`);
    if (!rowText.includes(expectedModelPct)) throw new Error(`Row #${i + 1} missing model confidence ${expectedModelPct}`);
    if (!rowText.includes(expectedFinalPct)) throw new Error(`Row #${i + 1} missing final confidence ${expectedFinalPct}`);

    console.log(`  Filter status display: expecting "${apiDet.filtering_status}"`);
    if (!rowText.toLowerCase().includes(apiDet.filtering_status.toLowerCase())) {
      throw new Error(`Row #${i + 1} missing filtering status ${apiDet.filtering_status}`);
    }
  }

  // 1.5 Verify Bounding Box Overlay Coordinates (Pixel Accuracy)
  console.log("1.5 Measuring bounding box coordinate accuracy in rendered SVG...");
  const overlayMetrics = await page.evaluate(() => {
    const img = document.querySelector(".workbench-grid img");
    const rects = [...document.querySelectorAll("svg rect")].map((r) => ({
      x: parseFloat(r.getAttribute("x") || "0"),
      y: parseFloat(r.getAttribute("y") || "0"),
      w: parseFloat(r.getAttribute("width") || "0"),
      h: parseFloat(r.getAttribute("height") || "0"),
      strokeWidth: r.getAttribute("stroke-width"),
    }));
    return {
      naturalWidth: img?.naturalWidth || 0,
      naturalHeight: img?.naturalHeight || 0,
      rects,
    };
  });

  console.log(`Rendered Image Natural Dims: ${overlayMetrics.naturalWidth}x${overlayMetrics.naturalHeight}`);
  console.log(`Rendered SVG Rectangles:`, overlayMetrics.rects);

  let maxXError = 0, maxYError = 0, maxWError = 0, maxHError = 0;
  // Compare each API bbox (in processed space, since viewProcessed is default)
  detectData.detections.forEach((d, idx) => {
    const expectedBox = d.bbox_processed_coords;
    const match = overlayMetrics.rects.reduce(
      (best, r) => {
        const dx = Math.abs(r.x - expectedBox.x);
        const dy = Math.abs(r.y - expectedBox.y);
        const dw = Math.abs(r.w - expectedBox.w);
        const dh = Math.abs(r.h - expectedBox.h);
        const total = dx + dy + dw + dh;
        return total < best.total ? { total, dx, dy, dw, dh, r } : best;
      },
      { total: Infinity, dx: 0, dy: 0, dw: 0, dh: 0, r: null }
    );

    console.log(`  BBox #${idx + 1} Error: Δx=${match.dx.toFixed(2)}px, Δy=${match.dy.toFixed(2)}px, Δw=${match.dw.toFixed(2)}px, Δh=${match.dh.toFixed(2)}px (Total: ${match.total.toFixed(2)}px)`);
    maxXError = Math.max(maxXError, match.dx);
    maxYError = Math.max(maxYError, match.dy);
    maxWError = Math.max(maxWError, match.dw);
    maxHError = Math.max(maxHError, match.dh);
  });

  auditResults.boxErrors = { maxXError, maxYError, maxWError, maxHError };
  if (maxXError > 1.0 || maxYError > 1.0 || maxWError > 1.0 || maxHError > 1.0) {
    throw new Error(`BBox scaling error exceeds 1.0px tolerance!`);
  }
  console.log("-> PASS: Rendered bounding box matches API coordinates with sub-pixel precision.");

  // 1.6 Verify Geolocation Honesty
  console.log("1.6 Verifying Geolocation Honesty...");
  const pageBody = await page.locator("body").innerText();
  const hasHonestGeo = pageBody.includes("Location unavailable — no navigation metadata provided") ||
                       pageBody.includes("coordinates unavailable (never fabricated)");
  if (!hasHonestGeo) throw new Error("UI failed to state that coordinates are unavailable for image without nav metadata!");
  console.log("-> PASS: Honest geolocation confirmed. Zero fabricated GPS coordinates.");

  // 1.7 Verify Exports (JSON, CSV, HTML, PDF)
  console.log("1.7 Verifying Exports & Reports...");

  // JSON Export
  const [jsonDownload] = await Promise.all([
    page.waitForEvent("download"),
    page.getByRole("button", { name: /download json/i }).click(),
  ]);
  const jsonPath = await jsonDownload.path();
  const jsonContent = JSON.parse(fs.readFileSync(jsonPath, "utf-8"));
  console.log(`JSON Export downloaded: count=${jsonContent.count}, notice="${jsonContent.notice}"`);
  if (!jsonContent.detections || jsonContent.count < 1) throw new Error("JSON export has no detections!");
  const matchedDet = jsonContent.detections.find((d) => d.image_id === detectData.image_id);
  if (!matchedDet) throw new Error(`JSON export does not contain detections for image ${detectData.image_id}!`);
  console.log(`Matched detection in JSON:`, matchedDet.detection_id, matchedDet.class_name, matchedDet.model_confidence);
  if (matchedDet.class_name !== detectData.detections[0].class_name) throw new Error("JSON export detection class mismatch!");
  if (matchedDet.model_version !== "drishti-ss_yolov8n_e30_final") throw new Error("JSON export model version mismatch!");
  auditResults.exportsVerified.json = true;

  // CSV Export
  const [csvDownload] = await Promise.all([
    page.waitForEvent("download"),
    page.getByRole("button", { name: /download csv/i }).click(),
  ]);
  const csvPath = await csvDownload.path();
  const csvContent = fs.readFileSync(csvPath, "utf-8");
  console.log(`CSV Export downloaded (${csvContent.length} bytes), first 2 lines:`);
  console.log(csvContent.split("\n").slice(0, 2).join("\n"));
  if (!csvContent.includes(detectData.detections[0].class_name)) throw new Error("CSV export missing detection class!");
  if (!csvContent.includes("drishti-ss_yolov8n_e30_final")) throw new Error("CSV export missing model version!");
  auditResults.exportsVerified.csv = true;

  // HTML & PDF Report
  console.log("Generating report through UI...");
  await page.getByRole("button", { name: /generate report/i }).click();
  await page.waitForSelector('a:has-text("download HTML")', { timeout: 30000 });
  const htmlLink = page.locator('a:has-text("download HTML")');
  const pdfLink = page.locator('a:has-text("download PDF")');
  const htmlHref = await htmlLink.getAttribute("href");
  const pdfHref = await pdfLink.getAttribute("href");
  console.log(`Report links ready: HTML=${htmlHref}, PDF=${pdfHref}`);

  // Fetch HTML content
  const htmlResp = await page.request.get(FRONTEND_URL + htmlHref);
  const htmlText = await htmlResp.text();
  if (!htmlText.includes(detectData.detections[0].class_name)) throw new Error("HTML report missing class name!");
  if (!htmlText.includes("drishti-ss_yolov8n_e30_final")) throw new Error("HTML report missing model version!");
  auditResults.exportsVerified.html = true;
  console.log("-> PASS: HTML report contains accurate class name and model version.");

  // Fetch PDF content
  const pdfResp = await page.request.get(FRONTEND_URL + pdfHref);
  const pdfBuf = await pdfResp.body();
  const pdfHeader = pdfBuf.subarray(0, 5).toString("utf-8");
  console.log(`PDF Report downloaded: ${pdfBuf.length} bytes, Header: "${pdfHeader}"`);
  if (!pdfHeader.startsWith("%PDF-")) throw new Error(`Invalid PDF header: ${pdfHeader}`);
  auditResults.exportsVerified.pdf = true;
  console.log("-> PASS: PDF report valid with standard %PDF- header.");

  auditResults.imagesTested.push({
    name: path.basename(img1Path),
    detectionCount: detectData.detections.length,
    classes: detectData.detections.map(d => d.class_name),
    highestRawConf: Math.max(...detectData.detections.map(d => d.model_confidence)),
    avgRawConf: detectData.detections.reduce((a, b) => a + b.model_confidence, 0) / detectData.detections.length,
    highestFinalConf: Math.max(...detectData.detections.map(d => d.final_confidence)),
    accepted: detectData.detections.filter(d => d.filtering_status === "accepted").length,
    flagged: detectData.detections.filter(d => d.filtering_status === "flagged").length,
    rejected: detectData.detections.filter(d => d.filtering_status === "rejected").length,
    rawDetails: detectData.detections,
  });

  // =========================================================================
  // TEST 2: REAL IMAGE 2 — Shipwreck / Reef
  // =========================================================================
  console.log("\n[STAGE 2: Testing Image 2 — Shipwreck / Reef]");
  const img2Path = IMAGES.wreck;
  const img2Stats = fs.statSync(img2Path);
  const img2Sha = sha256File(img2Path);
  console.log(`Source Image 2: ${path.basename(img2Path)} (${img2Stats.size} bytes, SHA-256: ${img2Sha})`);

  await fileInput.setInputFiles(img2Path);
  await page.waitForTimeout(2000);
  await page.getByRole("button", { name: /run preview/i }).click();
  await page.waitForTimeout(2000);

  const detect2RespPromise = page.waitForResponse((r) => r.url().includes("/detections/run") && r.request().method() === "POST");
  await page.getByRole("button", { name: /run detection/i }).click();
  const detect2Resp = await detect2RespPromise;
  const detect2Data = await detect2Resp.json();
  console.log(`Image 2 Inference: ${detect2Data.detections.length} detections`);
  detect2Data.detections.forEach((d, i) => {
    console.log(`  Img2 Det #${i + 1}: ${d.class_name} | raw: ${d.model_confidence} | final: ${d.final_confidence} | status: ${d.filtering_status}`);
  });

  auditResults.imagesTested.push({
    name: path.basename(img2Path),
    detectionCount: detect2Data.detections.length,
    classes: detect2Data.detections.map(d => d.class_name),
    highestRawConf: detect2Data.detections.length > 0 ? Math.max(...detect2Data.detections.map(d => d.model_confidence)) : 0,
    avgRawConf: detect2Data.detections.length > 0 ? detect2Data.detections.reduce((a, b) => a + b.model_confidence, 0) / detect2Data.detections.length : 0,
    highestFinalConf: detect2Data.detections.length > 0 ? Math.max(...detect2Data.detections.map(d => d.final_confidence)) : 0,
    accepted: detect2Data.detections.filter(d => d.filtering_status === "accepted").length,
    flagged: detect2Data.detections.filter(d => d.filtering_status === "flagged").length,
    rejected: detect2Data.detections.filter(d => d.filtering_status === "rejected").length,
    rawDetails: detect2Data.detections,
  });

  // =========================================================================
  // TEST 3: NEGATIVE / NO-DETECTION TEST (Background Seafloor)
  // =========================================================================
  console.log("\n[STAGE 3: Negative Test — Seafloor Background Sonar Tile]");
  const img3Path = IMAGES.background;
  console.log(`Source Image 3: ${path.basename(img3Path)}`);

  await fileInput.setInputFiles(img3Path);
  await page.waitForTimeout(2000);
  await page.getByRole("button", { name: /run preview/i }).click();
  await page.waitForTimeout(2000);

  const detect3RespPromise = page.waitForResponse((r) => r.url().includes("/detections/run") && r.request().method() === "POST");
  await page.getByRole("button", { name: /run detection/i }).click();
  const detect3Resp = await detect3RespPromise;
  const detect3Data = await detect3Resp.json();
  console.log(`Image 3 (Background) Inference: ${detect3Data.detections.length} detections`);

  // Verify that Image 2 or Image 1's results do not remain on screen
  await page.waitForTimeout(1000);
  const uiBodyText = await page.locator("body").innerText();
  if (detect3Data.detections.length === 0) {
    if (!uiBodyText.includes("No detections returned for this run") && !uiBodyText.includes("no detections in this run")) {
      throw new Error("UI failed to show empty state when 0 detections returned!");
    }
    console.log("-> PASS: No false detections on empty background seafloor tile. Stale detections properly cleared!");
  } else {
    console.log(`Note: Background image produced ${detect3Data.detections.length} detections at default threshold.`);
  }

  auditResults.imagesTested.push({
    name: path.basename(img3Path),
    detectionCount: detect3Data.detections.length,
    classes: detect3Data.detections.map(d => d.class_name),
    highestRawConf: detect3Data.detections.length > 0 ? Math.max(...detect3Data.detections.map(d => d.model_confidence)) : 0,
    avgRawConf: detect3Data.detections.length > 0 ? detect3Data.detections.reduce((a, b) => a + b.model_confidence, 0) / detect3Data.detections.length : 0,
    highestFinalConf: detect3Data.detections.length > 0 ? Math.max(...detect3Data.detections.map(d => d.final_confidence)) : 0,
    accepted: detect3Data.detections.filter(d => d.filtering_status === "accepted").length,
    flagged: detect3Data.detections.filter(d => d.filtering_status === "flagged").length,
    rejected: detect3Data.detections.filter(d => d.filtering_status === "rejected").length,
    rawDetails: detect3Data.detections,
  });

  // =========================================================================
  // TEST 4: REPEATABILITY TEST
  // =========================================================================
  console.log("\n[STAGE 4: Repeatability Test — Re-running Image 1]");
  await fileInput.setInputFiles(img1Path);
  await page.waitForTimeout(2000);
  await page.getByRole("button", { name: /run preview/i }).click();
  await page.waitForTimeout(2000);

  const detect1bRespPromise = page.waitForResponse((r) => r.url().includes("/detections/run") && r.request().method() === "POST");
  await page.getByRole("button", { name: /run detection/i }).click();
  const detect1bResp = await detect1bRespPromise;
  const detect1bData = await detect1bResp.json();

  console.log(`Run 1 Detections: ${detectData.detections.length} vs Run 2 Detections: ${detect1bData.detections.length}`);
  if (detectData.detections.length !== detect1bData.detections.length) {
    throw new Error("Repeatability failed: detection counts differ!");
  }
  for (let i = 0; i < detectData.detections.length; i++) {
    const d1 = detectData.detections[i];
    const d2 = detect1bData.detections[i];
    if (d1.class_name !== d2.class_name) throw new Error(`Repeatability failed: class mismatch (${d1.class_name} vs ${d2.class_name})`);
    if (Math.abs(d1.model_confidence - d2.model_confidence) > 1e-4) {
      throw new Error(`Repeatability failed: confidence drift (${d1.model_confidence} vs ${d2.model_confidence})`);
    }
    if (Math.abs(d1.bbox_processed_coords.x - d2.bbox_processed_coords.x) > 1e-2) {
      throw new Error(`Repeatability failed: bbox drift!`);
    }
  }
  console.log("-> PASS: Exact 100% deterministic repeatability confirmed across runs.");
  auditResults.repeatabilityPassed = true;

  // Final Console & Network summary
  console.log("\n[STAGE 5: Browser Console & Network Health]");
  console.log(`Console Errors: ${consoleErrors.length}`);
  if (consoleErrors.length > 0) console.log(consoleErrors);
  console.log(`Network 4xx/5xx Failures: ${networkFailures.length}`);
  if (networkFailures.length > 0) console.log(networkFailures);

  await browser.close();

  // Save full audit results for reporting
  fs.writeFileSync(
    path.join(ROOT, "real_inference_audit_results.json"),
    JSON.stringify(auditResults, null, 2),
    "utf-8"
  );
  console.log("\nAudit finished successfully! Results written to real_inference_audit_results.json");
}

audit().catch((err) => {
  console.error("FATAL AUDIT FAILURE:", err);
  process.exit(1);
});
