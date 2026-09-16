/**
 * Full browser E2E for the workbench (hackathon demo path).
 *
 * Verifies through the RUNNING UI + backend:
 *   upload -> preprocessing preview -> real model inference -> detection list
 *   -> bbox overlay alignment -> filtering status/reason -> honest geolocation
 *   -> CSV/JSON export -> report generation + download
 *
 * The overlay assertion is the important one: boxes are drawn in the coordinate
 * space of the image actually on screen, so a rendered rect must match the API's
 * bbox_processed_coords when the processed image is displayed (they differ by
 * the preprocessing letterbox offset whenever the source is not already 640x640).
 *
 * Usage:  node e2e_full.mjs            (needs backend :8000 + frontend :5173)
 */
import { chromium } from "playwright";
import path from "path";
import fs from "fs";

const ROOT = path.resolve(import.meta.dirname, "..");
const BASE = process.env.E2E_BASE_URL ?? "http://localhost:5173";
// 640x500 source tile: non-square on purpose (most DRISHTI tiles are), so a
// wrong coordinate space cannot accidentally line up.
const IMAGE = path.join(
  ROOT,
  "datasets/processed/drishti-sss/test/images/pipe_1693569383.780_x3500.jpg"
);

const failures = [];
const note = (ok, label, extra = "") => {
  console.log(`${ok ? "PASS" : "FAIL"}  ${label}${extra ? " :: " + extra : ""}`);
  if (!ok) failures.push(label + (extra ? ` (${extra})` : ""));
};

const errors = [];
const badResponses = [];
const browser = await chromium.launch({ channel: "chrome", headless: true });
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
page.on("pageerror", (e) => errors.push("PAGEERROR: " + e.message));
page.on("response", (r) => {
  if (r.status() >= 400) badResponses.push(`${r.status()} ${r.request().method()} ${r.url()}`);
});
page.on("console", (m) => {
  if (m.type() === "error") errors.push("CONSOLE: " + m.text());
});

// ---------------------------------------------------------------- 1. health
await page.goto(`${BASE}/`, { waitUntil: "networkidle", timeout: 30000 });
const loadedModel = await page.evaluate(async () => {
  const r = await fetch("/api/v1/health");
  const h = await r.json();
  return { version: h.model?.version, loaded: h.model?.loaded };
});
note(loadedModel.loaded === true, "backend reports a loaded model", String(loadedModel.version));

// ---------------------------------------------------------------- 2. upload
await page.locator('input[type="file"]').first().setInputFiles(IMAGE);
await page.waitForTimeout(2500);
const afterUpload = await page.locator("body").innerText();
note(/sha /.test(afterUpload), "upload shows sha256 + dimensions");

// ------------------------------------------------------------- 3. preprocess
await page.getByRole("button", { name: /run preview/i }).click();
await page.waitForTimeout(4000);
const afterPreview = await page.locator("body").innerText();
note(/resize_letterbox/i.test(afterPreview), "preprocessing preview lists applied ops");

// ------------------------------------------------------- 4. real inference
const [runResp] = await Promise.all([
  page.waitForResponse(
    (r) => r.url().includes("/detections/run") && r.request().method() === "POST",
    { timeout: 180000 }
  ),
  page.getByRole("button", { name: /run detection/i }).click(),
]);
const result = await runResp.json();
note(runResp.status() === 201, "POST /detections/run returned 201", `status=${runResp.status()}`);
note(
  result.model_version === loadedModel.version,
  "inference served by the loaded/active model",
  `${result.model_version}`
);

for (let i = 0; i < 12; i++) {
  if (/Detections \(\d+\)/.test(await page.locator("body").innerText())) break;
  await page.waitForTimeout(2000);
}
const body = await page.locator("body").innerText();
const detCount = Number(body.match(/Detections \((\d+)\)/)?.[1] ?? "-1");
note(detCount === result.detections.length, "UI detection count matches API", `ui=${detCount} api=${result.detections.length}`);
note(/why filtered/i.test(body), "filter reasons column is visible in the UI");
note(/(accepted|flagged|rejected)/i.test(body), "filtering status is visible in the UI");
note(/model .*final/is.test(body) || /model\s+\d+%/.test(body), "model vs final confidence both shown");
note(/Location unavailable — no navigation metadata provided/.test(body) || /unavailable \(unavailable\)/.test(body), "missing navigation metadata stated honestly");

// ----------------------------------------------------- 5. overlay alignment
const rendered = await page.evaluate(() => {
  const img = document.querySelector("img");
  const rects = [...document.querySelectorAll("svg rect")].map((r) => ({
    x: +r.getAttribute("x"),
    y: +r.getAttribute("y"),
    w: +r.getAttribute("width"),
    h: +r.getAttribute("height"),
  }));
  return {
    src: img?.src ?? "",
    naturalWidth: img?.naturalWidth ?? 0,
    naturalHeight: img?.naturalHeight ?? 0,
    rects,
  };
});
const showingProcessed = /\/processed/.test(rendered.src);
const space = showingProcessed ? "processed" : "source";
const expected = result.detections
  .map((d) => (space === "processed" ? d.bbox_processed_coords : d.bbox_source_coords))
  .filter(Boolean);
note(
  rendered.rects.length === expected.length,
  `overlay drew one box per detection (${space} space)`,
  `drawn=${rendered.rects.length} expected=${expected.length}`
);
// The overlay viewBox is the displayed image's natural size; a correct box must
// land inside it and match the API coordinates for the space on screen.
let worstDelta = 0;
for (const d of expected) {
  const match = rendered.rects.reduce(
    (best, r2) => {
      const delta =
        Math.abs(r2.x - d.x) + Math.abs(r2.y - d.y) + Math.abs(r2.w - d.w) + Math.abs(r2.h - d.h);
      return delta < best.delta ? { delta, box: r2 } : best;
    },
    { delta: Infinity, box: null }
  );
  worstDelta = Math.max(worstDelta, match.delta);
}
note(
  expected.length === 0 || worstDelta < 1.0,
  "rendered boxes match API coordinates for the displayed space",
  `worst |Δ| = ${worstDelta.toFixed(2)} px`
);
for (const r2 of rendered.rects) {
  const inside =
    r2.x >= -1 && r2.y >= -1 &&
    r2.x + r2.w <= rendered.naturalWidth + 1 &&
    r2.y + r2.h <= rendered.naturalHeight + 1;
  note(inside, "every box lies inside the rendered image", `img=${rendered.naturalWidth}x${rendered.naturalHeight}`);
}

// ------------------------------------------- 6. selection links list <-> box
if (expected.length > 0) {
  const strokeBefore = await page.evaluate(() => {
    const r = document.querySelector("svg rect");
    return { width: r.getAttribute("stroke-width"), parentOpacity: r.parentElement.getAttribute("opacity") };
  });
  await page.locator("tbody tr").first().click();
  await page.waitForTimeout(400);
  const strokeAfter = await page.evaluate(() => {
    const r = document.querySelector("svg rect");
    return { width: r.getAttribute("stroke-width"), parentOpacity: r.parentElement.getAttribute("opacity") };
  });
  note(
    strokeBefore.width !== strokeAfter.width || strokeBefore.parentOpacity !== strokeAfter.parentOpacity,
    "clicking a detection row highlights its box",
    `${strokeBefore.width}/${strokeBefore.parentOpacity} -> ${strokeAfter.width}/${strokeAfter.parentOpacity}`
  );
}

// ------------------------------------- 6b. detector operating-point control
// The threshold select is a real user control, not decoration: the chosen value
// must reach the API as an override, the API's post-floor applied value must come
// back, and the UI must show what actually ran. Covered here because a control
// that silently does nothing would look fine in a screenshot but be dishonest in
// a demo.
const thrSelect = page.locator("select").first();
const thrCount = await thrSelect.count();
note(thrCount > 0, "detector threshold control is present");
if (thrCount > 0) {
  await thrSelect.selectOption("0.05");
  const [lowResp] = await Promise.all([
    page.waitForResponse(
      (r) => r.url().includes("/detections/run") && r.request().method() === "POST",
      { timeout: 180000 }
    ),
    page.getByRole("button", { name: /run detection/i }).click(),
  ]);
  const sent = JSON.parse(lowResp.request().postData() || "{}");
  const lowJson = await lowResp.json();
  note(
    sent?.overrides?.confidence_threshold === 0.05,
    "lower threshold is sent as a detector override",
    `sent=${JSON.stringify(sent.overrides ?? null)}`
  );
  note(
    lowJson.applied_confidence_threshold === 0.05,
    "API reports the applied operating point",
    `applied=${lowJson.applied_confidence_threshold}`
  );
  for (let i = 0; i < 12; i++) {
    if (/Detections \(\d+\)/.test(await page.locator("body").innerText())) break;
    await page.waitForTimeout(2000);
  }
  note(
    /threshold 0\.05/.test(await page.locator("body").innerText()),
    "UI shows the threshold that actually ran"
  );
  // Restore the shipped default so the remaining flow (and stored state) uses it.
  await thrSelect.selectOption("0.25");
  const [defResp] = await Promise.all([
    page.waitForResponse(
      (r) => r.url().includes("/detections/run") && r.request().method() === "POST",
      { timeout: 180000 }
    ),
    page.getByRole("button", { name: /run detection/i }).click(),
  ]);
  const defJson = await defResp.json();
  note(
    defJson.applied_confidence_threshold === 0.25,
    "default operating point restored and reported",
    `applied=${defJson.applied_confidence_threshold}`
  );
}

// ------------------------------------------------------------- 7. map/geo
note(
  /No geolocated detections|geolocated/.test(await page.locator("body").innerText()),
  "map states geolocation availability honestly"
);

// ---------------------------------------------------------------- 8. exports
for (const [label, hrefNeedle] of [["CSV", "detections.csv"], ["JSON", "detections.json"]]) {
  const link = page.locator(`a[href*="${hrefNeedle}"]`).first();
  const ok = (await link.count()) > 0;
  note(ok, `${label} export link present`);
  if (ok) {
    const href = await link.getAttribute("href");
    const resp = await page.request.get(BASE + href);
    const text = await resp.text();
    note(resp.status() === 200 && text.length > 0, `${label} export downloads`, `${text.length} bytes`);
    note(/model_version/.test(text), `${label} export carries model provenance`);
    note(/geo_status/.test(text), `${label} export carries geo_status`);
  }
}

// ---------------------------------------------------------------- 9. report
const genBtn = page.getByRole("button", { name: /generate report/i }).first();
if ((await genBtn.count()) > 0) {
  await genBtn.click();
  let reportReady = false;
  for (let i = 0; i < 15; i++) {
    await page.waitForTimeout(3000);
    if (/report ready/i.test(await page.locator("body").innerText())) {
      reportReady = true;
      break;
    }
  }
  note(reportReady, "report generated and downloadable from the UI");
  if (reportReady) {
    const links = page.locator('a[href*="/reports/"]');
    const seen = new Set();
    for (let i = 0; i < (await links.count()); i++) {
      const href = await links.nth(i).getAttribute("href");
      if (!href || seen.has(href)) continue;
      seen.add(href);
      const resp = await page.request.get(BASE + href);
      const body2 = await resp.body();
      const isPdf = body2.subarray(0, 5).toString() === "%PDF-";
      note(resp.status() === 200 && body2.length > 500, `report artifact downloads (${href.slice(-30)})`, `${body2.length} bytes`);
      if (/format=pdf/.test(href)) note(isPdf, "PDF report has a valid PDF header");
      if (!/format=pdf/.test(href)) {
        const html = body2.toString();
        note(/bbox|x,y,w,h/i.test(html), "HTML report includes bounding boxes");
        note(/Model conf/i.test(html) && /Final conf/i.test(html), "HTML report includes both confidences");
        note(/Filter reason|Reasons/i.test(html), "HTML report includes filter reasons");
        // The operating point is user-selectable, so the report must state the
        // threshold the detections were produced at, not just the model version.
        note(
          /Detector confidence threshold:\s*0\.\d+/.test(html),
          "HTML report records the detector confidence threshold"
        );
      }
    }
  }
} else {
  note(false, "Generate report button present");
}

async function safeScreenshot(filePath, fullPage = true) {
  try {
    await page.screenshot({ path: filePath, fullPage });
  } catch (err) {
    try {
      await new Promise((r) => setTimeout(r, 400));
      await page.screenshot({ path: filePath, fullPage });
    } catch (e2) {
      console.warn(`[warn] Screenshot write skipped (${path.basename(filePath)}): ${e2.message}`);
    }
  }
}

await safeScreenshot(path.join(ROOT, "e2e/e2e_workbench.png"), true);

// ------------------------------------------------------- 10. other pages
await page.goto(`${BASE}/models`, { waitUntil: "networkidle" });
const modelsBody = await page.locator("body").innerText();
note(/Evaluation metrics/.test(modelsBody), "models page renders metrics");
note(/Split: TEST/i.test(modelsBody), "metrics are labelled with the evaluation split");
note(/Confusion matrix/i.test(modelsBody), "confusion matrix is displayed");
note(/training provenance/i.test(modelsBody), "model/training provenance is displayed");
await safeScreenshot(path.join(ROOT, "e2e/e2e_models.png"), true);

await page.goto(`${BASE}/history`, { waitUntil: "networkidle" });
note(/Detection history/.test(await page.locator("body").innerText()), "history page renders");
await safeScreenshot(path.join(ROOT, "e2e/e2e_history.png"), true);

// --------------------------- 12. survey batch + geolocation (real nav path)
const FIXTURE = path.join(ROOT, "e2e/fixtures/geo_survey_demo.zip");
if (fs.existsSync(FIXTURE)) {
  await page.goto(`${BASE}/survey`, { waitUntil: "networkidle" });
  note(/Upload survey archive/i.test(await page.locator("body").innerText()), "survey page renders");

  await page.locator('input[type="file"]').first().setInputFiles(FIXTURE);
  await page.waitForTimeout(4000);
  const afterSurveyUpload = await page.locator("body").innerText();
  note(/navigation:\s*present/i.test(afterSurveyUpload), "survey reports navigation metadata present");

  await page.getByRole("button", { name: /run batch detection/i }).click();
  let batchDone = false;
  for (let i = 0; i < 30; i++) {
    await page.waitForTimeout(2000);
    const b = await page.locator("body").innerText();
    if (/job \S+ · succeeded/.test(b)) {
      batchDone = true;
      break;
    }
    if (/· failed/.test(b)) break;
  }
  note(batchDone, "survey batch job succeeded");
  if (batchDone) {
    const surveyBody = await page.locator("body").innerText();
    const geoCount = Number(surveyBody.match(/(\d+) with coordinates/)?.[1] ?? "0");
    note(geoCount > 0, "survey geolocation produced real coordinates from the nav sidecar", `${geoCount} geolocated`);
    note(/±/.test(surveyBody), "coordinate uncertainty is displayed");
    const markers = await page.locator(".leaflet-container .leaflet-interactive").count();
    note(markers > 0, "map renders a marker per geolocated detection", `${markers} markers`);
    const scopedCsv = page.locator('a[href*="detections.csv"]').first();
    const href = await scopedCsv.getAttribute("href");
    note(/survey_id=/.test(href ?? ""), "survey-scoped export link carries the survey filter");
    const resp = await page.request.get(BASE + href);
    const rows = (await resp.text()).split("\n").filter((l) => l.trim()).length - 1;
    note(resp.status() === 200 && rows > 0, "survey-scoped CSV export downloads", `${rows} rows`);
  }
  await safeScreenshot(path.join(ROOT, "e2e/e2e_survey.png"), true);
} else {
  note(false, "survey fixture present (run scripts/make_e2e_survey_fixture.py)");
}

// ------------------------------------------------------------- 11. clean run
note(badResponses.length === 0, "no 4xx/5xx responses during the whole flow", badResponses.join(" | "));
note(errors.length === 0, "no console/runtime errors", errors.join(" | "));

await browser.close();

console.log("\n" + "=".repeat(66));
if (failures.length) {
  console.log(`E2E FAILURES (${failures.length}):`);
  failures.forEach((f) => console.log("  - " + f));
  process.exit(1);
}
console.log("E2E PASSED: upload -> preview -> real inference -> overlay -> exports -> report verified");
