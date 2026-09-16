import { chromium } from "playwright";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const artifactDir = "C:\\Users\\kr034\\.gemini\\antigravity-ide\\brain\\8dd617bd-53e3-49eb-a5c8-c41b514f32e2";

async function main() {
  const browser = await chromium.launch({ channel: "chrome", headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 960 } });

  // 1. Workbench page empty state in Live Mode
  console.log("Navigating to workbench (Live Mode)...");
  await page.goto("http://localhost:5173", { waitUntil: "networkidle" });
  await page.waitForTimeout(1000);
  await page.screenshot({ path: path.join(artifactDir, "demo_workbench_live_empty.png"), fullPage: false });

  // 2. Demo Mode - Pipeline Sample
  console.log("Switching to Demo Mode (Pipeline)...");
  await page.getByRole("button", { name: /DEMO MODE/i }).click();
  await page.waitForTimeout(600);
  await page.getByText("Submarine Pipeline").first().click();
  await page.waitForTimeout(1000);
  await page.screenshot({ path: path.join(artifactDir, "demo_workbench_demo_pipeline.png"), fullPage: false });

  // 3. Demo Mode - Shipwreck / Reef Sample
  console.log("Selecting Shipwreck / Reef Sample...");
  await page.getByText("Shipwreck / Reef").first().click();
  await page.waitForTimeout(1000);
  await page.screenshot({ path: path.join(artifactDir, "demo_workbench_demo_shipwreck.png"), fullPage: false });

  // 4. Demo Mode - Background Negative Sample
  console.log("Selecting Background Seafloor Sample...");
  await page.getByText("Background / Seafloor").first().click();
  await page.waitForTimeout(1000);
  await page.screenshot({ path: path.join(artifactDir, "demo_workbench_demo_background.png"), fullPage: false });

  // 5. Switch back to Live Mode and run real live inference
  console.log("Switching back to Live Mode and running real inference...");
  await page.getByRole("button", { name: /LIVE AI ANALYSIS/i }).click();
  await page.waitForTimeout(600);
  const samplePath = path.resolve(__dirname, "../datasets/processed/drishti-sss/test/images/pipe_1693569383.780_x3500.jpg");
  const fileInput = await page.locator('input[type="file"]').first();
  await fileInput.setInputFiles(samplePath);
  await page.waitForTimeout(2000);

  const [runResp] = await Promise.all([
    page.waitForResponse(
      (r) => r.url().includes("/detections/run") && r.request().method() === "POST",
      { timeout: 60000 }
    ),
    page.getByRole("button", { name: /run detection/i }).click(),
  ]);
  console.log("Live inference complete, status:", runResp.status());
  await page.waitForSelector('tbody tr', { timeout: 15000 });
  await page.waitForTimeout(1500);
  await page.screenshot({ path: path.join(artifactDir, "demo_workbench_live_active.png"), fullPage: false });

  await browser.close();
  console.log("All screenshots captured successfully!");
}

main().catch(err => {
  console.error("Error capturing screenshots:", err);
  process.exit(1);
});
