import { chromium } from 'playwright';

const errors = [];
const browser = await chromium.launch({ channel: 'chrome', headless: true });
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
page.on('pageerror', e => errors.push('PAGEERROR: ' + e.message));
page.on('console', m => { if (m.type() === 'error') errors.push('CONSOLE: ' + m.text()); });

// 1. Dashboard/Workbench
await page.goto('http://localhost:5173/', { waitUntil: 'networkidle', timeout: 30000 });
console.log('TITLE:', await page.title());
console.log('H1/H2:', (await page.locator('h1, h2').allTextContents()).slice(0, 5));
await page.screenshot({ path: 'e2e/shot_workbench.png', fullPage: false });

// 2. Models page (evaluation metrics)
await page.goto('http://localhost:5173/models', { waitUntil: 'networkidle' });
console.log('MODELS H1/H2:', (await page.locator('h1, h2').allTextContents()).slice(0, 5));
await page.screenshot({ path: 'e2e/shot_models.png' });

// 3. History page
await page.goto('http://localhost:5173/history', { waitUntil: 'networkidle' });
console.log('HISTORY H1/H2:', (await page.locator('h1, h2').allTextContents()).slice(0, 5));
await page.screenshot({ path: 'e2e/shot_history.png' });

console.log('RUNTIME ERRORS:', errors.length ? errors : 'none');
await browser.close();
