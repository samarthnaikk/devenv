import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const websiteDir = path.resolve(__dirname, "..");
const outputPath = process.argv[2]
  ? path.resolve(process.argv[2])
  : path.join(websiteDir, "artifacts", "homepage.png");
const targetUrl = process.env.VISUAL_URL || "http://127.0.0.1:4173/";
const executablePath = process.env.PLAYWRIGHT_CHROME_PATH || "/usr/sbin/google-chrome-stable";

const browser = await chromium.launch({
  executablePath,
  headless: true,
});

try {
  const page = await browser.newPage({
    viewport: { width: 1440, height: 2200 },
    colorScheme: "light",
  });

  await page.goto(targetUrl, { waitUntil: "networkidle" });
  await page.screenshot({ path: outputPath, fullPage: true });
  console.log(outputPath);
} finally {
  await browser.close();
}
