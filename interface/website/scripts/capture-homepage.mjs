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
const viewportWidth = Number(process.env.VISUAL_WIDTH || 1440);
const viewportHeight = Number(process.env.VISUAL_HEIGHT || 2200);
const colorScheme = process.env.VISUAL_COLOR_SCHEME === "dark" ? "dark" : "light";

const browser = await chromium.launch({
  executablePath,
  headless: true,
});

try {
  const page = await browser.newPage({
    viewport: { width: viewportWidth, height: viewportHeight },
    colorScheme,
  });

  await page.goto(targetUrl, { waitUntil: "networkidle" });
  await page.screenshot({ path: outputPath, fullPage: true });
  console.log(outputPath);
} finally {
  await browser.close();
}
