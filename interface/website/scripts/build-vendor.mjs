import { copyFile, mkdir } from "node:fs/promises";
import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { build } from "esbuild";

const rootDir = path.dirname(fileURLToPath(import.meta.url));
const websiteDir = path.resolve(rootDir, "..");
const vendorDir = path.join(websiteDir, "vendor");
const require = createRequire(import.meta.url);

await mkdir(vendorDir, { recursive: true });

const shared = {
  bundle: true,
  format: "esm",
  platform: "browser",
  target: "es2020",
  sourcemap: false,
  minify: false,
  logLevel: "info",
};

await build({
  ...shared,
  entryPoints: {
    react: path.join(websiteDir, "vendor-entries", "react.js"),
    "react-dom-client": path.join(websiteDir, "vendor-entries", "react-dom-client.js"),
    reactflow: path.join(websiteDir, "vendor-entries", "reactflow.js"),
    highlight: path.join(websiteDir, "vendor-entries", "highlight.js"),
  },
  outdir: vendorDir,
});

await copyFile(
  require.resolve("reactflow/dist/style.css"),
  path.join(vendorDir, "reactflow.css"),
);
