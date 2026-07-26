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

await build({
  entryPoints: [path.join(websiteDir, "src", "index.js")],
  outfile: path.join(vendorDir, "app.js"),
  bundle: true,
  format: "esm",
  platform: "browser",
  target: "es2020",
  sourcemap: false,
  minify: false,
  logLevel: "info",
  define: {
    "process.env.NODE_ENV": JSON.stringify("development"),
  },
});

await copyFile(
  require.resolve("reactflow/dist/style.css"),
  path.join(vendorDir, "reactflow.css"),
);
