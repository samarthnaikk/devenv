import hljs from "https://esm.sh/highlight.js@11.11.1";
import { renderMarkdown } from "./markdown.js";

const LANGUAGE_BY_EXTENSION = {
  ".bash": "bash",
  ".css": "css",
  ".html": "xml",
  ".js": "javascript",
  ".json": "json",
  ".md": "markdown",
  ".py": "python",
  ".sh": "bash",
  ".ts": "typescript",
  ".tsx": "tsx",
  ".xml": "xml",
  ".yml": "yaml",
  ".yaml": "yaml",
};

export function renderFilePreview(content, path = "") {
  const extension = path.includes(".") ? path.slice(path.lastIndexOf(".")) : "";
  if (extension === ".md") {
    return { html: renderMarkdown(content || "No preview content loaded."), className: "markdown-body preview-body" };
  }

  const language = LANGUAGE_BY_EXTENSION[extension];
  const safeContent = String(content || "");
  const highlighted = language
    ? hljs.highlight(safeContent, { language, ignoreIllegals: true }).value
    : hljs.highlightAuto(safeContent).value;

  return {
    html: `<pre class="code-frame"><code class="hljs ${language || "plaintext"}">${highlighted}</code></pre>`,
    className: "preview-body code-preview",
  };
}
