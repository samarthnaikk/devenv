export function renderMarkdown(markdown) {
  const escaped = escapeHtml(markdown || "");
  const withCodeBlocks = escaped.replace(/```([\s\S]*?)```/g, (_, code) => `<pre><code>${code.trim()}</code></pre>`);
  const withInlineCode = withCodeBlocks.replace(/`([^`]+)`/g, "<code>$1</code>");
  const withLinks = withInlineCode.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>');
  const withStrong = withLinks.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  const withEmphasis = withStrong.replace(/(^|[^\*])\*([^*]+)\*/g, "$1<em>$2</em>");
  const withHeadings = withEmphasis
    .replace(/^### (.*)$/gm, "<h3>$1</h3>")
    .replace(/^## (.*)$/gm, "<h2>$1</h2>")
    .replace(/^# (.*)$/gm, "<h1>$1</h1>");
  const withBlockquotes = withHeadings.replace(/(?:^> .*(?:\n|$))+?/gm, (block) => {
    const lines = block
      .trim()
      .split("\n")
      .map((line) => line.replace(/^>\s?/, ""))
      .join("<br />");
    return `<blockquote>${lines}</blockquote>`;
  });
  const withOrderedLists = withBlockquotes.replace(/(?:^\d+\. .*(?:\n|$))+?/gm, (block) => {
    const items = block
      .trim()
      .split("\n")
      .map((line) => `<li>${line.replace(/^\d+\. /, "")}</li>`)
      .join("");
    return `<ol>${items}</ol>`;
  });
  const withLists = withOrderedLists.replace(/(?:^- .*(?:\n|$))+?/gm, (block) => {
    const items = block
      .trim()
      .split("\n")
      .map((line) => `<li>${line.replace(/^- /, "")}</li>`)
      .join("");
    return `<ul>${items}</ul>`;
  });
  return withLists
    .split(/\n{2,}/)
    .map((chunk) => {
      if (chunk.startsWith("<")) {
        return chunk;
      }
      return `<p>${chunk.replace(/\n/g, "<br />")}</p>`;
    })
    .join("");
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}
