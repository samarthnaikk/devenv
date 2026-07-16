from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from .base import BaseTool, ToolResult


class GeneratePDFTool(BaseTool):
    name = "generate_pdf"
    description = "Create a polished PDF document from structured content using LaTeX."

    def input_schema(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Document title."},
                "subtitle": {"type": "string", "description": "Optional subtitle."},
                "author": {"type": "string", "description": "Optional author name."},
                "sections": {
                    "type": "array",
                    "description": "Ordered content sections for the PDF.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "heading": {"type": "string"},
                            "body": {"type": "string"},
                            "bullets": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["heading"],
                    },
                },
                "output_path": {
                    "type": "string",
                    "description": "Optional workspace-relative output path. Defaults to output/pdf/<slug>.pdf",
                },
                "keep_tex": {
                    "type": "boolean",
                    "description": "Whether to keep the generated .tex source alongside the PDF.",
                },
            },
            "required": ["title", "sections"],
        }

    def execute(self, **kwargs) -> ToolResult:
        title = str(kwargs.get("title") or "").strip()
        if not title:
            return ToolResult(success=False, output="Missing required argument: title", data={"status": "invalid_input"})

        sections = kwargs.get("sections")
        if not isinstance(sections, list) or not sections:
            return ToolResult(success=False, output="sections must be a non-empty array", data={"status": "invalid_input"})

        subtitle = str(kwargs.get("subtitle") or "").strip()
        author = str(kwargs.get("author") or "").strip()
        keep_tex = bool(kwargs.get("keep_tex"))
        output_path = str(kwargs.get("output_path") or "").strip()

        latex_engine = shutil.which("pdflatex") or shutil.which("xelatex") or shutil.which("lualatex")
        if not latex_engine:
            return ToolResult(success=False, output="No LaTeX engine is available. Install pdflatex, xelatex, or lualatex.", data={"status": "missing_toolchain"})

        safe_sections = _normalize_sections(sections)
        if not safe_sections:
            return ToolResult(success=False, output="sections did not contain any valid content", data={"status": "invalid_input"})

        file_stem = _slugify(Path(output_path).stem or title)
        relative_pdf_path = output_path or f"output/pdf/{file_stem}.pdf"
        pdf_target = Path(relative_pdf_path)
        tex_target = pdf_target.with_suffix(".tex")
        pdf_target.parent.mkdir(parents=True, exist_ok=True)

        latex_source = _build_latex_document(title=title, subtitle=subtitle, author=author, sections=safe_sections)

        with tempfile.TemporaryDirectory(prefix="devenv-pdf-") as tmpdir:
            temp_root = Path(tmpdir)
            tex_file = temp_root / f"{file_stem}.tex"
            tex_file.write_text(latex_source, encoding="utf-8")
            try:
                completed = subprocess.run(
                    [latex_engine, "-interaction=nonstopmode", "-halt-on-error", tex_file.name],
                    cwd=temp_root,
                    capture_output=True,
                    text=True,
                    timeout=40,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                return ToolResult(success=False, output=f"LaTeX compilation failed: {exc}", data={"status": "compile_failed"})

            compiled_pdf = temp_root / f"{file_stem}.pdf"
            if completed.returncode != 0 or not compiled_pdf.is_file():
                error_text = (completed.stdout or "") + "\n" + (completed.stderr or "")
                return ToolResult(
                    success=False,
                    output="LaTeX compilation failed.",
                    data={"status": "compile_failed", "log_excerpt": error_text[-3000:]},
                )

            pdf_target.write_bytes(compiled_pdf.read_bytes())
            if keep_tex:
                tex_target.write_text(latex_source, encoding="utf-8")
            elif tex_target.exists():
                tex_target.unlink()

        output = f"Generated PDF at {pdf_target}"
        data = {
            "status": "ok",
            "path": str(pdf_target),
            "tex_path": str(tex_target) if keep_tex else "",
            "title": title,
            "page_preview_ready": True,
        }
        return ToolResult(success=True, output=output, data=data)


def _normalize_sections(sections: list[object]) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for section in sections:
        if not isinstance(section, dict):
            continue
        heading = str(section.get("heading") or "").strip()
        body = str(section.get("body") or "").strip()
        bullets_raw = section.get("bullets")
        bullets = [str(item).strip() for item in bullets_raw if str(item).strip()] if isinstance(bullets_raw, list) else []
        if not heading:
            continue
        normalized.append({"heading": heading, "body": body, "bullets": bullets})
    return normalized


def _build_latex_document(*, title: str, subtitle: str, author: str, sections: list[dict[str, object]]) -> str:
    blocks = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage{lmodern}",
        r"\usepackage{xcolor}",
        r"\usepackage{hyperref}",
        r"\usepackage{enumitem}",
        r"\usepackage{titlesec}",
        r"\usepackage{parskip}",
        r"\definecolor{accent}{HTML}{0F766E}",
        r"\definecolor{textgray}{HTML}{46515A}",
        r"\hypersetup{colorlinks=true,linkcolor=accent,urlcolor=accent}",
        r"\titleformat{\section}{\Large\bfseries\color{accent}}{\thesection}{0.75em}{}",
        r"\setlist[itemize]{leftmargin=1.6em,itemsep=0.35em,topsep=0.35em}",
        r"\begin{document}",
        r"\pagestyle{plain}",
        rf"{{\huge\bfseries {_escape_latex(title)}}}\par",
    ]
    if subtitle:
        blocks.append(r"\vspace{0.35em}")
        blocks.append(rf"{{\large\color{{textgray}} {_escape_latex(subtitle)}}}\par")
    if author:
        blocks.append(r"\vspace{0.65em}")
        blocks.append(rf"{{\normalsize {_escape_latex(author)}}}\par")
    blocks.append(r"\vspace{1.2em}")

    for section in sections:
        blocks.append(rf"\section*{{{_escape_latex(str(section['heading']))}}}")
        body = str(section.get("body") or "").strip()
        if body:
            blocks.append(_render_paragraphs(body))
        bullets = section.get("bullets") or []
        if bullets:
            blocks.append(r"\begin{itemize}")
            for bullet in bullets:
                blocks.append(rf"\item {_escape_latex(str(bullet))}")
            blocks.append(r"\end{itemize}")
        blocks.append("")

    blocks.append(r"\end{document}")
    return "\n".join(blocks) + "\n"


def _render_paragraphs(body: str) -> str:
    chunks = [chunk.strip() for chunk in re.split(r"\n\s*\n", body) if chunk.strip()]
    return "\n\n".join(_escape_latex(chunk).replace("\n", r"\\ ") for chunk in chunks)


def _escape_latex(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    escaped = "".join(replacements.get(char, char) for char in text.replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-"))
    return escaped


def _slugify(value: str) -> str:
    lowered = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return lowered or "document"
