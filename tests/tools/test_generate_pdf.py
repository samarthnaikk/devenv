from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.tools.generate_pdf import GeneratePDFTool


class GeneratePDFToolTest(unittest.TestCase):
    def test_execute_requires_title_and_sections(self) -> None:
        tool = GeneratePDFTool()

        missing_title = tool.execute(sections=[{"heading": "Intro"}])
        missing_sections = tool.execute(title="Demo", sections=[])

        self.assertFalse(missing_title.success)
        self.assertFalse(missing_sections.success)

    @patch("shutil.which", side_effect=lambda name: "/usr/bin/pdflatex" if name == "pdflatex" else None)
    @patch("subprocess.run")
    def test_execute_writes_pdf_and_optional_tex(self, mock_run, _mock_which) -> None:
        def fake_run(command, cwd, capture_output, text, timeout, check):  # noqa: ARG001
            pdf_path = Path(cwd) / "demo-pdf.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n")
            return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

        mock_run.side_effect = fake_run

        with tempfile.TemporaryDirectory() as tempdir:
            previous = Path.cwd()
            try:
                os_path = Path(tempdir)
                import os
                os.chdir(os_path)
                result = GeneratePDFTool().execute(
                    title="Demo PDF",
                    keep_tex=True,
                    sections=[{"heading": "Summary", "body": "Professional output."}],
                )
            finally:
                os.chdir(previous)

            self.assertTrue(result.success)
            self.assertTrue((os_path / "output" / "pdf" / "demo-pdf.pdf").is_file())
            self.assertTrue((os_path / "output" / "pdf" / "demo-pdf.tex").is_file())

    @patch("core.tools.generate_pdf.search_image_web", return_value=[{"title": "Chat UI mockup", "url": "https://images.example.com/chat-ui.png"}])
    @patch("shutil.which", side_effect=lambda name: "/usr/bin/pdflatex" if name == "pdflatex" else None)
    @patch("subprocess.run")
    def test_execute_records_optional_image_references(self, mock_run, _mock_which, _mock_images) -> None:
        def fake_run(command, cwd, capture_output, text, timeout, check):  # noqa: ARG001
            pdf_path = Path(cwd) / "demo-pdf.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n")
            return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

        mock_run.side_effect = fake_run

        with tempfile.TemporaryDirectory() as tempdir:
            previous = Path.cwd()
            try:
                import os
                os.chdir(Path(tempdir))
                result = GeneratePDFTool().execute(
                    title="Demo PDF",
                    image_query="chat application ui",
                    sections=[{"heading": "Summary", "body": "Professional output."}],
                )
            finally:
                os.chdir(previous)

        self.assertTrue(result.success)
        self.assertEqual(result.data["image_references"][0]["title"], "Chat UI mockup")


if __name__ == "__main__":
    unittest.main()
