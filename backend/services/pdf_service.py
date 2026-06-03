"""
PDF export service.
Converts Markdown report content to a styled PDF using ReportLab.
"""

from __future__ import annotations

import io
import os
import re
from datetime import datetime
from typing import Optional

from loguru import logger
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from backend.config import get_settings


class PDFService:
    """Converts Markdown text to a styled PDF document."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def generate_pdf(self, markdown_content: str, session_id: str) -> str:
        """
        Generate a PDF from Markdown content and save to disk.

        Args:
            markdown_content: Full Markdown string.
            session_id: Session identifier (used in filename).

        Returns:
            Absolute path to the generated PDF file.
        """
        filename = f"report_{session_id[:8]}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"
        output_path = os.path.join(self.settings.reports_path, filename)
        os.makedirs(self.settings.reports_path, exist_ok=True)

        logger.info(f"Generating PDF: {output_path}")

        try:
            self._build_pdf(markdown_content, output_path)
            logger.info(f"PDF saved: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"PDF generation failed: {e}")
            raise

    def generate_pdf_bytes(self, markdown_content: str) -> bytes:
        """Generate PDF and return as bytes (for API streaming)."""
        buffer = io.BytesIO()
        self._build_pdf_to_buffer(markdown_content, buffer)
        return buffer.getvalue()

    # ------------------------------------------------------------------

    def _build_pdf(self, markdown: str, output_path: str) -> None:
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=2 * cm,
            leftMargin=2 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm,
        )
        story = self._build_story(markdown)
        doc.build(story)

    def _build_pdf_to_buffer(self, markdown: str, buffer: io.BytesIO) -> None:
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=2 * cm,
            leftMargin=2 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm,
        )
        story = self._build_story(markdown)
        doc.build(story)

    def _build_story(self, markdown: str) -> list:
        styles = self._get_styles()
        story = []

        lines = markdown.split("\n")
        i = 0
        while i < len(lines):
            line = lines[i].rstrip()

            if not line:
                story.append(Spacer(1, 0.2 * cm))
                i += 1
                continue

            # H1
            if line.startswith("# ") and not line.startswith("## "):
                text = self._clean(line[2:])
                story.append(Paragraph(text, styles["h1"]))
                story.append(Spacer(1, 0.3 * cm))
                i += 1
                continue

            # H2
            if line.startswith("## "):
                text = self._clean(line[3:])
                story.append(HRFlowable(width="100%", color=colors.HexColor("#1e40af")))
                story.append(Spacer(1, 0.1 * cm))
                story.append(Paragraph(text, styles["h2"]))
                story.append(Spacer(1, 0.2 * cm))
                i += 1
                continue

            # H3
            if line.startswith("### "):
                text = self._clean(line[4:])
                story.append(Paragraph(text, styles["h3"]))
                story.append(Spacer(1, 0.15 * cm))
                i += 1
                continue

            # H4
            if line.startswith("#### "):
                text = self._clean(line[5:])
                story.append(Paragraph(text, styles["h4"]))
                i += 1
                continue

            # Horizontal rule
            if line.startswith("---"):
                story.append(HRFlowable(width="100%", color=colors.HexColor("#e2e8f0")))
                story.append(Spacer(1, 0.2 * cm))
                i += 1
                continue

            # Table (simple | table |)
            if "|" in line and line.strip().startswith("|"):
                table_lines = []
                while i < len(lines) and "|" in lines[i]:
                    table_lines.append(lines[i])
                    i += 1
                table_elem = self._parse_table(table_lines, styles)
                if table_elem:
                    story.append(table_elem)
                    story.append(Spacer(1, 0.3 * cm))
                continue

            # Bullet / list
            if line.startswith("- ") or line.startswith("* "):
                text = self._clean(line[2:])
                story.append(Paragraph(f"• {text}", styles["bullet"]))
                i += 1
                continue

            # Numbered list
            if re.match(r"^\d+\. ", line):
                text = self._clean(re.sub(r"^\d+\. ", "", line))
                story.append(Paragraph(text, styles["numbered"]))
                i += 1
                continue

            # Bold metadata line (Key: value)
            if line.startswith("**") and ":**" in line:
                text = self._clean(line)
                story.append(Paragraph(text, styles["meta"]))
                i += 1
                continue

            # Normal paragraph
            text = self._clean(line)
            if text:
                story.append(Paragraph(text, styles["body"]))
            i += 1

        return story

    # def _clean(self, text: str) -> str:
    #     """Convert Markdown inline elements to ReportLab XML."""
    #     # Bold
    #     text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    #     # Italic
    #     text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    #     # Code
    #     text = re.sub(r"`(.+?)`", r"<font name='Courier'>\1</font>", text)
    #     # Links [text](url) → text (url)
    #     text = re.sub(r"\[([^\]]+)\]\(([^\)]+)\)", r"\1", text)
    #     # Escape XML special chars (except our tags)
    #     text = text.replace("&", "&amp;").replace("<b>", "\x00b\x01").replace("</b>", "\x00/b\x01")
    #     text = text.replace("<i>", "\x00i\x01").replace("</i>", "\x00/i\x01")
    #     text = text.replace("<font", "\x00font").replace("</font>", "\x00/font\x01")
    #     text = text.replace("<", "&lt;").replace(">", "&gt;")
    #     text = text.replace("\x00b\x01", "<b>").replace("\x00/b\x01", "</b>")
    #     text = text.replace("\x00i\x01", "<i>").replace("\x00/i\x01", "</i>")
    #     text = text.replace("\x00font", "<font").replace("\x00/font\x01", "</font>")
    #     return text
    
    def _clean(self, text: str) -> str:
        text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
        text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
        text = re.sub(r"`(.+?)`", r"<font name='Courier'>\1</font>", text)
        text = re.sub(r"\[([^\]]+)\]\(([^\)]+)\)", r"\1", text)

        # Protect tags completely
        text = text.replace("<b>", "__B_OPEN__")
        text = text.replace("</b>", "__B_CLOSE__")

        text = text.replace("<i>", "__I_OPEN__")
        text = text.replace("</i>", "__I_CLOSE__")

        text = text.replace("<font name='Courier'>", "__FONT_OPEN__")
        text = text.replace("</font>", "__FONT_CLOSE__")

        text = text.replace("&", "&amp;")
        text = text.replace("<", "&lt;")
        text = text.replace(">", "&gt;")

        text = text.replace("__B_OPEN__", "<b>")
        text = text.replace("__B_CLOSE__", "</b>")

        text = text.replace("__I_OPEN__", "<i>")
        text = text.replace("__I_CLOSE__", "</i>")

        text = text.replace("__FONT_OPEN__", "<font name='Courier'>")
        text = text.replace("__FONT_CLOSE__", "</font>")

        return text

    def _parse_table(self, lines: list, styles: dict) -> Optional[Table]:
        rows = []
        for line in lines:
            if re.match(r"^\|[-| ]+\|$", line.strip()):
                continue  # separator row
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if cells:
                rows.append(cells)

        if len(rows) < 1:
            return None

        # Build Paragraph cells
        para_rows = []
        for row_idx, row in enumerate(rows):
            style = styles["table_header"] if row_idx == 0 else styles["table_cell"]
            para_rows.append([Paragraph(self._clean(cell), style) for cell in row])

        col_count = max(len(r) for r in para_rows)
        col_width = (A4[0] - 4 * cm) / col_count

        t = Table(para_rows, colWidths=[col_width] * col_count)
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e40af")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")]),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        return t

    def _get_styles(self) -> dict:
        base = getSampleStyleSheet()

        return {
            "h1": ParagraphStyle(
                "H1",
                parent=base["Title"],
                fontSize=22,
                textColor=colors.HexColor("#1e40af"),
                spaceAfter=12,
                fontName="Helvetica-Bold",
                alignment=TA_CENTER,
            ),
            "h2": ParagraphStyle(
                "H2",
                parent=base["Heading1"],
                fontSize=16,
                textColor=colors.HexColor("#1e40af"),
                spaceBefore=12,
                spaceAfter=6,
                fontName="Helvetica-Bold",
            ),
            "h3": ParagraphStyle(
                "H3",
                parent=base["Heading2"],
                fontSize=13,
                textColor=colors.HexColor("#1e3a8a"),
                spaceBefore=8,
                spaceAfter=4,
                fontName="Helvetica-Bold",
            ),
            "h4": ParagraphStyle(
                "H4",
                parent=base["Heading3"],
                fontSize=11,
                textColor=colors.HexColor("#374151"),
                spaceBefore=6,
                spaceAfter=3,
                fontName="Helvetica-Bold",
            ),
            "body": ParagraphStyle(
                "Body",
                parent=base["Normal"],
                fontSize=10,
                textColor=colors.HexColor("#1f2937"),
                spaceAfter=4,
                leading=14,
                alignment=TA_JUSTIFY,
            ),
            "bullet": ParagraphStyle(
                "Bullet",
                parent=base["Normal"],
                fontSize=10,
                textColor=colors.HexColor("#374151"),
                leftIndent=15,
                spaceAfter=2,
                leading=13,
            ),
            "numbered": ParagraphStyle(
                "Numbered",
                parent=base["Normal"],
                fontSize=10,
                textColor=colors.HexColor("#374151"),
                leftIndent=15,
                spaceAfter=2,
                leading=13,
            ),
            "meta": ParagraphStyle(
                "Meta",
                parent=base["Normal"],
                fontSize=9,
                textColor=colors.HexColor("#6b7280"),
                spaceAfter=2,
            ),
            "table_header": ParagraphStyle(
                "TableHeader",
                parent=base["Normal"],
                fontSize=9,
                textColor=colors.white,
                fontName="Helvetica-Bold",
                alignment=TA_CENTER,
            ),
            "table_cell": ParagraphStyle(
                "TableCell",
                parent=base["Normal"],
                fontSize=9,
                textColor=colors.HexColor("#1f2937"),
                alignment=TA_LEFT,
            ),
        }

    def save_markdown(self, content: str, session_id: str) -> str:
        """Save Markdown report to disk and return the path."""
        filename = f"report_{session_id[:8]}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.md"
        output_path = os.path.join(self.settings.reports_path, filename)
        os.makedirs(self.settings.reports_path, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        return output_path
