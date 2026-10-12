"""Render the reviewed Markdown sources with the isolated reporting environment."""

import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import markdown
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def render(source: Path, destination: Path, qa: Path) -> dict:
    import fitz

    fonts = Path("C:/Windows/Fonts")
    for name, filename in [
        ("Research", "times.ttf"),
        ("Research-Bold", "timesbd.ttf"),
        ("Research-Italic", "timesi.ttf"),
    ]:
        pdfmetrics.registerFont(TTFont(name, str(fonts / filename)))
    pdfmetrics.registerFontFamily(
        "Research",
        normal="Research",
        bold="Research-Bold",
        italic="Research-Italic",
        boldItalic="Research-Bold",
    )
    styles = getSampleStyleSheet()
    for name in ("Normal", "BodyText"):
        styles[name].fontName = "Research"
        styles[name].fontSize = 11
        styles[name].leading = 14.5
        styles[name].spaceAfter = 7
    for level in range(1, 5):
        style = styles[f"Heading{level}"]
        style.fontName = "Research-Bold"
        style.fontSize = {1: 20, 2: 14, 3: 12, 4: 11}[level]
        style.leading = style.fontSize * 1.2
        style.textColor = colors.black
        style.spaceBefore = 12 if level > 1 else 0
        style.spaceAfter = 8
        style.keepWithNext = True
    small = ParagraphStyle("Cell", parent=styles["Normal"], fontSize=9, leading=11, spaceAfter=0)
    cell_head = ParagraphStyle(
        "CellHead", parent=small, fontName="Research-Bold", textColor=colors.white
    )
    caption = ParagraphStyle("Caption", parent=styles["Normal"], fontSize=9.5, leading=12)
    html = markdown.markdown(
        source.read_text(encoding="utf-8"), extensions=["tables", "fenced_code"]
    )
    tree = ET.fromstring("<document>" + html + "</document>")
    width = letter[0] - 96
    story = []

    def inline(node):
        value = ET.tostring(node, encoding="unicode", method="xml")
        value = re.sub(r"^<[^>]+>|</[^>]+>$", "", value)
        value = value.replace("<strong>", "<b>").replace("</strong>", "</b>")
        value = value.replace("<em>", "<i>").replace("</em>", "</i>")
        value = value.replace("<code>", '<font name="Courier" size="9">').replace(
            "</code>", "</font>"
        )
        value = re.sub(
            r'<a href="(https?://[^"]+)"[^>]*>', r'<link href="\1" color="#163c63">', value
        )
        value = re.sub(r'<a href="[^"]+"[^>]*>', "<link>", value)
        value = value.replace("</a>", "</link>")
        value = value.replace("—", "-").replace("–", "-").replace("−", "-")
        return value

    nodes = list(tree)
    skip = False
    for index, node in enumerate(nodes):
        if skip:
            skip = False
            continue
        if node.tag in {"h1", "h2", "h3", "h4"}:
            story.append(Paragraph(inline(node), styles["Heading" + node.tag[1:]]))
        elif node.tag == "p":
            img = node.find("img")
            if img is not None:
                image_path = (source.parent / img.attrib["src"]).resolve()
                if not image_path.is_relative_to(source.resolve().parents[2]):
                    raise ValueError("Figure path must remain inside the project")
                visual = Image(str(image_path))
                scale = min(width / visual.imageWidth, 340 / visual.imageHeight)
                visual.drawWidth, visual.drawHeight = (
                    visual.imageWidth * scale,
                    visual.imageHeight * scale,
                )
                block = [Spacer(1, 8), visual, Spacer(1, 6)]
                if index + 1 < len(nodes) and "".join(nodes[index + 1].itertext()).startswith(
                    "Figure "
                ):
                    block.append(Paragraph(inline(nodes[index + 1]), caption))
                    skip = True
                story.append(KeepTogether(block))
            else:
                story.append(Paragraph(inline(node), styles["Normal"]))
        elif node.tag == "table":
            rows = node.findall(".//tr")
            cells = [
                [Paragraph(inline(cell), cell_head if row_index == 0 else small) for cell in row]
                for row_index, row in enumerate(rows)
            ]
            heads = ["".join(cell.itertext()) for cell in rows[0]]
            if "Accuracy mean ± SD" in heads:
                weights = [56, 65, 148, 148, 62]
            elif "95% interval pp" in heads:
                weights = [59, 59, 64, 67, 82, 148]
            elif "F1 delta pp" in heads:
                weights = [76, 77, 77, 77, 85, 87]
            else:
                weights = [max(7, min(18, len(h))) for h in heads]
            widths = [width * w / sum(weights) for w in weights]
            table = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#303b46")),
                        (
                            "ROWBACKGROUNDS",
                            (0, 1),
                            (-1, -1),
                            [colors.white, colors.HexColor("#f3f5f7")],
                        ),
                        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d9d9d9")),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 6),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                        ("TOPPADDING", (0, 0), (-1, -1), 5),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ]
                )
            )
            story.extend([table, Spacer(1, 10)])
        elif node.tag in {"ul", "ol"}:
            for counter, item in enumerate(node, 1):
                prefix = f"{counter}. " if node.tag == "ol" else "• "
                story.append(Paragraph(prefix + inline(item), styles["Normal"]))
        elif node.tag == "pre":
            for line in "".join(node.itertext()).splitlines():
                from xml.sax.saxutils import escape

                story.append(Paragraph(escape(line), small))
            story.append(Spacer(1, 7))
        else:
            raise ValueError(f"Unsupported report block: {node.tag}")

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Research", 9)
        canvas.setFillColor(colors.HexColor("#444444"))
        canvas.drawString(48, 26, "Domain Shift Workbench | Research evidence")
        canvas.drawRightString(letter[0] - 48, 26, str(doc.page))
        canvas.restoreState()

    destination.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(destination),
        pagesize=letter,
        leftMargin=48,
        rightMargin=48,
        topMargin=44,
        bottomMargin=48,
        title="Evaluating Domain Adaptation Techniques for Classical Machine Learning Models",
        author="",
        subject="Research report and evidence",
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    qa.mkdir(parents=True, exist_ok=True)
    pdf = fitz.open(destination)
    issues, images = [], []
    for number, page in enumerate(pdf, 1):
        image_path = qa / f"page-{number:02d}.png"
        page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).save(image_path)
        images.append(
            {"page": number, "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest()}
        )
        for block in page.get_text("dict")["blocks"]:
            if block["type"] == 0:
                x0, y0, x1, y1 = block["bbox"]
                if x0 < 42 or x1 > letter[0] - 42 or y0 < 35 or y1 > letter[1] - 16:
                    issues.append({"page": number, "bbox": [x0, y0, x1, y1]})
    receipt = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "pdf_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "page_count": len(pdf),
        "page_images": images,
        "text_bounds_issues": issues,
        "renderer": "ReportLab PDF; PyMuPDF page rasterization; isolated reporting environment",
        "visual_review_pending": True,
    }
    (qa / "render_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(f"Rendered {destination.name}: {len(pdf)} pages; {len(issues)} text boundary issues")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--qa", type=Path, required=True)
    args = parser.parse_args()
    render(args.source, args.destination, args.qa)
