"""Endnotes: parse the numbered notes so that [^n] markers elsewhere can be linked to their text."""
import re

import pymupdf

from .extract import clean

NOTE_NUMBER_SIZE = 5.5  # note numbers are tiny; the note text is 7pt
BULLET = re.compile(r"^•\s*")


def is_endnote_page(page: pymupdf.Page) -> bool:
    """True for a page headed "Endnotes" in large type."""
    for b in page.get_text("dict")["blocks"]:
        for line in b.get("lines", []):
            for s in line["spans"]:
                if s["size"] >= 20 and s["text"].strip() == "Endnotes":
                    return True
    return False


def read_endnotes(doc: pymupdf.Document, skip_boxes: dict[int, list[tuple[float, float, float, float]]]) -> dict[int, str]:
    """Collect endnotes in order across the endnote pages.

    @param doc Opened PDF.
    @param skip_boxes Per page (1-based), boxes of tables that sit between notes and must not leak into note text.
    @return Mapping note number -> note text (bullet lines kept as "- ..." lines).
    """
    notes: dict[int, str] = {}
    current = 0
    for pn in range(1, doc.page_count + 1):
        page = doc[pn - 1]
        if not is_endnote_page(page):
            continue
        boxes = skip_boxes.get(pn, [])
        lines = []
        for b in page.get_text("dict")["blocks"]:
            for line in b.get("lines", []):
                x0, y0, x1, y1 = line["bbox"]
                if any(bx0 - 3 <= x0 and y0 >= by0 - 3 and y1 <= by1 + 3 for bx0, by0, bx1, by1 in boxes):
                    continue
                lines.append((y0, line))
        for y0, line in sorted(lines, key=lambda t: t[0]):
            spans = [s for s in line["spans"] if s["text"].strip()]
            if not spans or y0 < 60 or y0 > 715 or spans[0]["size"] >= 20 or max(s["size"] for s in spans) >= 11.5:
                continue
            first = spans[0]
            if first["size"] < NOTE_NUMBER_SIZE and first["text"].strip().isdigit():
                current = int(first["text"].strip())
                notes[current] = clean(" ".join(s["text"] for s in spans[1:]))
            elif current:
                text = clean(" ".join(s["text"] for s in spans))
                if BULLET.match(text):
                    notes[current] += "\n- " + BULLET.sub("", text)
                else:
                    notes[current] += (" " if not notes[current].endswith("\n") else "") + text
    return {n: re.sub(r"\n- ", "\n- ", t.strip()) for n, t in notes.items()}
