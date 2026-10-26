"""Turn the PDF into one chunk per content element: prose passage, stat call-out, whole table, chart, page description, endnote.

Footnote markers are resolved in both directions: an element lists the endnotes it cites, and an endnote lists the claims that cite it.
"""
import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from .extract import Block, MARKER, marker_numbers, page_blocks, page_spans, strip_markers
from .models import Chunk
from .notes import is_endnote_page, read_endnotes
from .tables import Table, find_tables, verbalise
from .visuals import describe_pages, visual_pages

STAT = re.compile(r"[–−-]?\d[\d.,]*\s?(?:%|kg|W|g)?(?:\[\^\d+\])?")
SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9“\"(])")
MAX_WORDS = 160  # a prose block longer than this is split on sentence boundaries
MIN_WORDS = 15  # shorter blocks are merged with their neighbours in the same section


@dataclass
class Raw:
    """An element before footnote resolution; `text` still carries [^n] markers."""

    page: int
    kind: str
    text: str
    heading: str
    extra_notes: tuple[int, ...] = ()


def words(text: str) -> int:
    """Whitespace word count."""
    return len(text.split())


def chart_box(page: pymupdf.Page) -> pymupdf.Rect | None:
    """Box around a chart: the union of its coloured bars, grown to take in the labels around them."""
    bars = []
    for d in page.get_drawings():
        f = d.get("fill")
        r = d["rect"]
        if f and max(f[:3]) - min(f[:3]) > 0.15 and r.width * r.height > 1500:
            bars.append(r)
    if len(bars) < 3:
        return None
    box = pymupdf.Rect(bars[0])
    for r in bars[1:]:
        box |= r
    return pymupdf.Rect(box.x0 - 40, box.y0 - 50, box.x1 + 40, box.y1 + 60)


def _inside(b: Block, box: pymupdf.Rect | tuple) -> bool:
    cx, cy = (b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2
    x0, y0, x1, y1 = tuple(box)
    return x0 <= cx <= x1 and y0 <= cy <= y1


def is_heading(b: Block) -> bool:
    """Headings are bold and short, or simply large."""
    if STAT.fullmatch(b.text):
        return False
    return (b.bold and b.size >= 8.5 and b.lines <= 3 and len(b.text) <= 90 and not b.text.endswith(".")) or b.size >= 14


def chart_text(blocks: list[Block]) -> str:
    """Chart labels as text: each big figure with the label under it, then the remaining segment labels."""
    stats = [b for b in blocks if STAT.fullmatch(b.text) and b.size >= 10]
    used: set[int] = set()
    pairs = []
    for s in sorted(stats, key=lambda b: b.x0):
        below = [
            (i, b) for i, b in enumerate(blocks)
            if b is not s and not STAT.fullmatch(b.text) and b.y0 >= s.y1 - 2 and b.y0 - s.y1 < 40
            and abs((b.x0 + b.x1) / 2 - (s.x0 + s.x1) / 2) < 45
        ]
        if below:
            i, label = min(below, key=lambda t: t[1].y0)
            used.add(i)
            pairs.append(f"{s.text} — {label.text}")
    rest: list[str] = []
    for i, b in sorted(enumerate(blocks), key=lambda t: (t[1].x0, t[1].y0)):
        if i not in used and b.text not in stats and b.text not in rest and not STAT.fullmatch(b.text):
            rest.append(b.text)
    return "Chart labels: " + "; ".join(pairs) + ". Other labels in the chart: " + "; ".join(rest) + "."


def merge_headings(heads: list[Block]) -> list[Block]:
    """Join heading blocks that are really one wrapped heading ("Progress toward" / "our 2030 goal")."""
    out: list[Block] = []
    for h in sorted(heads, key=lambda b: (b.x0, b.y0)):
        prev = out[-1] if out else None
        if prev and abs(prev.x0 - h.x0) <= 30 and 0 <= h.y0 - prev.y1 <= 8 and abs(prev.size - h.size) < 0.6:
            out[-1] = Block(min(prev.x0, h.x0), prev.y0, max(prev.x1, h.x1), h.y1, f"{prev.text} {h.text}",
                            prev.size, True, prev.lines + h.lines, prev.notes + h.notes)
        else:
            out.append(h)
    return out


def clusters(bodies: list[Block]) -> list[list[Block]]:
    """Split a section's blocks into columns (centres more than 60pt apart), each ordered top to bottom."""
    cols: list[list[Block]] = []
    for b in sorted(bodies, key=lambda b: (b.x0 + b.x1) / 2):
        cx = (b.x0 + b.x1) / 2
        if cols and cx - sum((c.x0 + c.x1) / 2 for c in cols[-1]) / len(cols[-1]) <= 60:
            cols[-1].append(b)
        else:
            cols.append([b])
    return [sorted(c, key=lambda b: b.y0) for c in cols]


def prose_elements(page_no: int, blocks: list[Block]) -> list[Raw]:
    """Group a page's blocks under their headings and cut them into passages and stat call-outs."""
    blocks = [b for b in blocks if not b.text.isdigit() and not ("Product Environmental Report" in b.text[:40] and b.y0 > 720)]
    titles = [b for b in blocks if b.size >= 20 and is_heading(b)]
    title = strip_markers(titles[0].text) if titles else f"Page {page_no}"
    heads = merge_headings([b for b in blocks if is_heading(b) and b not in titles])
    taken: set[int] = set()
    elements: list[Raw] = []

    for s in [b for b in blocks if STAT.fullmatch(b.text) and b.size >= 10]:
        desc = [
            b for b in blocks
            if b is not s and id(b) not in taken and not STAT.fullmatch(b.text)
            and 0 <= b.y0 - s.y1 < 60 and abs(b.x0 - s.x0) < 25 and not is_heading(b)
        ]
        if desc:
            d = min(desc, key=lambda b: b.y0)
            taken.update({id(s), id(d)})
            elements.append(Raw(page_no, "stat", f"{s.text} {d.text}", title))

    sections: dict[int, list[Block]] = {id(h): [] for h in heads}
    sections[0] = []
    for b in blocks:
        if id(b) in taken or b in titles or any(b.text in h.text and b.y0 >= h.y0 - 2 and b.y1 <= h.y1 + 2 for h in heads):
            continue
        above = [h for h in heads if h.y0 <= b.y0 + 2 and abs(h.x0 - b.x0) <= 90]
        sections[id(max(above, key=lambda h: h.y0)) if above else 0].append(b)
    by_id = {id(h): h for h in heads}

    for key, bodies in sections.items():
        head = by_id.get(key)
        heading = strip_markers(head.text) if head else ""
        path = f"{title} > {heading}" if heading else title
        carried = tuple(head.notes) if head else ()
        if not bodies:
            if head and words(heading) >= 4:
                elements.append(Raw(page_no, "text", head.text, title))
            continue
        for column in clusters(bodies):
            pending, last = "", None
            for b in column:
                continues = last is not None and 0 <= b.y0 - last.y1 <= 4 and abs(b.x0 - last.x0) <= 12
                far = last is not None and b.y0 - last.y1 > 30
                if pending and not continues and (words(pending) >= MIN_WORDS or far):
                    elements += [Raw(page_no, "text", piece, path, carried) for piece in _split(pending)]
                    pending = ""
                pending = f"{pending} {b.text}".strip()
                last = b
            if pending:
                elements += [Raw(page_no, "text", piece, path, carried) for piece in _split(pending)]
    return elements


def _split(text: str) -> list[str]:
    """Cut long text on sentence boundaries into pieces of at most MAX_WORDS words."""
    if words(text) <= MAX_WORDS:
        return [text]
    out, cur = [], ""
    for sent in SENT.split(text):
        if cur and words(cur) + words(sent) > MAX_WORDS:
            out.append(cur)
            cur = ""
        cur = f"{cur} {sent}".strip()
    return out + ([cur] if cur else [])


def table_elements(page: pymupdf.Page, page_no: int, tables: list[Table], title: str) -> list[Raw]:
    """One element per table: every cell as a sentence with its row and column heading."""
    out = []
    for t in tables:
        text, _ = verbalise(t)
        extra: tuple[int, ...] = ()
        if is_endnote_page(page):
            above = [
                int(s.text.strip()) for s in page_spans(page)
                if s.size < 5.5 and s.text.strip().isdigit() and s.y0 < t.bbox[1]
            ]
            extra = (above[-1],) if above else ()
        out.append(Raw(page_no, "table", text, f"{title} > table", extra))
    return out


def build_chunks(doc: pymupdf.Document, pdf_path: Path, llm, vision_cache: Path, use_vision: bool = True) -> tuple[list[Chunk], dict]:
    """Extract every element of the report and resolve footnotes in both directions.

    @param doc Opened PDF.
    @param pdf_path PDF path (keys the vision cache).
    @param llm LLM wrapper used for page descriptions.
    @param vision_cache JSON cache of page descriptions.
    @param use_vision Describe visual pages with the vision model when True.
    @return (chunks, report with counts, endnotes and pages that could not be described).
    """
    tables = {pn: find_tables(doc[pn - 1], pn) for pn in range(1, doc.page_count + 1)}
    notes = read_endnotes(doc, {pn: [t.bbox for t in ts] for pn, ts in tables.items() if ts})
    raws: list[Raw] = []
    failed: list[int] = []
    described: dict[int, str] = {}
    if use_vision:
        described, failed = describe_pages(doc, pdf_path, llm, vision_cache)

    for pn in range(1, doc.page_count + 1):
        page = doc[pn - 1]
        blocks = page_blocks(page)
        titles = [b.text for b in blocks if b.size >= 20 and is_heading(b)]
        title = titles[0] if titles else f"Page {pn}"
        raws += table_elements(page, pn, tables[pn], title)
        if is_endnote_page(page):
            continue
        boxes = [t.bbox for t in tables[pn]]
        blocks = [b for b in blocks if not any(_inside(b, bx) for bx in boxes)]
        cb = chart_box(page)
        if cb:
            chart_blocks = [b for b in blocks if _inside(b, cb)]
            blocks = [b for b in blocks if not _inside(b, cb)]
            raws.append(Raw(pn, "chart", chart_text(chart_blocks), f"{title} > chart"))
        raws += prose_elements(pn, blocks)
        if pn in described:
            raws.append(Raw(pn, "vision", described[pn], f"{title} > full-page visual description"))

    chunks: list[Chunk] = []
    claims: dict[int, list[str]] = {n: [] for n in notes}
    for r in raws:
        cited = [n for n in dict.fromkeys([*marker_numbers(r.text), *r.extra_notes]) if n in notes]
        for n in cited:
            for sent in SENT.split(r.text.replace("\n", " ")):
                if f"[^{n}]" in sent:
                    claims[n].append(f"p.{r.page}: {strip_markers(sent)[:240]}")
                    break
        text = strip_markers(r.text)
        extra = "".join(f"\nNote {n}: {notes[n][:500]}" for n in cited)
        chunks.append(Chunk(f"e{len(chunks):03d}", text, r.page, r.kind, r.heading, cited, index_text=f"{r.heading}\n{text}{extra}"))
    pages_of_notes = [pn for pn in range(1, doc.page_count + 1) if is_endnote_page(doc[pn - 1])]
    for n, body in notes.items():
        claim_text = " ".join(claims[n])
        chunks.append(Chunk(
            f"n{n:02d}", f"Endnote {n}: {body}", pages_of_notes[0] if n <= 16 else pages_of_notes[-1], "footnote",
            f"Endnotes > Note {n}", note_no=n, claims=claims[n],
            index_text=f"Endnotes > Note {n}\nEndnote {n}: {body}\nCited by: {claim_text}",
        ))
    report = {"notes": len(notes), "vision_failed": failed, "visual_pages": visual_pages(doc), "tables": sum(map(len, tables.values()))}
    return chunks, report
