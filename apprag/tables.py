"""Table reconstruction for designed pages.

PyMuPDF's table finder returns one-row fragments here, but every table sits under a dark header bar and ends on a darker
rule, both visible in the page's vector drawings. The bar and the rule give the table's box; word positions give the cells.
The thresholds were tuned on this report.
"""
from dataclasses import dataclass

import pymupdf

from .extract import Span, marker_numbers, page_spans, strip_markers

DARK = 0.35  # a fill darker than this (grey level) is a header bar
CELL_GAP = 9.0  # horizontal gap that separates two cells of one row
ROW_TOL = 3.5  # vertical tolerance for spans on one row
COLUMN_GAP = 25.0  # centres closer than this belong to one value column


@dataclass
class Table:
    """A rebuilt table: header cells per value column, then rows of cells."""

    page: int
    bbox: tuple[float, float, float, float]
    title: str  # spanning header text, if any
    columns: list[str]  # value-column headers
    label_header: str
    rows: list[tuple[str, list[str]]]  # (row label with markers, value per column)


def _gray(fill) -> float | None:
    """Grey level of an RGB fill when it is grey, else None."""
    if not fill:
        return None
    r, g, b = fill[:3]
    return (r + g + b) / 3 if max(r, g, b) - min(r, g, b) < 0.05 else None


def header_bars(page: pymupdf.Page) -> list[pymupdf.Rect]:
    """Dark rectangles of header height, merged when they sit side by side."""
    rects = []
    for d in page.get_drawings():
        g = _gray(d.get("fill"))
        r = d["rect"]
        if g is not None and g < DARK and 12 <= r.height <= 50 and r.width > 60:
            rects.append(r)
    bars: list[pymupdf.Rect] = []
    for r in sorted(rects, key=lambda r: (r.y0, r.x0)):
        grown = pymupdf.Rect(r.x0 - 1.5, r.y0 - 1.5, r.x1 + 1.5, r.y1 + 1.5)
        touching = [i for i, bar in enumerate(bars) if grown.intersects(bar)]
        merged = pymupdf.Rect(r)
        for i in touching:
            merged |= bars[i]
        bars = [bar for i, bar in enumerate(bars) if i not in touching] + [merged]
    return bars


def bottom_rule(page: pymupdf.Page, bar: pymupdf.Rect, limit: float) -> float | None:
    """y of the first dark thin horizontal rule under the bar that spans most of its width."""
    ys = []
    for d in page.get_drawings():
        r = d["rect"]
        color = d.get("color")
        if (
            r.height < 1.5 and r.width >= 0.8 * bar.width and bar.y1 < r.y0 < limit
            and color and max(color[:3]) < 0.6 and (d.get("width") or 0) >= 0.45
        ):
            ys.append(r.y0)
    return min(ys) if ys else None


def _rows(spans: list[Span]) -> list[list[list[Span]]]:
    """Group spans into rows by y, then into cells by horizontal gap; each row is a list of cells (span lists)."""
    rows: list[list[Span]] = []
    for s in sorted(spans, key=lambda s: ((s.y0 + s.y1) / 2, s.x0)):
        cy = (s.y0 + s.y1) / 2
        if rows and abs(cy - sum((t.y0 + t.y1) / 2 for t in rows[-1]) / len(rows[-1])) <= ROW_TOL:
            rows[-1].append(s)
        else:
            rows.append([s])
    out = []
    for row in rows:
        cells: list[list[Span]] = []
        for s in sorted(row, key=lambda s: s.x0):
            if cells and (s.sup or s.x0 - cells[-1][-1].x1 <= CELL_GAP):
                cells[-1].append(s)
            else:
                cells.append([s])
        out.append(cells)
    return out


def _text(cell: list[Span]) -> str:
    """Cell text with footnote markers as [^n] and single spaces."""
    out, prev = "", None
    for s in cell:
        piece = f"[^{s.text.strip()}]" if s.sup else s.text.strip()
        glued = s.sup or prev is None or s.x0 - prev.x1 <= 1.2  # subscripts and markers touch the preceding text
        out += piece if glued else " " + piece
        prev = s
    return out


def _bounds(cell: list[Span]) -> tuple[float, float]:
    return min(s.x0 for s in cell), max(s.x1 for s in cell)


def build_table(page: pymupdf.Page, page_no: int, bar: pymupdf.Rect, next_top: float) -> Table | None:
    """Rebuild one table under a header bar.

    @param page The PDF page.
    @param page_no 1-based page number.
    @param bar The dark header bar.
    @param next_top y of the next bar (or page bottom) that the table must not cross.
    @return The table, or None when no data rows are found under the bar.
    """
    spans = page_spans(page)
    bottom = bottom_rule(page, bar, next_top) or next_top
    head = [s for s in spans if bar.y0 - 1 <= s.y0 and s.y1 <= bar.y1 + 1 and bar.x0 - 6 <= s.x0 <= bar.x1 + 6]
    body = [s for s in spans if bar.y1 - 0.5 <= s.y0 and s.y1 <= bottom + 1 and bar.x0 - 6 <= s.x0 <= bar.x1 + 6]
    rows = _rows(body)
    if not rows:
        return None

    left_edge = bar.x0 + 25
    value_cells = [(_bounds(c), c) for row in rows for c in row if _bounds(c)[0] > left_edge]
    centres = sorted((b[0] + b[1]) / 2 for b, _ in value_cells)
    anchors: list[float] = []
    for c in centres:
        if anchors and c - anchors[-1] <= COLUMN_GAP:
            anchors[-1] = (anchors[-1] + c) / 2
        else:
            anchors.append(c)
    if not anchors:
        return None

    def column_of(cell: list[Span]) -> int:
        lo, hi = _bounds(cell)
        return min(range(len(anchors)), key=lambda i: abs(anchors[i] - (lo + hi) / 2))

    # Header: cells spanning several value columns are the table title, the rest label a column.
    label_header, title = "", ""
    col_head: dict[int, list[str]] = {i: [] for i in range(len(anchors))}
    head_rows = _rows(head)
    has_column_row = any(sum(_bounds(c)[0] > left_edge for c in row) >= 2 for row in head_rows)
    for row in head_rows:
        value_cells = sum(_bounds(c)[0] > left_edge for c in row)
        for cell in row:
            lo, hi = _bounds(cell)
            text = strip_markers(_text(cell))
            inside = [i for i, a in enumerate(anchors) if lo - 2 <= a <= hi + 2]
            if lo <= left_edge:
                label_header = f"{label_header} {text}".strip()
            elif len(inside) >= 2 or (has_column_row and value_cells == 1):
                title = f"{title} {text}".strip()
            else:
                col_head[column_of(cell)].append(text)
    columns = [" ".join(col_head[i]) or f"column {i + 1}" for i in range(len(anchors))]

    out_rows: list[tuple[str, list[str]]] = []
    for row in rows:
        label_cells = [c for c in row if _bounds(c)[0] <= left_edge]
        vals = [""] * len(anchors)
        for c in row:
            if _bounds(c)[0] > left_edge:
                vals[column_of(c)] = _text(c)
        if label_cells or any(vals):
            out_rows.append((" ".join(_text(c) for c in label_cells), vals))
    return Table(page_no, (bar.x0, bar.y0, bar.x1, bottom), title, columns, label_header, out_rows)


def find_tables(page: pymupdf.Page, page_no: int) -> list[Table]:
    """All tables on a page, one per header bar."""
    bars = sorted(header_bars(page), key=lambda r: r.y0)
    out = []
    for i, bar in enumerate(bars):
        limit = bars[i + 1].y0 - 2 if i + 1 < len(bars) else page.rect.height - 40
        t = build_table(page, page_no, bar, limit)
        if t:
            out.append(t)
    return out


def verbalise(t: Table) -> tuple[str, list[int]]:
    """Turn a table into sentences that keep every value next to its row and column heading.

    Bulleted rows are children of the row above them ("Life cycle product emissions (scope 3) > Production").

    @param t Rebuilt table.
    @return (text with [^n] markers kept, endnotes cited anywhere in the table).
    """
    head = f"Table: {t.label_header or 'rows'} by {', '.join(t.columns)}" + (f" ({t.title})" if t.title else "") + "."
    lines = [head]
    parent = ""
    for label, vals in t.rows:
        is_child = label.startswith("•")
        name = label.lstrip("• ").strip()
        shown = f"{parent} > {name}" if is_child and parent else name
        if not is_child:
            parent = strip_markers(name)
        cells = [(c, v) for c, v in zip(t.columns, vals) if v]
        if len(cells) == 1:
            c, v = cells[0]
            lines.append(f"{shown} — {c}: {v}.")
        else:
            lines.extend(f"{shown} — {(t.title + ' ') if t.title else ''}{c}: {v}." for c, v in cells)
    text = "\n".join(lines)
    return text, marker_numbers(text)
