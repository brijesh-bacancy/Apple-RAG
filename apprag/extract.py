"""Low-level page reading: text blocks and spans with footnote markers turned into [^n] tokens."""
import re
from dataclasses import dataclass, field

import pymupdf

MARKER = re.compile(r"\[\^(\d+)\]")


@dataclass
class Span:
    """A run of text with its box; `sup` is True for superscript digits (footnote markers)."""

    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    size: float
    sup: bool = False


@dataclass
class Block:
    """A paragraph-like text block with the typography needed to tell headings, stats and prose apart."""

    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    size: float
    bold: bool
    lines: int
    notes: list[int] = field(default_factory=list)


def _is_sup(span: dict, line_size: float, following: str = "") -> bool:
    """Footnote markers are digit-only spans that are superscript-flagged or much smaller than their line.

    A small digit glued to the next letter is a chemical subscript ("CO2e"), not a marker.
    """
    text = span["text"].strip()
    glued = following[:1].isalpha()
    return bool(text) and text.isdigit() and not glued and (bool(span["flags"] & 1) or span["size"] < 0.72 * line_size)


def clean(text: str) -> str:
    """Normalise whitespace (including non-breaking spaces) and tidy marker spacing."""
    text = text.replace("\xa0", " ").replace("\t", " ")
    return re.sub(r"\s+", " ", text).strip()


def page_spans(page: pymupdf.Page) -> list[Span]:
    """Every non-empty span on the page, with superscript digits flagged."""
    out: list[Span] = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            line_size = max((s["size"] for s in line["spans"] if s["text"].strip()), default=0)
            spans = line["spans"]
            for k, s in enumerate(spans):
                if s["text"].strip():
                    nxt = spans[k + 1]["text"] if k + 1 < len(spans) else ""
                    out.append(Span(*s["bbox"], s["text"], s["size"], _is_sup(s, line_size, nxt)))
    return out


def page_blocks(page: pymupdf.Page) -> list[Block]:
    """Text blocks of a page; superscript markers become [^n] and are also listed in Block.notes."""
    out: list[Block] = []
    for b in page.get_text("dict")["blocks"]:
        if b["type"] != 0:
            continue
        parts: list[str] = []
        notes: list[int] = []
        sizes: list[float] = []
        bold_flags: list[bool] = []
        for line in b["lines"]:
            line_size = max((s["size"] for s in line["spans"] if s["text"].strip()), default=0)
            spans = line["spans"]
            for k, s in enumerate(spans):
                if not s["text"].strip():
                    parts.append(" ")
                    continue
                nxt = spans[k + 1]["text"] if k + 1 < len(spans) else ""
                if _is_sup(s, line_size, nxt):
                    notes.append(int(s["text"].strip()))
                    parts.append(f"[^{s['text'].strip()}]")
                else:
                    parts.append(s["text"])
                    sizes.append(s["size"])
                    bold_flags.append("Semibold" in s["font"] or "Bold" in s["font"])
            parts.append(" ")
        text = clean("".join(parts)).replace(" [^", "[^")
        if text and sizes:
            out.append(Block(*b["bbox"], text, max(sizes), all(bold_flags), len(b["lines"]), notes))
    return out


def strip_markers(text: str) -> str:
    """Remove [^n] tokens (what is stored and shown) and the space they leave behind."""
    return re.sub(r"\s+([.,;:])", r"\1", MARKER.sub("", text)).strip()


def marker_numbers(text: str) -> list[int]:
    """Endnote numbers cited in a text, in order of appearance, without duplicates."""
    seen: list[int] = []
    for m in MARKER.finditer(text):
        n = int(m.group(1))
        if n not in seen:
            seen.append(n)
    return seen
