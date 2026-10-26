"""The plain-text baseline, kept only so the evaluation can show what element-aware extraction adds.

Raw `get_text()` of every page (tables flattened in output order, markers as stray digits, no vision, no endnote links),
cut into fixed 1000-character windows with 150 overlap.
"""
import pymupdf

from .models import Chunk


def build_naive_chunks(doc: pymupdf.Document, size: int = 1000, overlap: int = 150) -> list[Chunk]:
    """Slide a fixed character window over the raw text of every page.

    @param doc Opened PDF.
    @param size Window length in characters.
    @param overlap Characters shared by neighbouring windows.
    @return Flat chunks with only a page number.
    """
    chunks: list[Chunk] = []
    for pn in range(1, doc.page_count + 1):
        text = " ".join(doc[pn - 1].get_text().split())
        start = 0
        while start < len(text):
            piece = text[start:start + size]
            if piece.strip():
                chunks.append(Chunk(id=f"f{len(chunks):04d}", text=piece, page=pn, kind="text", index_text=piece))
            if start + size >= len(text):
                break
            start += size - overlap
    return chunks
