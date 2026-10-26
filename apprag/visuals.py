"""Pages whose meaning is in artwork: find them, describe them with a vision model once, and cache the result."""
import hashlib
import json
from pathlib import Path

import pymupdf

VISION_PROMPT = """You are describing one page of Apple's iMac Product Environmental Report so it can be searched.
Write plain prose, 100 to 220 words. Transcribe every number with its unit and label exactly as printed.
For a chart, say what the bars or segments are, give each bar's total and each labelled segment, and state the arithmetic
relationship between the numbers (for example which total minus which reduction gives which result).
For an icon or diagram, say what it shows and the text attached to it. Do not guess values that are not printed."""

MIN_DRAWINGS = 20


def visual_pages(doc: pymupdf.Document) -> list[int]:
    """1-based pages that carry images or real vector artwork (a plain text page has nothing to see)."""
    return [
        i for i, p in enumerate(doc, 1)
        if p.get_images() or len(p.get_drawings()) >= MIN_DRAWINGS
    ]


def _load(cache: Path) -> dict[str, str]:
    return json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}


def describe_pages(doc: pymupdf.Document, pdf_path: Path, llm, cache: Path) -> tuple[dict[int, str], list[int]]:
    """Describe every visual page, reusing cached descriptions so ingestion is repeatable and cheap.

    @param doc Opened PDF.
    @param pdf_path PDF path (its content hash keys the cache, so an edited PDF is described again).
    @param llm LLM wrapper exposing describe_image(); a failure on one page is reported, not raised.
    @param cache JSON file holding page descriptions.
    @return (descriptions by page, pages that could not be described).
    """
    digest = hashlib.sha1(Path(pdf_path).read_bytes()).hexdigest()[:12]
    store = _load(cache)
    out: dict[int, str] = {}
    failed: list[int] = []
    for pn in visual_pages(doc):
        key = f"{digest}:{pn}"
        if key not in store:
            try:
                store[key] = llm.describe_image(doc[pn - 1].get_pixmap(dpi=110).tobytes("png"), VISION_PROMPT).strip()
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(store, ensure_ascii=False, indent=1), encoding="utf-8")
            except Exception:  # noqa: BLE001 - one failed page must not stop ingestion
                failed.append(pn)
                continue
        out[pn] = store[key]
    return out, failed
