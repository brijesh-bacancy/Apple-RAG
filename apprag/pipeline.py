"""End-to-end RAG: ingest elements -> route -> hybrid retrieval -> (optional) rerank -> context with linked endnotes -> generate."""
from pathlib import Path

import pymupdf

from .elements import build_chunks
from .embeddings import Embedder
from .llm import LLM, NOT_FOUND
from .models import Answer, Chunk, Query, Retrieved
from .retriever import HybridRetriever, Reranker
from .router import route
from .store import VectorStore

MIN_VECTOR_SCORE = 0.2  # best cosine below this means nothing in the report is about the question


class RAGPipeline:
    def __init__(self, settings):
        self.s = settings
        self.embedder = Embedder(settings.embedding_model)
        self.store = VectorStore(settings.index_dir)
        self.reranker = Reranker(settings.rerank_model)
        self.llm = LLM(settings)
        self.retriever: HybridRetriever | None = None
        self.last_report: dict = {}

    def ingest(self, pdf_path: str | Path) -> dict:
        """Extract every element (tables, charts, endnotes, vision), embed and persist.

        @param pdf_path Path of the report PDF.
        @return Counts per kind plus extraction report (endnotes found, visual pages, pages the vision model failed on).
        """
        doc = pymupdf.open(str(pdf_path))
        chunks, report = build_chunks(doc, Path(pdf_path), self.llm, self.s.vision_cache, self.s.use_vision)
        self.store.build(chunks, self.embedder.encode([c.index_text or c.text for c in chunks]))
        self.store.save()
        self.retriever = HybridRetriever(self.store, self.embedder)
        kinds: dict[str, int] = {}
        for c in chunks:
            kinds[c.kind] = kinds.get(c.kind, 0) + 1
        self.last_report = {"chunks": len(chunks), "kinds": kinds, **report}
        return self.last_report

    def load_index(self) -> None:
        """Load the persisted index and prepare the retriever."""
        self.store.load()
        self.retriever = HybridRetriever(self.store, self.embedder)

    def note_chunk(self, number: int) -> Chunk | None:
        """The endnote chunk with this number, if the report has one."""
        return next((c for c in self.store.chunks if c.kind == "footnote" and c.note_no == number), None)

    def search(self, question: str, top_k: int | None = None) -> tuple[Query, list[Retrieved]]:
        """Route, retrieve and rerank; a question naming "note N" gets that endnote first.

        @param question User question.
        @param top_k Chunks to keep (defaults to settings).
        @return (query plan, ranked hits).
        @throws RuntimeError when no index is loaded.
        """
        if self.retriever is None:
            raise RuntimeError("No index loaded. Run ingest first.")
        top_k = top_k or self.s.rerank_top_k
        query = route(question)
        hits = self.retriever.retrieve(question, self.s.top_k, self.store.mask(query))
        hits = self.reranker.rerank(question, hits, self.s.top_k, self.s.rerank_mode)[:top_k]
        if query.note_no is not None and (note := self.note_chunk(query.note_no)):
            hits = [Retrieved(note, fused_score=1.0), *[h for h in hits if h.chunk.id != note.id]][:top_k]
        return query, hits

    def render(self, chunk: Chunk) -> str:
        """Passage text for the model: the element plus the endnotes it cites (or the claims citing an endnote)."""
        text = chunk.text
        for n in chunk.notes:
            if note := self.note_chunk(n):
                text += f"\n[Note {n}] {note.text.split(': ', 1)[1][:700]}"
        if chunk.kind == "footnote" and chunk.claims:
            text += "\nCited by: " + " | ".join(chunk.claims)
        return text

    def ask(self, question: str) -> Answer:
        """Answer a question from the indexed report, with page citations.

        @param question User question.
        @return Answer carrying the query plan, the hits and the exact passages sent to the model.
        """
        query, hits = self.search(question)
        if not hits or (query.note_no is None and max(h.vector_score for h in hits) < MIN_VECTOR_SCORE):
            return Answer(question, NOT_FOUND, query, hits, [])
        context = [
            f"[{i}] (PDF page {h.chunk.page}, {h.chunk.kind}: {h.chunk.heading_path})\n{self.render(h.chunk)}"
            for i, h in enumerate(hits, 1)
        ]
        answer = self.llm.generate(question, "\n\n".join(context), [h.chunk for h in hits])
        return Answer(question, answer, query, hits, context)
