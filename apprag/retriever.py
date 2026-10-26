"""Hybrid retrieval (dense + BM25, Reciprocal Rank Fusion) with optional cross-encoder reranking."""
import numpy as np
from sentence_transformers import CrossEncoder

from .bm25 import BM25
from .embeddings import Embedder
from .models import Retrieved
from .store import VectorStore

RRF_K = 60


class HybridRetriever:
    def __init__(self, store: VectorStore, embedder: Embedder):
        self.store, self.embedder = store, embedder
        self.bm25 = BM25([c.index_text or c.text for c in store.chunks])

    def retrieve(self, query: str, top_k: int, mask: np.ndarray | None = None) -> list[Retrieved]:
        """Fuse dense and keyword rankings over the allowed chunks.

        @param query Search text.
        @param top_k Number of fused candidates to return.
        @param mask Boolean mask of searchable chunks; None searches everything.
        @return Candidates ordered by fused score.
        """
        pool = max(top_k * 2, 20)
        qv = self.embedder.encode([query])[0]
        dense = self.store.search(qv, pool, mask)
        dense_sim = dict(dense)
        kw = np.asarray(self.bm25.scores(query))
        if mask is not None:
            kw = np.where(mask, kw, 0.0)
        sparse = [int(i) for i in np.argsort(-kw)[:pool] if kw[i] > 0]
        fused: dict[int, Retrieved] = {}
        for rank, (i, sim) in enumerate(dense):
            r = fused.setdefault(i, Retrieved(self.store.chunks[i]))
            r.vector_score = sim
            r.fused_score += 1 / (RRF_K + rank + 1)
        for rank, i in enumerate(sparse):
            r = fused.setdefault(i, Retrieved(self.store.chunks[i]))
            r.vector_score = dense_sim.get(i, float(self.store.vectors[i] @ qv))
            r.keyword_score = float(kw[i])
            r.fused_score += 1 / (RRF_K + rank + 1)
        return sorted(fused.values(), key=lambda r: -r.fused_score)[:top_k]


class Reranker:
    """Cross-encoder reranker; falls back to fused order if the model cannot load."""

    PROSE_KINDS = {"text", "stat", "footnote", "vision"}

    def __init__(self, model_name: str):
        self.model_name, self.model = model_name, None

    def rerank(self, query: str, items: list[Retrieved], top_k: int, mode: str = "all") -> list[Retrieved]:
        """Reorder candidates by cross-encoder relevance and keep the best top_k.

        @param query Question text.
        @param items Candidates from the hybrid stage.
        @param top_k Number to keep.
        @param mode "none" keeps the fused order; "prose" reranks prose-like chunks and leaves tables and charts at their
            fused rank among themselves; "all" reranks everything.
        @return Best candidates, each with rerank_score set where the model ran.
        """
        if not items or mode == "none":
            return items[:top_k]
        target = [r for r in items if mode == "all" or r.chunk.kind in self.PROSE_KINDS]
        try:
            if self.model is None:
                self.model = CrossEncoder(self.model_name)
            scores = self.model.predict([(query, r.chunk.index_text or r.chunk.text) for r in target])
        except Exception:  # noqa: BLE001 - offline / missing model: keep the fused order
            return items[:top_k]
        for r, sc in zip(target, scores):
            r.rerank_score = float(sc)
        if mode == "all":
            return sorted(items, key=lambda r: -r.rerank_score)[:top_k]
        # "prose": non-prose chunks keep their fused position; prose chunks are re-sorted among the prose slots.
        prose_sorted = iter(sorted(target, key=lambda r: -r.rerank_score))
        return [next(prose_sorted) if r in target else r for r in items][:top_k]
