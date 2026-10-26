"""File-backed index: element vectors in .npy, chunks (with metadata) in .json. Supports page / note lookups."""
import json
from pathlib import Path

import numpy as np

from .models import Chunk, Query


class VectorStore:
    def __init__(self, index_dir: Path):
        self.dir = Path(index_dir)
        self.chunks: list[Chunk] = []
        self.vectors = np.zeros((0, 0), dtype=np.float32)

    def build(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        """Replace the in-memory index."""
        self.chunks, self.vectors = chunks, vectors

    def save(self) -> None:
        """Persist the index to disk."""
        self.dir.mkdir(parents=True, exist_ok=True)
        np.save(self.dir / "vectors.npy", self.vectors)
        (self.dir / "chunks.json").write_text(json.dumps([c.__dict__ for c in self.chunks], ensure_ascii=False), encoding="utf-8")

    def load(self) -> None:
        """Load a previously saved index; raises FileNotFoundError if absent."""
        self.vectors = np.load(self.dir / "vectors.npy")
        self.chunks = [Chunk(**d) for d in json.loads((self.dir / "chunks.json").read_text(encoding="utf-8"))]

    def mask(self, query: Query | None) -> np.ndarray:
        """Boolean mask of searchable chunks: only the explicit "page N" of a question narrows the search."""
        keep = np.ones(len(self.chunks), dtype=bool)
        if query is not None and query.page is not None:
            keep = np.array([c.page == query.page for c in self.chunks])
        return keep

    def search(self, query_vec: np.ndarray, k: int, mask: np.ndarray | None = None) -> list[tuple[int, float]]:
        """Top-k (chunk index, cosine similarity) among the allowed chunks."""
        sims = self.vectors @ query_vec
        if mask is not None:
            sims = np.where(mask, sims, -np.inf)
        return [(int(i), float(sims[i])) for i in np.argsort(-sims)[:k] if np.isfinite(sims[i])]
