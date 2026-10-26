"""Sentence-transformer embeddings (L2-normalised, so dot product equals cosine)."""
import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        """Embed texts into a float32 matrix of shape (len(texts), dim)."""
        return np.asarray(
            self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=32),
            dtype=np.float32,
        )
