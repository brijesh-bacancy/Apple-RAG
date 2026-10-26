"""Small BM25 implementation for keyword matching over the whole corpus."""
import math
import re
from collections import Counter

_STOP = {
    "the", "a", "an", "of", "in", "to", "and", "is", "are", "was", "were", "for", "on", "by",
    "what", "how", "many", "did", "does", "do", "which", "with", "as", "at", "from", "that",
    "this", "it", "its",
}


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, keeping numbers like 874.5 or 1,737 intact, minus stopwords."""
    return [t for t in re.findall(r"\w+(?:[.,]\w+)*", text.lower()) if t not in _STOP]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(tokenize(d)) for d in docs]
        self.len = [sum(t.values()) for t in self.tf]
        self.avgdl = sum(self.len) / max(len(docs), 1)
        self.df = Counter(t for tf in self.tf for t in tf)
        self.n = len(docs)

    def scores(self, query: str) -> list[float]:
        """BM25 score of the query against every document."""
        q = tokenize(query)
        out = []
        for tf, dl in zip(self.tf, self.len):
            s = 0.0
            for term in q:
                f = tf.get(term, 0)
                if not f:
                    continue
                idf = math.log(1 + (self.n - self.df[term] + 0.5) / (self.df[term] + 0.5))
                s += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1)))
            out.append(s)
        return out
