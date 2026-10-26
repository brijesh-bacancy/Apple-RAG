"""Validate the pipeline against eval/questions.json and measure what each technique adds.

A case has `expected` (strings that must all appear in one retrieved element, with the endnotes it cites), `answer`
(strings the answer must contain), `pages` (acceptable PDF pages) and optionally `refusal: true` for questions the report
cannot answer (the right output is "I could not find this in the document.").
"""
import json
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pymupdf

from .llm import NOT_FOUND
from .models import Chunk
from .naive import build_naive_chunks
from .retriever import HybridRetriever
from .store import VectorStore


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower())


def full_text(rag, chunk: Chunk) -> str:
    """What a reader of this element sees: its text plus the endnotes it cites."""
    return rag.render(chunk) if chunk.kind != "footnote" else chunk.text


def _rank(texts: list[str], expected: list[str]) -> int:
    """1-based rank of the first text containing every expected string, or 0."""
    return next((i for i, t in enumerate(texts, 1) if all(_norm(e) in _norm(t) for e in expected)), 0)


def run_eval(rag, path: str | Path) -> int:
    """Run all cases through the full pipeline; print per-question results and a summary.

    @param rag Loaded RAGPipeline.
    @param path Question file.
    @return 0 when every retrieval and refusal check passes.
    """
    cases = json.loads(Path(path).read_text(encoding="utf-8"))
    ret = ans = pg = refused = n_refuse = 0
    by_cat: dict[str, list[int]] = {}
    for n, c in enumerate(cases, 1):
        res = rag.ask(c["question"])
        if c.get("refusal"):
            n_refuse += 1
            ok = res.answer.strip() == NOT_FOUND
            refused += ok
            print(f"[{n:02d}] refusal={'OK ' if ok else 'MISS'} | {c['question']}\n      {res.answer.splitlines()[0][:140]}")
            continue
        evidence = [full_text(rag, h.chunk) for h in res.used]
        r_ = _rank(evidence, c["expected"]) > 0
        a_ = all(_norm(x) in _norm(res.answer) for x in c["answer"])
        p_ = any(h.chunk.page in c["pages"] for h in res.used)
        ret, ans, pg = ret + r_, ans + a_, pg + p_
        by_cat.setdefault(c["category"], []).append(int(r_))
        kinds = ",".join(h.chunk.kind[:3] for h in res.used)
        print(f"[{n:02d}] retrieval={'OK ' if r_ else 'MISS'} answer={'OK ' if a_ else 'MISS'} page={'OK ' if p_ else 'MISS'} "
              f"[{kinds}] | {c['question']}\n      {res.answer.splitlines()[0][:150] if res.answer else ''}")
    t = len(cases) - n_refuse
    print(f"\nRetrieval {ret}/{t} | Answer {ans}/{t} | Page {pg}/{t} | Refusals {refused}/{n_refuse}")
    print("Retrieval by type: " + ", ".join(f"{k} {sum(v)}/{len(v)}" for k, v in by_cat.items()))
    return 0 if ret == t and refused == n_refuse else 1


def run_compare(rag, pdf: str | Path, path: str | Path, k: int = 5) -> int:
    """Compare plain-text chunks with element-aware retrieval, one technique at a time.

    All systems return k elements per question; hit means one of them contains every expected string.

    @param rag Loaded RAGPipeline (embedder, reranker and the element index).
    @param pdf Report PDF used to build the plain-text baseline.
    @param path Question file.
    @param k Elements scored per question.
    @return 0.
    """
    cases = [c for c in json.loads(Path(path).read_text(encoding="utf-8")) if not c.get("refusal")]
    flat_chunks = build_naive_chunks(pymupdf.open(str(pdf)))
    flat_store = VectorStore(Path(pdf).parent / "_flat_index")
    flat_store.build(flat_chunks, rag.embedder.encode([c.text for c in flat_chunks]))
    flat = HybridRetriever(flat_store, rag.embedder)

    def element_ranks(mode: str, drop: set[str] = frozenset()) -> list[int]:
        out = []
        for c in cases:
            mask = np.array([ch.kind not in drop for ch in rag.store.chunks])
            hits = rag.retriever.retrieve(c["question"], rag.s.top_k, mask)
            hits = rag.reranker.rerank(c["question"], hits, k, mode)[:k]
            out.append(_rank([full_text(rag, h.chunk) for h in hits], c["expected"]))
        return out

    rows: dict[str, list[int]] = {}
    rows["plain text, 1000-char chunks + rerank"] = [
        _rank([h.chunk.text for h in rag.reranker.rerank(c["question"], flat.retrieve(c["question"], rag.s.top_k), k, "all")], c["expected"])
        for c in cases
    ]
    rows["elements, no rerank"] = element_ranks("none")
    rows["elements + rerank prose only"] = element_ranks("prose")
    rows["elements + rerank everything"] = element_ranks("all")
    rows["elements, no vision descriptions"] = element_ranks("none", {"vision"})
    for i, c in enumerate(cases):
        print(" ".join(f"{r[i]}" for r in rows.values()), "|", c["category"], "|", c["question"][:70])
    print(f"\n{'system':<42}{'hit@' + str(k):>8}{'hit@1':>8}{'MRR':>8}")
    for name, ranks in rows.items():
        a = np.array(ranks)
        print(f"{name:<42}{(a > 0).sum():>5}/{len(a):<2}{(a == 1).sum():>5}/{len(a):<2}{np.where(a > 0, 1 / np.maximum(a, 1), 0).mean():>8.3f}")
    print("\nHits by question type (plain text -> elements):")
    for cat in dict.fromkeys(c["category"] for c in cases):
        idx = [i for i, c in enumerate(cases) if c["category"] == cat]
        base, ours = list(rows.values())[0], list(rows.values())[1]
        print(f"  {cat:<16} {sum(base[i] > 0 for i in idx)}/{len(idx)} -> {sum(ours[i] > 0 for i in idx)}/{len(idx)}")
    return 0
