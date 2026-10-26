"""CLI:  python -m apprag ingest | ask "question" | eval | compare"""
import sys

from .config import BASE_DIR, get_settings

PDF = BASE_DIR / "data" / "iMac_PER_Oct2024.pdf"
QUESTIONS = BASE_DIR / "eval" / "questions.json"


def main(argv: list[str]) -> int:
    """Dispatch a sub-command."""
    from .pipeline import RAGPipeline

    cmd = argv[0] if argv else "help"
    if cmd == "ingest":
        rep = RAGPipeline(get_settings()).ingest(argv[1] if len(argv) > 1 else PDF)
        print(f"Indexed {rep['chunks']} elements {rep['kinds']}; {rep['notes']} endnotes linked; visual pages {rep['visual_pages']}")
        if rep["vision_failed"]:
            print(f"WARNING: no vision description for pages {rep['vision_failed']} (set GEMINI_API_KEY or retry later)")
        return 0
    if cmd in ("ask", "eval", "compare"):
        rag = RAGPipeline(get_settings())
        rag.load_index()
        if cmd == "ask":
            res = rag.ask(" ".join(argv[1:]))
            print(res.answer)
            for r in res.used:
                print(f"  p.{r.chunk.page} {r.chunk.kind:<8} {r.chunk.heading_path}")
            return 0
        from .evaluate import run_compare, run_eval

        return run_eval(rag, QUESTIONS) if cmd == "eval" else run_compare(rag, PDF, QUESTIONS)
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
