"""Browser UI:  streamlit run app.py"""
import pymupdf
import streamlit as st

from apprag.config import BASE_DIR, get_settings
from apprag.pipeline import RAGPipeline

PDF = BASE_DIR / "data" / "iMac_PER_Oct2024.pdf"

BY_TYPE = {
    "📊 Tables": [
        "How much power does the iMac use in sleep mode at 115V?",
        "What is the carbon footprint of the iMac with four ports and 512GB?",
    ],
    "📈 Chart": [
        "What was the baseline emissions figure, and how many kilograms did the reductions remove?",
    ],
    "🔖 Footnotes": [
        "Does the recycled copper in the thermal module apply to all iMac models?",
        "What does footnote 16 say?",
    ],
    "💬 Call-outs and text": [
        "How much recycled content does the iMac fiber packaging contain?",
        "What is the screen resolution of the iMac?",
    ],
}
BADGE = {
    "table": ("#0f766e", "TABLE"), "chart": ("#7c3aed", "CHART"), "vision": ("#9333ea", "PAGE IMAGE"),
    "footnote": ("#b45309", "ENDNOTE"), "stat": ("#be185d", "CALL-OUT"), "text": ("#475569", "TEXT"),
}

st.set_page_config(page_title="iMac Evidence Board", page_icon="🖥️", layout="wide")
st.markdown(
    """
    <style>
      .badge {color:#fff; border-radius:4px; padding:1px 8px; font-size:.7rem; font-weight:700; letter-spacing:.6px;}
      .card {border:1px solid #e5e7eb; border-radius:10px; padding:.7rem .9rem; margin-bottom:.6rem;}
      .card small {color:#6b7280;}
      .answer {border-left:5px solid #0f766e; background:#f0fdfa; padding:.9rem 1.1rem; border-radius:6px; color:#134e4a;}
      pre.cells {white-space:pre-wrap; font-size:.8rem; margin:.3rem 0 0 0;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading models and index...")
def get_pipeline() -> RAGPipeline:
    """Load models once; build the index on first run if it does not exist."""
    rag = RAGPipeline(get_settings())
    try:
        rag.load_index()
    except FileNotFoundError:
        rag.ingest(PDF)
    return rag


@st.cache_data(show_spinner=False)
def page_image(number: int) -> bytes:
    """PNG of one PDF page for the preview pane."""
    return pymupdf.open(str(PDF))[number - 1].get_pixmap(dpi=85).tobytes("png")


rag = get_pipeline()
kinds: dict[str, int] = {}
for c in rag.store.chunks:
    kinds[c.kind] = kinds.get(c.kind, 0) + 1

with st.sidebar:
    st.title("🖥️ iMac Evidence Board")
    st.caption("Questions whose answer sits in a table, a chart or a footnote.")
    for group, questions in BY_TYPE.items():
        st.markdown(f"**{group}**")
        for q_text in questions:
            if st.button(q_text, key=q_text, width="stretch"):
                st.session_state["q"] = q_text
    st.divider()
    st.caption("Indexed elements: " + " · ".join(f"{n} {k}" for k, n in sorted(kinds.items())))
    st.caption(f"Answer engine: {rag.s.llm_provider} · rerank: {rag.s.rerank_mode}")

question = st.text_input("Ask about the iMac Product Environmental Report", key="q", placeholder="e.g. What is the idle power at 230V?")
if not question:
    st.info("Pick an example on the left or type a question. Each answer shows the exact table, chart or endnote it came from.")
    st.stop()

try:
    with st.spinner("Looking through tables, charts and endnotes..."):
        res = rag.ask(question)
except Exception as exc:
    st.error(f"The LLM call failed: {exc}")
    st.stop()

left, right = st.columns([3, 2], gap="large")
with left:
    st.markdown(f'<div class="answer">{res.answer.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)
    st.markdown("#### Evidence")
    for i, (hit, passage) in enumerate(zip(res.used, res.context), 1):
        colour, label = BADGE.get(hit.chunk.kind, ("#475569", hit.chunk.kind.upper()))
        body = passage.split("\n", 1)[1] if "\n" in passage else passage
        st.markdown(
            f'<div class="card"><span class="badge" style="background:{colour}">{label}</span> '
            f"<b>[{i}]</b> <small>page {hit.chunk.page} · {hit.chunk.heading_path}</small>"
            f'<pre class="cells">{body}</pre></div>',
            unsafe_allow_html=True,
        )
with right:
    pages = sorted({h.chunk.page for h in res.used})
    if pages:
        shown = st.radio("Cited pages", pages, horizontal=True, format_func=lambda p: f"p.{p}")
        st.image(page_image(shown), width="stretch")
