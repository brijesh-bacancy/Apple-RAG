from pathlib import Path

import pymupdf
import pytest

from apprag.elements import build_chunks, chart_text
from apprag.extract import Block, marker_numbers, page_blocks, strip_markers
from apprag.models import Chunk, Query, Retrieved
from apprag.notes import read_endnotes
from apprag.retriever import Reranker
from apprag.router import route
from apprag.store import VectorStore
from apprag.tables import find_tables, verbalise

PDF = Path(__file__).resolve().parent.parent / "data" / "iMac_PER_Oct2024.pdf"
pytestmark = pytest.mark.skipif(not PDF.exists(), reason="report PDF not present")


@pytest.fixture(scope="module")
def doc():
    return pymupdf.open(str(PDF))


@pytest.fixture(scope="module")
def chunks(doc):
    return build_chunks(doc, PDF, None, Path("unused.json"), use_vision=False)[0]


def test_markers_are_extracted_and_stripped():
    text = "Reduced by 20 percent.[^7] Later[^10]"
    assert marker_numbers(text) == [7, 10]
    assert strip_markers(text) == "Reduced by 20 percent. Later"


def test_superscript_is_marker_but_subscript_is_not(doc):
    text = " ".join(b.text for b in page_blocks(doc[1]))
    assert "[^7]" in text and "[^11]" in text
    table = verbalise(find_tables(doc[9], 10)[0])[0]
    assert "346 kg CO2e" in table and "[^2]" not in table


def test_page_10_table_keeps_every_value_with_its_row(doc):
    text = verbalise(find_tables(doc[9], 10)[0])[0]
    assert "Total product footprint — iMac (two ports) 256GB: 346 kg CO2e." in text
    assert "Life cycle product emissions (scope 3) > Production — iMac (two ports) 256GB: 50%." in text
    assert "End-of-life processing — iMac (two ports) 256GB: <1%." in text


def test_page_12_table_has_row_and_column_for_every_cell(doc):
    (table,) = find_tables(doc[11], 12)
    text = verbalise(table)[0]
    assert table.columns == ["100V", "115V", "230V"]
    assert "Sleep — Power consumption for iMac 115V: 0.57W." in text
    assert "Power adapter efficiency — Power consumption for iMac 230V: 91.9%." in text
    assert text.count("\n") == 15  # header line + 5 rows x 3 columns


def test_all_nineteen_endnotes_are_parsed_without_table_leakage(doc):
    notes = read_endnotes(doc, {12: [t.bbox for t in find_tables(doc[11], 12)]})
    assert sorted(notes) == list(range(1, 20))
    assert notes[14] == "Recycled copper in the thermal module applies to iMac (four ports) only."
    assert "0.27W" not in notes[18] and "Sleep: Low power state" in notes[18]


def test_chart_labels_pair_each_figure_with_its_label():
    blocks = [
        Block(250, 248, 300, 262, "445 kg", 12, True, 1),
        Block(240, 264, 310, 274, "Baseline emissions[^7]", 7.5, False, 1),
        Block(350, 248, 400, 262, "–99 kg", 12, True, 1),
        Block(335, 264, 410, 274, "Emissions reductions", 7.5, False, 1),
    ]
    text = chart_text(blocks)
    assert "445 kg — Baseline emissions[^7]" in text and "–99 kg — Emissions reductions" in text


def test_elements_cover_every_content_type(chunks):
    kinds = {c.kind for c in chunks}
    assert {"text", "stat", "table", "chart", "footnote"} <= kinds
    assert sum(c.kind == "table" for c in chunks) == 3


def test_footnotes_link_both_ways(chunks):
    copper = next(c for c in chunks if c.kind == "text" and c.text.startswith("Copper."))
    assert 14 in copper.notes
    note14 = next(c for c in chunks if c.note_no == 14)
    assert any("Copper" in claim or "thermal module" in claim for claim in note14.claims)
    assert "Note 14" in copper.index_text and "four ports" in copper.index_text


def test_table_on_the_endnote_page_is_linked_to_the_note_it_follows(chunks):
    power = next(c for c in chunks if c.kind == "table" and c.page == 12)
    assert power.notes == [18]


def test_markers_never_leak_into_stored_text(chunks):
    assert not [c.id for c in chunks if "[^" in c.text]


def test_router_finds_explicit_note_and_page_only():
    assert route("What does footnote 16 say?").note_no == 16
    assert route("what is on page 10?").page == 10
    q = route("How much power in sleep mode?")
    assert q.note_no is None and q.page is None


def test_page_mask_narrows_the_search(tmp_path):
    store = VectorStore(tmp_path)
    store.chunks = [Chunk("a", "x", 1, "text"), Chunk("b", "x", 10, "table")]
    assert store.mask(Query("q", page=10)).tolist() == [False, True]
    assert store.mask(Query("q")).tolist() == [True, True]


class FakeModel:
    """Scores a passage by its length so the reordering is predictable."""

    def predict(self, pairs):
        return [float(len(text)) for _, text in pairs]


def _hits():
    return [
        Retrieved(Chunk("t", "table row", 10, "table", index_text="table row")),
        Retrieved(Chunk("a", "a", 1, "text", index_text="a")),
        Retrieved(Chunk("b", "bbbbbb", 2, "text", index_text="bbbbbb")),
    ]


def test_rerank_modes():
    r = Reranker("fake")
    r.model = FakeModel()
    assert [h.chunk.id for h in r.rerank("q", _hits(), 3, "none")] == ["t", "a", "b"]
    assert [h.chunk.id for h in r.rerank("q", _hits(), 3, "all")] == ["t", "b", "a"]
    # prose-only: the table keeps its slot, prose chunks are re-sorted among theirs
    assert [h.chunk.id for h in r.rerank("q", _hits(), 3, "prose")] == ["t", "b", "a"]
