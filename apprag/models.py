"""Plain data objects shared by the pipeline."""
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """One retrievable element of the report: a prose passage, a whole table, a chart, a page description or an endnote."""

    id: str
    text: str  # what is shown and cited, footnote markers removed
    page: int
    kind: str  # text | stat | table | chart | vision | footnote
    heading_path: str = ""
    notes: list[int] = field(default_factory=list)  # endnotes this element cites
    note_no: int = 0  # for kind == "footnote": which endnote this is
    claims: list[str] = field(default_factory=list)  # for footnotes: the claims that cite it
    index_text: str = ""  # what gets embedded and keyword-indexed (adds heading, cited notes, claims)


@dataclass
class Query:
    """What the router understood about a question."""

    text: str
    note_no: int | None = None
    page: int | None = None


@dataclass
class Retrieved:
    """A chunk with its scores through the stages."""

    chunk: Chunk
    vector_score: float = 0.0
    keyword_score: float = 0.0
    fused_score: float = 0.0
    rerank_score: float = 0.0


@dataclass
class Answer:
    """Final result of one question."""

    question: str
    answer: str
    query: Query
    used: list[Retrieved]
    context: list[str]  # the numbered passages exactly as sent to the model
