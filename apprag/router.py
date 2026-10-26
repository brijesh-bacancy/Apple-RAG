"""Query understanding by rules only: this report is small and flat, so there is nothing to filter except explicit references."""
import re

from .models import Query

_NOTE = re.compile(r"\b(?:end|foot)?note\s+(\d{1,2})\b", re.I)
_PAGE = re.compile(r"\bpage\s+(\d{1,2})\b", re.I)


def route(question: str) -> Query:
    """Find an explicit footnote number or page number in the question.

    @param question User question.
    @return Query with note_no / page set when the question names one.
    """
    q = Query(text=question)
    if m := _NOTE.search(question):
        q.note_no = int(m.group(1))
    if m := _PAGE.search(question):
        q.page = int(m.group(1))
    return q
