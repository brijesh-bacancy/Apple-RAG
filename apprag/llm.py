"""Answer generation. Providers: gemini / openai / ollama, or `extractive` (no LLM, fully offline)."""
import re
import time

import requests

import base64

from .bm25 import tokenize

NOT_FOUND = "I could not find this in the document."

SYSTEM_PROMPT = f"""You answer questions about one document, the Apple iMac Product Environmental Report (October 2024), using ONLY the numbered passages.
- Use only facts stated in the passages. Never add outside knowledge.
- Passages can be prose, whole tables (one "row - column: value" line per cell), chart descriptions, page descriptions or numbered endnotes. Read the row and column of a table value before quoting it.
- Quote numbers, units and configurations exactly (iMac two ports 256GB and four ports 512GB have different figures).
- When a passage ends with [Note n], that endnote qualifies the claim: include the caveat when it matters.
- Cite the passages you used like [1] or [2][3] and name the PDF page. If passages give different figures, report each with its context.
- If the passages do not contain the answer, reply exactly: {NOT_FOUND}
- Be brief: one to three sentences."""

_RETRY = {429, 500, 502, 503, 504}


def build_prompt(question: str, context: str) -> str:
    """Assemble the grounded prompt sent to a hosted/local LLM."""
    return f"{SYSTEM_PROMPT}\n\nPASSAGES:\n{context}\n\nQUESTION:\n{question}\n\nANSWER:"


def _post(url: str, **kw) -> requests.Response:
    """POST with a second attempt on transient errors."""
    last: Exception | None = None
    for attempt in range(1, 4):
        try:
            r = requests.post(url, **kw)
            if r.status_code not in _RETRY:
                r.raise_for_status()
                return r
            last = RuntimeError(f"HTTP {r.status_code}")
        except requests.exceptions.RequestException as exc:
            last = exc
        time.sleep(2 * attempt)
    raise last


class LLM:
    def __init__(self, settings):
        self.s = settings

    def generate(self, question: str, context: str, evidence: list) -> str:
        """Produce a grounded answer with the configured provider.

        @param question User question.
        @param context Numbered passages already rendered as text.
        @param evidence Passages (Chunk objects) for the offline extractive fallback.
        @return Answer text.
        """
        if self.s.llm_provider == "extractive":
            return self._extractive(question, evidence)
        return self.complete(build_prompt(question, context))

    def complete(self, prompt: str) -> str:
        """Send a raw prompt to the configured hosted/local model.

        @param prompt Full prompt text.
        @return Model output.
        @throws RuntimeError when the provider is unknown or its key is missing.
        """
        p = self.s.llm_provider
        if p == "gemini":
            return self._gemini(prompt)
        if p == "openai":
            return self._openai(prompt)
        if p == "ollama":
            return self._ollama(prompt)
        raise RuntimeError("LLM_PROVIDER must be extractive, ollama, openai or gemini")

    def _extractive(self, question: str, evidence: list) -> str:
        """Offline fallback: the best question-matching sentences from the top passages, with page cites."""
        if evidence and evidence[0].kind == "footnote" and re.search(r"(?:end|foot)?note\s+\d", question, re.I):
            return f"{evidence[0].text} [p.{evidence[0].page}]"
        q = set(tokenize(question))
        scored = []
        for rank, chunk in enumerate(evidence[:3]):
            for sent in re.split(r"(?<=[.!?])\s+|\n", chunk.text):
                if sent.startswith("Table:"):
                    continue  # the table's header line matches many questions but holds no value
                overlap = len(q & set(tokenize(sent)))
                if overlap >= 2:
                    scored.append((overlap - rank * 0.25, sent.strip(), chunk.page))
        if not scored:
            return NOT_FOUND
        top = sorted(scored, key=lambda x: -x[0])[:3]
        return "\n".join(f"- {s} [p.{p}]" for _, s, p in top)

    def describe_image(self, png: bytes, prompt: str) -> str:
        """Ask the vision model (Gemini) to describe one page image.

        @param png PNG bytes of the rendered page.
        @param prompt Instruction for the description.
        @return The description text.
        @throws RuntimeError when GEMINI_API_KEY is missing or the call keeps failing.
        """
        if not self.s.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is missing")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.s.gemini_model}:generateContent"
        parts = [{"text": prompt}, {"inline_data": {"mime_type": "image/png", "data": base64.b64encode(png).decode()}}]
        r = _post(
            url,
            headers={"x-goog-api-key": self.s.gemini_api_key},
            json={"contents": [{"parts": parts}], "generationConfig": {"temperature": 0.0}},
            timeout=(10, 90),
        )
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]

    def _gemini(self, prompt: str) -> str:
        if not self.s.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is missing")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.s.gemini_model}:generateContent"
        r = _post(
            url,
            headers={"x-goog-api-key": self.s.gemini_api_key},
            json={"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.1}},
            timeout=(10, 60),
        )
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]

    def _openai(self, prompt: str) -> str:
        if not self.s.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is missing")
        r = _post(
            self.s.openai_base_url.rstrip("/") + "/chat/completions",
            headers={"Authorization": f"Bearer {self.s.openai_api_key}"},
            json={"model": self.s.openai_model, "temperature": 0.1, "messages": [{"role": "user", "content": prompt}]},
            timeout=(10, 60),
        )
        return r.json()["choices"][0]["message"]["content"]

    def _ollama(self, prompt: str) -> str:
        r = _post(
            self.s.ollama_base_url.rstrip("/") + "/api/chat",
            json={
                "model": self.s.ollama_model,
                "stream": False,
                "options": {"temperature": 0.1},
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=(10, 120),
        )
        return r.json()["message"]["content"]
