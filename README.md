# Apple-RAG — RAG over the iMac Product Environmental Report (12 pages, Oct 2024)

The answers in this report are not in running prose: they sit in tables, a bar chart, call-outs and 19 numbered endnotes.
Built on `WB-RAG` (hybrid search + reranking); the new work is almost all in **extraction**.

## What changed compared with WB-RAG

| Where the fact lives | What plain text extraction gives | What Apple-RAG does |
|---|---|---|
| **Tables** | values separated from their row and column headings | `tables.py` finds each table by its dark header bar and closing rule in the page's vector drawings, rebuilds rows and columns from span positions, and writes **one sentence per cell**: `Sleep — Power consumption for iMac 115V: 0.57W.` Bulleted rows keep their parent (`Life cycle product emissions (scope 3) > Production — …: 50%`). One chunk per table, never split. |
| **Chart** | loose labels (`445 kg`, `–99 kg`, `346 kg`) with no relationship | the chart is found from its coloured bars; each figure is paired with its label; and each visual page is described once by a Gemini vision call (cached in `data/vision_cache.json`), which states that 445 kg minus 99 kg gives 346 kg |
| **Footnotes** | the marker becomes a stray digit (`Apple.7`) | superscript digits become `[^n]` markers (a digit glued to a letter, as in CO₂e, is a subscript and is left alone). Endnotes are parsed into 19 chunks. **Both directions are linked**: an element lists the notes it cites and carries their text; a note lists the claims that cite it |
| **Call-outs** | `100%` and its caption come out in separate pieces | a big figure and the block under it become one `stat` chunk |
| **Page artwork** (icons, diagram) | nothing | the vision description of the page is a separate chunk |

Retrieval stays simple, because the structure is now in the chunks: hybrid dense + BM25 over `heading + text + cited notes`,
RRF, cross-encoder rerank (`RERANK_MODE=all|prose|none`), then the top 5 elements go to the model **with their endnotes attached**.
A question that names "footnote N" or "page N" gets that endnote / page directly.

Chunks are whole elements, so there are no parent/child windows (the guide's advice for short documents, and the opposite of the
long-report setup in WB-RAG).

## Layout
```
apprag/
  extract.py    spans/blocks, footnote markers vs subscripts
  tables.py     header bars + rules -> rows/columns -> cell sentences
  notes.py      endnote parser
  visuals.py    visual-page detection, Gemini descriptions, cache
  elements.py   one chunk per element; marker resolution in both directions
  retriever.py  hybrid + rerank modes     router.py  "footnote N" / "page N"
  pipeline.py   ingest / search / ask     llm.py     gemini | openai | ollama | extractive
  naive.py      plain-text baseline, used only by `compare`
  evaluate.py   eval + technique comparison
app.py          Streamlit "evidence board": answer, typed evidence cards, cited page preview
```

## Run (Windows PowerShell)
```
pip install -r requirements.txt
copy .env.example .env       # optional: LLM_PROVIDER=gemini + GEMINI_API_KEY (also used for the vision descriptions)
python -m apprag ingest      # 91 elements; vision pages are cached after the first run
python -m apprag ask "How much power does the iMac use in sleep mode at 115V?"
python -m apprag eval        # 24 questions
python -m apprag compare     # plain text vs element-aware, one technique at a time
pytest tests
streamlit run app.py
```
`USE_VISION=0` skips the vision calls; without a Gemini key ingestion still works and reports the pages it could not describe.

## Configuration (`.env`)

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `extractive` | `extractive` (offline), `gemini`, `openai` or `ollama`; `gemini` also enables the vision descriptions |
| `GEMINI_API_KEY` / `GEMINI_MODEL` | – / `gemini-flash-lite-latest` | `gemini-flash-latest` can hang; use the lite model |
| `RERANK_MODE` | `all` | `all`, `prose` (tables keep their slot) or `none` |
| `USE_VISION` | `1` | `0` skips the page-image descriptions |
| `TOP_K` / `RERANK_TOP_K` | 15 / 5 | candidates retrieved / elements sent to the model |

Ingestion makes at most 8 vision calls (one per artwork page); they are cached afterwards.

## Results
`eval/questions.json` has 24 questions **written by me from the report** (none were supplied): 7 table, 1 chart, 1 chart+footnote,
8 footnote, 2 call-out, 2 prose, 1 prose+footnote, 2 the report cannot answer. Expected strings were checked against the text.

| | extractive (no LLM) | Gemini (`gemini-flash-lite-latest`) |
|---|---|---|
| Right element retrieved (22 answerable) | 22/22 | 22/22 |
| Right PDF page among citations | 22/22 | 22/22 |
| Answer contains the fact | 20/22 | 22/22 |
| Refuses the 2 out-of-scope questions | 2/2 | 2/2 |

Retrieval, top 5 elements, same questions (`python -m apprag compare`):

| system | hit@5 | hit@1 | MRR |
|---|---|---|---|
| plain text, 1000-char chunks + rerank | 16/22 | 14/22 | 0.682 |
| elements, no rerank | 22/22 | 20/22 | 0.955 |
| elements + rerank prose only | 22/22 | 21/22 | 0.977 |
| elements + rerank everything (default) | 22/22 | 22/22 | 1.000 |
| elements, no vision descriptions | 22/22 | 20/22 | 0.955 |

By type, plain text → elements: tables 2/7 → 7/7, chart 0/1 → 1/1; footnote, call-out and prose questions are already found by plain text.

## Read these numbers carefully
- 22 questions and a 91-element corpus: differences under ~0.1 MRR are noise. The table and chart gain is large and consistent; the rerank rows are not distinguishable.
- Reranking everything did **not** hurt here, unlike the guide's warning. The likely reason is that tables are verbalised into sentences before the cross-encoder sees them. It would probably hurt on raw table text.
- Vision added nothing to retrieval on these questions: the label-pairing fallback already puts `445 kg` and `99 kg` in one chunk. What only the vision text states is the arithmetic (445 − 99 = 346), which no question asked for.
- The plain-text baseline is generous on footnotes: the endnotes are findable as raw text. The linking pays off in the answer (the caveat arrives with the claim), which retrieval hit@5 does not measure.

## Known limitations
- Table thresholds (header-bar colour, row gap, cell gap) were tuned on this PDF; the idea transfers, the numbers do not.
- The report contradicts itself: the chart's endnote 10 says 512GB for the 346 kg figure, the table says 256GB. Both are indexed as written; nothing reconciles them.
- Page 3's lifecycle diagram is split into one chunk per stage by position, and some call-out text is read in an odd order.
- The extractive generator quotes sentences, so it misses two answers that need a sentence from the middle of a long endnote.
- The Gemini answers depend on the API being available (a free-tier rate limit stopped an earlier WB-RAG run); ingestion caches the vision text so it is only needed once.
