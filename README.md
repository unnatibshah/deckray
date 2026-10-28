# Office RAG

**Structure-aware ingestion of PowerPoint and Excel files for retrieval-augmented generation.**
Most pipelines flatten Office files into a wall of text and lose what made them useful: slide titles, bullet hierarchy, table headers, merged cells, chart data, speaker notes. Office RAG keeps the structure, shows exactly what it extracted, and answers with citations to the slide or row.

**[Try it in your browser →](https://unnatibshah.github.io/office-rag/)** Files are parsed on your device and never uploaded.

## What it extracts

| PowerPoint (`officerag/pptx.py`) | Excel (`officerag/xlsx.py`) |
|---|---|
| Slides in presentation order (not filename order, so slide 10 follows slide 9) | Header row detected under title rows; title rows become the sheet description |
| Title, text with bullet and numbered hierarchy, including bullets inherited from the layout | Merged cells unmerged everywhere, so every row carries its full context |
| Tables with merged cells resolved (only real merges are filled) | Columns keep their positions: a blank header never shifts later columns |
| Charts: type, title and every point of every series | One chunk per row, with structured fields for exact filtering |
| SmartArt as a real tree, built from the diagram's own connections | Dates as ISO dates, cell hyperlinks kept, hidden sheets flagged |
| Speaker notes (without slide-number placeholders), hyperlinks with their link text | |
| Image alt text, plus optional image description by a vision model | |

Everything is read straight from the OOXML parts with the standard library, `openpyxl` and `defusedxml`, so the same package runs in CPython and in the browser (Pyodide).

## How answering works

1. **Plan** (optional model): one call routes the question to slides, rows or both, picks the sheet, and extracts column filters. The plan is validated against the real sheets, columns and values, so a model can't invent a filter; synonyms are mapped to existing values.
2. **Retrieve**: chunks are embedded (all-MiniLM-L6-v2 in the browser, `bge-m3` via Ollama in the CLI). Row questions apply the plan's exact filters first; slide hits pull in their sibling chunks (text, table, chart, notes) so the answer sees the whole slide.
3. **Answer** (optional model): written only from the numbered sources, with citations that link back to the slide or row.

Without a model you still get extraction, the X-ray view and semantic search.

## Bring your own model

In the web app, open **Model** and pick a provider: **Ollama (local)**, OpenAI, Anthropic, Groq or OpenRouter. The key is sent only from your browser to that provider (it can be remembered on your device if you choose). The model list is loaded live from the provider. For Ollama, allow the page's origin:

```bash
OLLAMA_ORIGINS=https://unnatibshah.github.io ollama serve
```

## Command line (local, private)

Defaults to Ollama on your machine: `qwen2.5:14b` for planning and answers, `bge-m3` for embeddings.

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m officerag inspect "samples/Solar Sites Directory.xlsx"
.venv/bin/python -m officerag ask samples/*.pptx samples/*.xlsx -q "Which sites in Spain are offline?"
.venv/bin/python -m officerag ask deck.pptx -q "..." --provider openai --model <model-id> --key "$OPENAI_API_KEY"
```

## Safety

Uploads are limited to 20 MB, `.pptx`/`.xlsx`/`.xlsm`, at most 5,000 zip entries and 200 MB uncompressed (zip-bomb guard). XML is parsed with `defusedxml`. Sheets are capped at 5,000 rows × 60 columns.

## Tests

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python scripts/make_samples.py   # regenerate the fictional sample files
.venv/bin/python -m pytest
```

The tests cover each extraction fix (slide order past 9, inherited bullets, merged cells in tables and sheets, full chart series, SmartArt trees, header detection with an optional column, blank-header alignment), the upload limits, plan validation and context selection.

---

Sample files are fictional. Built by [Unnati Shah](https://unnatibshah.github.io/).
