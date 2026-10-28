"""Bring-your-own-model LLM layer: planning (route + sheet + filters), grounded answers, optional image description.

Transport is injected (`post(url, headers, body) -> dict`), so the same code runs in the CLI (httpx) and in the
browser (Pyodide calling fetch). Every provider is OpenAI-compatible except Anthropic, which gets a small adapter.
"""
from __future__ import annotations

import json
from typing import Any, Awaitable, Callable

Post = Callable[[str, dict, dict], Awaitable[dict]]

PROVIDERS = {
    "ollama": {"label": "Ollama (local)", "base": "http://localhost:11434/v1", "needs_key": False},
    "openai": {"label": "OpenAI", "base": "https://api.openai.com/v1", "needs_key": True},
    "groq": {"label": "Groq", "base": "https://api.groq.com/openai/v1", "needs_key": True},
    "openrouter": {"label": "OpenRouter", "base": "https://openrouter.ai/api/v1", "needs_key": True},
    "anthropic": {"label": "Anthropic", "base": "https://api.anthropic.com/v1", "needs_key": True},
}


def _headers(cfg: dict) -> dict:
    if cfg["provider"] == "anthropic":
        return {"x-api-key": cfg.get("key", ""), "anthropic-version": "2023-06-01",
                "anthropic-dangerous-direct-browser-access": "true", "content-type": "application/json"}
    h = {"content-type": "application/json"}
    if cfg.get("key"):
        h["authorization"] = f"Bearer {cfg['key']}"
    return h


async def chat(cfg: dict, post: Post, messages: list[dict], *, json_mode: bool = False, max_tokens: int = 900) -> str:
    base = (cfg.get("base") or PROVIDERS[cfg["provider"]]["base"]).rstrip("/")
    if cfg["provider"] == "anthropic":
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        body = {"model": cfg["model"], "max_tokens": max_tokens, "temperature": 0.1, "system": system,
                "messages": [m for m in messages if m["role"] != "system"]}
        r = await post(f"{base}/messages", _headers(cfg), body)
        text = "".join(b.get("text", "") for b in r.get("content", []))
        if not text.strip():
            raise RuntimeError(f"{cfg['model']} returned an empty response")
        return text
    body = {"model": cfg["model"], "messages": messages, "max_tokens": max_tokens, "temperature": 0.1}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    r = await post(f"{base}/chat/completions", _headers(cfg), body)
    text = ((r.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    if not text.strip():  # e.g. Ollama out of GPU memory still answers 200 with empty content
        raise RuntimeError(f"{cfg['model']} returned an empty response" + (f" ({r['error']})" if r.get("error") else ""))
    return text


def _json(text: str) -> dict:
    text = text.strip()
    s, e = text.find("{"), text.rfind("}")
    return json.loads(text[s:e + 1]) if s >= 0 and e > s else {}


PLAN_PROMPT = """You plan retrieval over a user's PowerPoint decks and Excel sheets.
Loaded sources (decks with slide titles; sheets with columns and their distinct values):
{sources}

Decide:
1. route: "pptx" if only slides can answer, "xlsx" if only spreadsheet rows can, "both" if both may help, "none" if nothing loaded is relevant.
2. sheet: for spreadsheet questions, the single most relevant sheet name, else null.
3. filters: exact column -> value conditions implied by the question, using ONLY column names and values from the lists above
   (map synonyms and spelling variants to the listed value). Use {{}} when the question isn't a filter over rows.
4. query: the question rewritten as a short search query.
Reply with JSON only: {{"route": "...", "sheet": null, "filters": {{}}, "query": "..."}}"""


def validate_plan(raw: Any, sources: list[dict]) -> dict:
    """Keep only sheets, columns and values that really exist (case-insensitive), so a model can't invent filters."""
    raw = raw if isinstance(raw, dict) else {}
    plan = {"route": raw.get("route") if raw.get("route") in ("pptx", "xlsx", "both", "none") else "both",
            "sheet": None, "filters": {}, "query": str(raw.get("query") or "")[:300]}
    sheets = {sh["sheet"].lower(): sh for s in sources if s["kind"] == "xlsx" for sh in s["sheets"]}
    sh = sheets.get(str(raw.get("sheet") or "").lower())
    if sh:
        plan["sheet"] = sh["sheet"]
        cols = {c.lower(): (c, vals) for c, vals in sh["columns"].items()}
        for col, val in (raw.get("filters") or {}).items() if isinstance(raw.get("filters"), dict) else []:
            hit = cols.get(str(col).lower())
            if hit and (v := next((x for x in hit[1] if x != "…" and x.lower() == str(val).lower()), None)):
                plan["filters"][hit[0]] = v
    return plan


async def plan(cfg: dict, post: Post, question: str, sources: list[dict]) -> dict:
    try:
        raw = _json(await chat(cfg, post, [{"role": "system", "content": PLAN_PROMPT.format(sources=json.dumps(sources)[:12000])},
                                           {"role": "user", "content": question}], json_mode=True, max_tokens=400))
        return {**validate_plan(raw, sources), "source": "llm"}
    except Exception as e:
        return {"route": "both", "sheet": None, "filters": {}, "query": question, "source": "fallback", "error": str(e)[:200]}


ANSWER_PROMPT = """Answer the question using ONLY the numbered sources below, extracted from PowerPoint slides and Excel rows.
- Cite every fact with its source number in square brackets, like [2] or [1][3].
- Slide sources from the same deck can be combined; keep different decks apart.
- If decks and sheets give different answers, present them separately ("From the slides:", "From the spreadsheet:").
- For row questions, list every matching row the sources contain.
- If the sources don't contain the answer, reply exactly: I couldn't find that in the loaded documents.
- Be concise: short paragraphs or bullets. Never invent values.

Sources:
{sources}"""


async def answer(cfg: dict, post: Post, question: str, context: list[dict]) -> str:
    numbered = "\n\n".join(f"[{i}] {c['text'][:2500]}" for i, c in enumerate(context, 1))
    return (await chat(cfg, post, [{"role": "system", "content": ANSWER_PROMPT.format(sources=numbered)},
                                   {"role": "user", "content": question}], max_tokens=900)).strip()


IMAGE_PROMPT = """You are an OCR and visual-understanding step in a document ingestion pipeline. This image comes from a PowerPoint slide.
Transcribe all visible text literally. If it contains a table, give it as rows of "header: value". If it's mainly visual
(diagram, chart, photo, icon), describe what it shows in one or two sentences. Plain text only, no preamble."""


async def describe_image(cfg: dict, post: Post, b64: str, mime: str) -> str:
    if cfg["provider"] == "anthropic":
        content = [{"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}},
                   {"type": "text", "text": IMAGE_PROMPT}]
    else:
        content = [{"type": "text", "text": IMAGE_PROMPT}, {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}]
    return (await chat(cfg, post, [{"role": "user", "content": content}], max_tokens=600)).strip()
