"""Local RAG over PowerPoint and Excel files. Defaults to Ollama on this machine; nothing leaves it.

  python -m officerag ask samples/*.pptx samples/*.xlsx -q "Which sites in Spain are offline?"
  python -m officerag inspect "samples/Solar Sites Directory.xlsx"
  python -m officerag ask docs/*.xlsx -q "..." --provider openai --model <model> --key $OPENAI_API_KEY
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
from pathlib import Path

import httpx

from . import ingest, llm
from .chunks import select_context, sources

CACHE = Path(".officerag-cache.json")


async def _post(url: str, headers: dict, body: dict) -> dict:
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as c:
        r = await c.post(url, headers=headers, json=body)
    if r.status_code >= 400:
        raise RuntimeError(f"{r.status_code} from {url}: {r.text[:200]}")
    return r.json()


async def embed(texts: list[str], model: str, ollama: str) -> list[list[float]]:
    """Ollama embeddings, cached on disk by text hash so re-runs only embed new chunks."""
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    keyed = [f"{model}:{hashlib.sha256(t.encode()).hexdigest()}" for t in texts]
    todo = [t for t, k in zip(texts, keyed) if k not in cache]
    for i in range(0, len(todo), 32):
        batch = todo[i:i + 32]
        vecs = (await _post(f"{ollama}/api/embed", {}, {"model": model, "input": batch}))["embeddings"]
        cache.update({f"{model}:{hashlib.sha256(t.encode()).hexdigest()}": v for t, v in zip(batch, vecs)})
    if todo:
        CACHE.write_text(json.dumps(cache))
    return [cache[k] for k in keyed]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)) or 1)


async def ask(args) -> None:
    docs = [ingest(Path(f).name, Path(f).read_bytes()) for f in args.files]
    chunks = [c for d in docs for c in d["chunks"]]
    cfg = {"provider": args.provider, "model": args.model, "key": args.key, "base": args.base}
    plan = await llm.plan(cfg, _post, args.question, sources({d["name"]: d["extraction"] for d in docs}))
    print(f"plan ({plan['source']}): route={plan['route']} sheet={plan['sheet']} filters={plan['filters']}")
    ollama = (args.base or llm.PROVIDERS["ollama"]["base"]).removesuffix("/v1") if args.provider == "ollama" else "http://localhost:11434"
    vecs = await embed([c["text"] for c in chunks] + [plan.get("query") or args.question], args.embed_model, ollama)
    q = vecs.pop()
    context = select_context(chunks, [cosine(q, v) for v in vecs], plan)
    print(f"\n{len(context)} sources:")
    for i, c in enumerate(context, 1):
        print(f"  [{i}] {c['id']}")
    print("\n" + await llm.answer(cfg, _post, args.question, context))


def inspect(args) -> None:
    for f in args.files:
        d = ingest(Path(f).name, Path(f).read_bytes())
        print(f"== {d['name']}: {len(d['chunks'])} chunks (naive baseline: {len(d['naive'])})")
        for c in d["chunks"]:
            print(f"\n[{c['type']}] {c['id']} (~{c['tokens']} tokens)\n{c['text']}")


def main() -> None:
    p = argparse.ArgumentParser(prog="officerag", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ask", help="answer a question over the files")
    a.add_argument("files", nargs="+")
    a.add_argument("-q", "--question", required=True)
    a.add_argument("--provider", default="ollama", choices=list(llm.PROVIDERS))
    a.add_argument("--model", default="qwen2.5:14b")
    a.add_argument("--key", default="")
    a.add_argument("--base", default="", help="override the provider's API base URL")
    a.add_argument("--embed-model", default="bge-m3", help="Ollama embedding model")
    i = sub.add_parser("inspect", help="print the structure-aware chunks")
    i.add_argument("files", nargs="+")
    args = p.parse_args()
    asyncio.run(ask(args)) if args.cmd == "ask" else inspect(args)
