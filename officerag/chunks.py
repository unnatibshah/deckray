"""Structure-aware chunks (and a naive baseline to compare against), plus context selection for answering."""
from __future__ import annotations

from typing import Any


def tokens(text: str) -> int:
    return max(1, len(text) // 4)  # ponytail: chars/4 estimate; swap in a real tokenizer if budgets get tight


def _chunk(doc: str, cid: str, ctype: str, where: dict, text: str, **extra) -> dict:
    return {"id": cid, "doc": doc, "type": ctype, "where": where, "text": text, "tokens": tokens(text), **extra}


def _tree(nodes: list[dict], depth: int = 0) -> list[str]:
    return [line for n in nodes for line in ["  " * depth + "- " + n["text"], *_tree(n["children"], depth + 1)]]


def pptx_chunks(doc: str, ex: dict) -> list[dict]:
    out = []
    for s in ex["slides"]:
        n, base = s["index"], f"{doc} › slide {s['index']}"
        head = f"[{base}{' · ' + s['title'] if s['title'] else ''}]"
        where = {"slide": n}
        if s["lines"] or s["title"]:
            out.append(_chunk(doc, f"{doc}#s{n}-text", "text", where, "\n".join([head, *s["lines"]])))
        for i, t in enumerate(s["tables"], 1):
            rows = [" | ".join(f"{h}: {v}" for h, v in r.items() if v) for r in t["rows"]]
            out.append(_chunk(doc, f"{doc}#s{n}-table{i}", "table", where, "\n".join([f"{head} table {i}", *rows])))
        for i, c in enumerate(s["charts"], 1):
            pts = [f"{ser['name']} · {p['label']}: {p['value']:g}" for ser in c["series"] for p in ser["points"]]
            out.append(_chunk(doc, f"{doc}#s{n}-chart{i}", "chart", where, "\n".join([f"{head} {c['kind']} chart: {c['title']}", *pts])))
        for i, tree in enumerate(s["smartart"], 1):
            out.append(_chunk(doc, f"{doc}#s{n}-smartart{i}", "smartart", where, "\n".join([f"{head} diagram", *_tree(tree)])))
        if s["notes"]:
            out.append(_chunk(doc, f"{doc}#s{n}-notes", "notes", where, f"{head} speaker notes\n{s['notes']}"))
        if s["links"]:
            out.append(_chunk(doc, f"{doc}#s{n}-links", "links", where,
                              "\n".join([f"{head} links", *(f"{l['text'] or 'link'}: {l['url']}" for l in s["links"])])))
        alts = [i.get("description") or i["alt"] for i in s["images"] if i.get("description") or i["alt"]]
        if alts:
            out.append(_chunk(doc, f"{doc}#s{n}-images", "images", where, "\n".join([f"{head} images", *alts])))
    return out


def xlsx_chunks(doc: str, ex: dict) -> list[dict]:
    out = []
    for sh in ex["sheets"]:
        ctx = f"[{doc} › {sh['name']}" + (f" · {sh['description']}" if sh["description"] else "") + "]"
        for r in sh["rows"]:
            text = ctx + " " + " | ".join(f"{k}: {v}" for k, v in r["fields"].items())
            if r["links"]:
                text += " | links: " + ", ".join(f"{k} → {u}" for k, u in r["links"].items())
            out.append(_chunk(doc, f"{doc}#{sh['name']}!{r['row']}", "row", {"sheet": sh["name"], "row": r["row"]},
                              text, fields=r["fields"]))
    return out


def chunk(doc: str, ex: dict) -> list[dict]:
    return pptx_chunks(doc, ex) if ex["kind"] == "pptx" else xlsx_chunks(doc, ex)


def naive_chunks(doc: str, ex: dict, size: int = 700) -> list[dict]:
    """The baseline most pipelines start with: all text flattened, cut into fixed windows. No headers, no structure."""
    if ex["kind"] == "pptx":
        parts = [w for s in ex["slides"] for w in [s["title"], *(l.strip("• ").strip() for l in s["lines"]),
                 *(v for t in s["tables"] for r in t["rows"] for v in r.values()), s["notes"]] if w]
    else:
        parts = [sh["flat"] for sh in ex["sheets"]]
    flat = " ".join(parts)
    return [_chunk(doc, f"{doc}#naive{i // size + 1}", "naive", {}, flat[i:i + size]) for i in range(0, len(flat), size)]


def select_context(chunks: list[dict], scores: list[float], plan: dict, k: int = 8) -> list[dict]:
    """Top-k by similarity after the plan's structured filters; slide hits pull in their sibling chunks."""
    route, filters, sheet = plan.get("route", "both"), plan.get("filters") or {}, plan.get("sheet")
    def allowed(c: dict) -> bool:
        if route == "pptx" and c["type"] == "row" or route == "xlsx" and c["type"] != "row":
            return False
        if c["type"] == "row":
            if sheet and c["where"]["sheet"] != sheet:
                return False
            return all(str(c["fields"].get(col, "")).lower() == str(v).lower() for col, v in filters.items())
        return True
    ranked = sorted((i for i, c in enumerate(chunks) if allowed(c) and c["type"] != "naive"), key=lambda i: -scores[i])
    picked, seen = [], set()
    for i in ranked[: (40 if filters else k)]:  # filtered queries return every matching row (up to 40)
        c = chunks[i]
        group = [x for x in chunks if x["doc"] == c["doc"] and x["where"].get("slide") == c["where"].get("slide")] \
            if "slide" in c["where"] else [c]
        for x in group:
            if x["id"] not in seen:
                seen.add(x["id"])
                picked.append(x)
    return picked[:40]


def sources(docs: dict[str, dict]) -> list[dict]:
    """Compact description of what's loaded, for the planner: sheets with columns and their distinct values."""
    out = []
    for name, ex in docs.items():
        if ex["kind"] == "pptx":
            out.append({"doc": name, "kind": "pptx", "slides": [s["title"] for s in ex["slides"]][:40]})
        else:
            sheets = []
            for sh in ex["sheets"]:
                cols = {}
                for h in sh["headers"]:
                    vals = list(dict.fromkeys(r["fields"][h] for r in sh["rows"] if h in r["fields"]))
                    cols[h] = vals[:25] if len(vals) <= 25 else vals[:12] + ["…"]
                sheets.append({"sheet": sh["name"], "description": sh["description"], "columns": cols})
            out.append({"doc": name, "kind": "xlsx", "sheets": sheets})
    return out
