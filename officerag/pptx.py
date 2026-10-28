"""PowerPoint extraction straight from the OOXML parts (no python-pptx needed, so it also runs in the browser).

Per slide: title, text with bullet/numbered hierarchy, tables (merged cells resolved), charts (every series),
SmartArt (real parent/child tree), speaker notes, hyperlinks with their link text, and image alt text.
"""
from __future__ import annotations

import posixpath
import re
import zipfile
from typing import Any

from defusedxml import ElementTree as ET

NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "c": "http://schemas.openxmlformats.org/drawingml/2006/chart",
    "dgm": "http://schemas.openxmlformats.org/drawingml/2006/diagram",
}
R_ID = f"{{{NS['r']}}}id"
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/"
TITLE_PH = {"title", "ctrTitle"}
SKIP_PH = {"sldNum", "dt", "ftr", "hdr", "sldImg"}
NUMBERED = re.compile(r"^(\d+[.)]|[a-z][.)]|[ivxlcdm]+\.)\s", re.I)


def _xml(z: zipfile.ZipFile, part: str):
    with z.open(part) as f:
        return ET.parse(f).getroot()


def _rels(z: zipfile.ZipFile, part: str) -> dict[str, dict[str, str]]:
    """rId -> {type, target}; internal targets resolved to absolute part names."""
    path = posixpath.join(posixpath.dirname(part), "_rels", posixpath.basename(part) + ".rels")
    if path not in z.namelist():
        return {}
    out = {}
    for rel in _xml(z, path):
        target = rel.get("Target", "")
        if rel.get("TargetMode") != "External":
            target = posixpath.normpath(posixpath.join(posixpath.dirname(part), target)).lstrip("/")
        out[rel.get("Id")] = {"type": rel.get("Type", "").removeprefix(REL), "target": target}
    return out


def _slide_parts(z: zipfile.ZipFile) -> list[str]:
    """Slides in presentation order (sldIdLst), not filename order: slide10 must not come before slide2."""
    rels = _rels(z, "ppt/presentation.xml")
    ids = _xml(z, "ppt/presentation.xml").findall("p:sldIdLst/p:sldId", NS)
    parts = [rels[s.get(R_ID)]["target"] for s in ids if s.get(R_ID) in rels]
    if parts:
        return parts
    names = [n for n in z.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)]
    return sorted(names, key=lambda n: int(re.search(r"\d+", n.rsplit("/", 1)[1]).group()))


def _placeholder(sp) -> str | None:
    ph = sp.find("p:nvSpPr/p:nvPr/p:ph", NS)
    return None if ph is None else ph.get("type", "body")


def _para_text(p, skip: set) -> str:
    return "".join(t.text or "" for t in p.iter(f"{{{NS['a']}}}t") if t not in skip).strip()


def _text_lines(root, table_runs: set) -> tuple[str, list[str]]:
    """Title plus body lines; bullets and numbering rendered with indentation by outline level."""
    title, lines, counters = "", [], {}
    for sp in root.iter(f"{{{NS['p']}}}sp"):
        ph = _placeholder(sp)
        if ph in SKIP_PH:
            continue
        body = sp.find("p:txBody", NS)
        if body is None:
            continue
        is_body = ph in ("body", "obj")  # subtitles and other placeholders don't inherit bullets
        for p in body.findall("a:p", NS):
            text = _para_text(p, table_runs)
            if not text:
                continue
            if ph in TITLE_PH:
                title = f"{title} {text}".strip()
                continue
            ppr = p.find("a:pPr", NS)
            lvl = int(ppr.get("lvl", 0)) if ppr is not None else 0
            no_bullet = ppr is not None and ppr.find("a:buNone", NS) is not None
            numbered = ppr is not None and ppr.find("a:buAutoNum", NS) is not None
            bulleted = ppr is not None and ppr.find("a:buChar", NS) is not None
            if not (numbered or bulleted or no_bullet) and NUMBERED.match(text):
                numbered = True
            elif not (numbered or bulleted or no_bullet) and is_body:
                bulleted = True  # body placeholders inherit bullets from the layout
            for deeper in [k for k in counters if k > lvl]:
                del counters[deeper]
            if numbered:
                counters[lvl] = counters.get(lvl, 0) + 1
                marker = "" if NUMBERED.match(text) else f"{counters[lvl]}. "
                lines.append("  " * lvl + marker + text)
            else:
                counters.pop(lvl, None)
                lines.append("  " * lvl + ("• " if bulleted else "") + text)
    return title, lines


def _tables(root) -> list[dict]:
    """Grid with gridSpan/rowSpan resolved: merged areas repeat their value, genuinely empty cells stay empty."""
    tables = []
    for tbl in root.iter(f"{{{NS['a']}}}tbl"):
        grid: list[list[str]] = []
        for tr in tbl.findall("a:tr", NS):
            row = []
            for tc in tr.findall("a:tc", NS):
                text = " ".join(t.text for t in tc.iter(f"{{{NS['a']}}}t") if t.text).strip()
                if tc.get("hMerge") == "1" and row:
                    text = row[-1]
                elif tc.get("vMerge") == "1" and grid and len(grid[-1]) > len(row):
                    text = grid[-1][len(row)]
                row.append(text)
            grid.append(row)
        if not grid:
            continue
        width = max(map(len, grid))
        grid = [r + [""] * (width - len(r)) for r in grid]
        h = max(range(min(3, len(grid))), key=lambda i: sum(1 for v in grid[i] if v))
        headers = [v or f"Column {i + 1}" for i, v in enumerate(grid[h])]
        rows = [dict(zip(headers, r)) for r in grid[h + 1:] if any(r)]
        if rows:
            tables.append({"headers": headers, "rows": rows})
    return tables


def _chart(z, part: str) -> dict:
    root = _xml(z, part)
    title = " ".join(t.text for t in root.findall(".//c:title//a:t", NS) if t.text).strip() or "Untitled chart"
    series = []
    for ser in root.iter(f"{{{NS['c']}}}ser"):
        name = " ".join(v.text for v in ser.findall("c:tx//c:v", NS) if v.text) or f"Series {len(series) + 1}"
        cats = [v.text for v in ser.findall("c:cat//c:pt/c:v", NS)]
        vals = [v.text for v in ser.findall("c:val//c:pt/c:v", NS)]
        series.append({"name": name, "points": [{"label": c, "value": float(v)} for c, v in zip(cats, vals)
                                                 if v not in (None, "")]})
    plot = root.find(".//c:plotArea", NS)
    kind = next((el.tag.split("}")[1].removesuffix("Chart") for el in (plot if plot is not None else [])
                 if el.tag.endswith("Chart")), "chart")
    return {"title": title, "kind": kind, "series": series}


def _smartart(z, part: str) -> list[dict]:
    """Tree from the diagram's own connections (parOf), not from guessing pairs of text runs."""
    root = _xml(z, part)
    text = {}
    for pt in root.iter(f"{{{NS['dgm']}}}pt"):
        if pt.get("type", "node") == "node":
            t = " ".join(x.text for x in pt.iter(f"{{{NS['a']}}}t") if x.text).strip()
            if t:
                text[pt.get("modelId")] = t
    children: dict[str, list[str]] = {}
    for cx in root.iter(f"{{{NS['dgm']}}}cxn"):
        if cx.get("type", "parOf") == "parOf":
            children.setdefault(cx.get("srcId"), []).append(cx.get("destId"))

    def build(mid: str) -> list[dict]:
        out = []
        for c in children.get(mid, []):
            kids = build(c)
            if c in text:
                out.append({"text": text[c], "children": kids})
            else:
                out.extend(kids)  # pass-through nodes (doc root, transitions)
        return out
    roots = [m for m in children if not any(m in v for v in children.values())]
    return [n for r in roots for n in build(r)]


def _notes(z, part: str) -> str:
    lines = []
    for sp in _xml(z, part).iter(f"{{{NS['p']}}}sp"):
        if _placeholder(sp) in SKIP_PH:
            continue
        for p in sp.iter(f"{{{NS['a']}}}p"):
            if t := _para_text(p, set()):
                lines.append(t)
    return "\n".join(lines)


def extract(data_or_zip) -> dict[str, Any]:
    z = data_or_zip if isinstance(data_or_zip, zipfile.ZipFile) else zipfile.ZipFile(data_or_zip)
    slides = []
    for n, part in enumerate(_slide_parts(z), 1):
        root, rels = _xml(z, part), _rels(z, part)
        table_runs = {t for tbl in root.iter(f"{{{NS['a']}}}tbl") for t in tbl.iter(f"{{{NS['a']}}}t")}
        title, lines = _text_lines(root, table_runs)
        links, seen = [], set()
        for run in root.iter(f"{{{NS['a']}}}r"):
            h = run.find("a:rPr/a:hlinkClick", NS)
            rel = rels.get(h.get(R_ID)) if h is not None else None
            if rel and rel["type"] == "hyperlink" and (key := (run.findtext("a:t", "", NS), rel["target"])) not in seen:
                seen.add(key)
                links.append({"text": key[0].strip(), "url": key[1]})
        images = []
        for pic in root.iter(f"{{{NS['p']}}}pic"):
            c, blip = pic.find("p:nvPicPr/p:cNvPr", NS), pic.find(".//a:blip", NS)
            part = rels.get(blip.get(f"{{{NS['r']}}}embed"), {}).get("target") if blip is not None else None
            images.append({"alt": (c.get("descr") or "").strip() if c is not None else "",
                           "name": c.get("name", "") if c is not None else "", "part": part})
        charts, smartart, notes = [], [], ""
        for r in rels.values():
            if r["target"] not in z.namelist():
                continue
            if r["type"] == "chart":
                charts.append(_chart(z, r["target"]))
            elif r["type"] == "diagramData":
                smartart.append(_smartart(z, r["target"]))
            elif r["type"] == "notesSlide":
                notes = _notes(z, r["target"])
        slides.append({"index": n, "title": title, "lines": lines, "tables": _tables(root), "charts": charts,
                       "smartart": [s for s in smartart if s], "notes": notes, "links": links, "images": images,
                       "hidden": root.get("show") == "0"})
    return {"kind": "pptx", "slides": slides}
