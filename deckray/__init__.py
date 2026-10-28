"""Deckray: structure-aware ingestion of PowerPoint and Excel files for retrieval-augmented generation."""
from __future__ import annotations

import io
import zipfile

from . import pptx, xlsx
from .chunks import chunk, naive_chunks

MAX_BYTES = 20 * 1024 * 1024
MAX_UNZIPPED = 200 * 1024 * 1024
MAX_ENTRIES = 5000


class IngestError(ValueError):
    pass


def validate(name: str, data: bytes) -> tuple[zipfile.ZipFile, dict]:
    """Size, type and zip-bomb checks. Returns the open archive and what was checked."""
    if len(data) > MAX_BYTES:
        raise IngestError(f"{name} is over {MAX_BYTES // 2**20} MB.")
    ext = name.lower().rsplit(".", 1)[-1]
    if ext not in ("pptx", "xlsx", "xlsm"):
        raise IngestError(f"{name}: only .pptx, .xlsx and .xlsm files are supported.")
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise IngestError(f"{name} isn't a valid Office file (old .ppt/.xls formats aren't supported).") from None
    infos = z.infolist()
    unzipped = sum(i.file_size for i in infos)
    if len(infos) > MAX_ENTRIES or unzipped > MAX_UNZIPPED:
        raise IngestError(f"{name} expands to too much data to process safely.")
    return z, {"kind": "pptx" if ext == "pptx" else "xlsx", "bytes": len(data), "entries": len(infos), "unzipped": unzipped,
               "limits": {"bytes": MAX_BYTES, "entries": MAX_ENTRIES, "unzipped": MAX_UNZIPPED}}


def extract(name: str, data: bytes, z: zipfile.ZipFile) -> dict:
    try:
        return pptx.extract(z) if name.lower().endswith(".pptx") else xlsx.extract(io.BytesIO(data))
    except Exception as e:
        raise IngestError(f"Couldn't read {name}: {e}") from None


def ingest(name: str, data: bytes) -> dict:
    """Validate an uploaded file, extract it and chunk it."""
    z, info = validate(name, data)
    ex = extract(name, data, z)
    return {"name": name, "info": info, "extraction": ex, "chunks": chunk(name, ex), "naive": naive_chunks(name, ex)}
