"""Office RAG: structure-aware ingestion of PowerPoint and Excel files for retrieval-augmented generation."""
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


def ingest(name: str, data: bytes) -> dict:
    """Validate an uploaded file (size, type, zip-bomb limits), extract it and chunk it."""
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
    if len(infos) > MAX_ENTRIES or sum(i.file_size for i in infos) > MAX_UNZIPPED:
        raise IngestError(f"{name} expands to too much data to process safely.")
    try:
        ex = pptx.extract(z) if ext == "pptx" else xlsx.extract(io.BytesIO(data))
    except Exception as e:
        raise IngestError(f"Couldn't read {name}: {e}") from None
    return {"name": name, "extraction": ex, "chunks": chunk(name, ex), "naive": naive_chunks(name, ex)}
