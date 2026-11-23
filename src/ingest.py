"""Ingestion: load rule documents, chunk them, tag each chunk with its category.

Chunking is fixed-size (600 chars) with 100 chars of overlap, breaking at paragraph
boundaries where possible so no sentence is cut in half. The category comes from the
source filename: aml -> AML, pep -> PEP, kyc -> KYC, fatf -> FATF, anything else ->
GENERAL.
"""

from pathlib import Path

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, DOCS_DIR


def category_from_filename(filename: str) -> str:
    name = filename.lower()
    if "aml" in name:
        return "AML"
    if "pep" in name:
        return "PEP"
    if "kyc" in name:
        return "KYC"
    if "fatf" in name:
        return "FATF"
    return "GENERAL"


def chunk_text(text: str) -> list[str]:
    """Split text into <=CHUNK_SIZE chunks with CHUNK_OVERLAP overlap.

    Before cutting at exactly CHUNK_SIZE, look back for the nearest paragraph
    break (``\\n\\n``) and cut there instead.
    """
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        if end < len(text):
            newline_pos = text.rfind("\n\n", start, end)
            if newline_pos > start:
                end = newline_pos
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - CHUNK_OVERLAP
    return chunks


def load_documents(docs_dir: Path = DOCS_DIR) -> list[dict]:
    """Return one record per chunk: text, source filename, category, chunk index."""
    records: list[dict] = []
    for path in sorted(docs_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        category = category_from_filename(path.name)
        for i, chunk in enumerate(chunk_text(text)):
            records.append(
                {
                    "text": chunk,
                    "source": path.name,
                    "rule_category": category,
                    "chunk_index": i,
                }
            )
    return records
