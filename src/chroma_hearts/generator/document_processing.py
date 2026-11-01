"""
Document processing utilities for loading and chunking PDF files.
"""

from pathlib import Path
from typing import List, Tuple

import fitz  # PyMuPDF

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 300


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from all pages in a PDF file."""
    text_parts: List[str] = []
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            block = page.get_text()
            if block:
                text_parts.append(block)
        doc.close()
    except Exception as exc:
        print(f"[WARN] Failed to read PDF '{pdf_path}': {exc}")
        return ""
    return "\n".join(text_parts).strip()


def split_text_into_chunks(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> List[str]:
    """Split text into fixed-size chunks with overlap."""
    if not text:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size.")

    chunks: List[str] = []
    step = chunk_size - overlap
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start += step
    return chunks


def collect_pdf_files(input_dir: Path) -> List[Path]:
    """Collect all PDF files from the input directory."""
    resolved = input_dir.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Input directory does not exist: {resolved}")
    if not resolved.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {resolved}")
    return sorted(resolved.glob("*.pdf"))


def load_documents_from_dir(input_dir: Path) -> Tuple[List[str], List[str], List[dict]]:
    """Build ids/documents/metadatas from all PDF files in a directory."""
    pdf_files = collect_pdf_files(input_dir)

    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[dict] = []

    for pdf_path in pdf_files:
        text = extract_text_from_pdf(pdf_path)
        chunks = split_text_into_chunks(text)
        for idx, chunk in enumerate(chunks):
            ids.append(f"{pdf_path.stem}_chunk_{idx:05d}")
            documents.append(chunk)
            metadatas.append(
                {
                    "source": pdf_path.name,
                    "path": str(pdf_path),
                    "chunk_index": idx,
                    "chunk_size": len(chunk),
                }
            )

    return ids, documents, metadatas
