"""
Auxiliary byte scanner: where do known bytes of an item exist in a store?

This is an independent cross-check for the manifest-based accuracy evaluation.
It does not use the parser: it reads every file of a persisted store as raw bytes
and searches for an item's known vector bytes, chunk text and id. Hits are
reported by location:

* ``sqlite_allocated``: ``chroma.sqlite3`` outside its unallocated space, i.e.
  live rows the parser reads (WAL records, metadata segment);
* ``sqlite_unallocated``: free pages, freeblocks and in-page gaps;
* ``hnsw``: the vector segment binaries (``data_level0.bin`` and the rest);
* ``id_map``: ``index_metadata.pickle`` (the id mapping).

A hit means the bytes exist; it does not say whether a reader can attribute them
(e.g. an Orphan Vector has vector bytes in ``hnsw`` but no id in ``id_map``).
Like the SQLite remnant check, it reads raw bytes and never opens a file through
SQLite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .sqlite_unallocated import KnownTexts, _text_needles, _windows, read_unallocated_space

_VECTOR_WINDOW = 128
LOCATIONS = ("sqlite_allocated", "sqlite_unallocated", "hnsw", "id_map")


@dataclass
class StoreBytes:
    """Raw bytes of one store, grouped by location."""

    sqlite_file: bytes
    sqlite_unallocated: bytes
    hnsw: bytes
    id_map: bytes


@dataclass
class ItemHits:
    vector: set[str] = field(default_factory=set)
    text: set[str] = field(default_factory=set)
    id: set[str] = field(default_factory=set)


def load_store(chroma_db: Path) -> StoreBytes:
    """Read every file of a persisted Chroma directory."""
    chroma_db = Path(chroma_db)
    sqlite_path = chroma_db / "chroma.sqlite3"
    hnsw, id_map = [], []
    for segment in sorted(p for p in chroma_db.iterdir() if p.is_dir()):
        for file in sorted(segment.iterdir()):
            (id_map if file.name == "index_metadata.pickle" else hnsw).append(file.read_bytes())
    separator = b"\x00\xff\x00\xfe\x00\xfd\x00\xfc"
    return StoreBytes(
        sqlite_file=sqlite_path.read_bytes() if sqlite_path.exists() else b"",
        sqlite_unallocated=read_unallocated_space(sqlite_path).blob if sqlite_path.exists() else b"",
        hnsw=separator.join(hnsw),
        id_map=separator.join(id_map),
    )


def scan_item(store: StoreBytes, doc_id: str, vector: bytes | None, text: str | None,
              known: KnownTexts | None = None) -> ItemHits:
    """Locations holding the item's vector, text and id.

    ``known`` (every text of the run) drops text windows shared with another
    item. A needle counts as allocated when it occurs more often in the whole
    SQLite file than in its unallocated space.
    """
    hits = ItemHits()
    needles = {
        "vector": _windows(vector, _VECTOR_WINDOW) if vector else [],
        "text": _text_needles(text, known) if text else [],
        "id": [doc_id.encode("utf-8")] if doc_id else [],
    }
    for kind, kind_needles in needles.items():
        found: set[str] = getattr(hits, kind)
        for needle in kind_needles:
            in_free = store.sqlite_unallocated.count(needle)
            if store.sqlite_file.count(needle) > in_free:
                found.add("sqlite_allocated")
            if in_free:
                found.add("sqlite_unallocated")
            if needle in store.hnsw:
                found.add("hnsw")
            if needle in store.id_map:
                found.add("id_map")
    return hits
