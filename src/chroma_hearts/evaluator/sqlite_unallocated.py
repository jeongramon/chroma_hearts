"""
Knowledge-based audit of SQLite unallocated space in ``chroma.sqlite3``.

This is not carving. The evaluator already knows every deleted item (its id,
chunk text and, under hash embeddings, its exact vector bytes), so this module
only answers: *do those known bytes still exist somewhere SQLite no longer
references?* That makes the result a lower bound that needs no record parsing.

Unallocated space covered here:

* free pages on the freelist (whole leaf pages, and the unused tail of trunk pages);
* freeblocks inside in-use b-tree pages;
* the gap between the cell pointer array and the cell content area of in-use
  b-tree pages.

Not covered: the unused tail of overflow pages, the ``-wal``/``-journal`` files
and the filesystem around the database file.

The file is read as raw bytes and never opened through SQLite, so reading it
cannot checkpoint or otherwise modify the store.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

_BTREE_PAGE_TYPES = {2, 5, 10, 13}
_INTERIOR_PAGE_TYPES = {2, 5}
# Placed between regions so a match cannot span two unrelated regions.
_REGION_SEPARATOR = b"\x00\xff\x00\xfe\x00\xfd\x00\xfc"
_VECTOR_WINDOW = 128
_TEXT_WINDOW = 64

# Remnant classes, most informative first. An item outside the three residue
# forms is assigned the first class whose bytes are present.
REMNANT_VECTOR_TEXT = "vector_text"
REMNANT_VECTOR = "vector"
REMNANT_TEXT = "text"
REMNANT_ID_ONLY = "id_only"
REMNANT_NONE = "none"


@dataclass
class UnallocatedSpace:
    """Unallocated bytes of one SQLite file, concatenated with separators."""

    page_size: int
    page_count: int
    free_pages: int
    freeblock_bytes: int
    gap_bytes: int
    nonzero_bytes: int
    blob: bytes
    wal_file_present: bool


def _u16(data: bytes, offset: int) -> int:
    return struct.unpack_from(">H", data, offset)[0]


def _u32(data: bytes, offset: int) -> int:
    return struct.unpack_from(">I", data, offset)[0]


def _freelist(data: bytes, page_size: int, page_count: int) -> tuple[set[int], list[bytes]]:
    """Return (all freelist page numbers, unallocated byte regions they hold)."""
    pages: set[int] = set()
    regions: list[bytes] = []
    trunk = _u32(data, 32)
    while trunk and trunk not in pages and trunk <= page_count:
        pages.add(trunk)
        base = (trunk - 1) * page_size
        next_trunk = _u32(data, base)
        leaf_count = min(_u32(data, base + 4), (page_size - 8) // 4)
        leaves = struct.unpack_from(f">{leaf_count}I", data, base + 8)
        # A trunk page only uses its header and leaf pointers; the rest is stale.
        regions.append(data[base + 8 + 4 * leaf_count: base + page_size])
        for leaf in leaves:
            if 0 < leaf <= page_count and leaf not in pages:
                pages.add(leaf)
                regions.append(data[(leaf - 1) * page_size: leaf * page_size])
        trunk = next_trunk
    return pages, regions


def _pointer_map_pages(data: bytes, usable_size: int, page_count: int) -> set[int]:
    """Pointer-map pages exist only in auto-vacuum databases and must be skipped."""
    if _u32(data, 52) == 0:
        return set()
    stride = usable_size // 5 + 1
    return set(range(2, page_count + 1, stride))


def _btree_page_regions(page: bytes, header_offset: int) -> tuple[list[bytes], int, int]:
    """Return (regions, freeblock bytes, gap bytes) of one in-use b-tree page."""
    page_type = page[header_offset]
    header_size = 12 if page_type in _INTERIOR_PAGE_TYPES else 8
    first_freeblock = _u16(page, header_offset + 1)
    cell_count = _u16(page, header_offset + 3)
    content_start = _u16(page, header_offset + 5) or 65536
    regions: list[bytes] = []

    pointer_end = header_offset + header_size + 2 * cell_count
    gap = page[pointer_end:min(content_start, len(page))] if content_start > pointer_end else b""
    if gap:
        regions.append(gap)

    freeblock_bytes = 0
    offset, seen = first_freeblock, set()
    while offset and offset not in seen and offset + 4 <= len(page):
        seen.add(offset)
        next_offset, size = _u16(page, offset), _u16(page, offset + 2)
        if size < 4 or offset + size > len(page):
            break
        # The first 4 bytes are the freeblock header that overwrote the cell start.
        regions.append(page[offset + 4: offset + size])
        freeblock_bytes += size - 4
        offset = next_offset
    return regions, freeblock_bytes, len(gap)


def read_unallocated_space(sqlite_path: Path) -> UnallocatedSpace:
    """Collect the unallocated byte regions of a SQLite database file."""
    data = Path(sqlite_path).read_bytes()
    if len(data) < 100 or not data.startswith(b"SQLite format 3\x00"):
        raise ValueError(f"Not a SQLite database file: {sqlite_path}")
    page_size = _u16(data, 16)
    page_size = 65536 if page_size == 1 else page_size
    usable_size = page_size - data[20]
    page_count = len(data) // page_size

    free_pages, regions = _freelist(data, page_size, page_count)
    skip = free_pages | _pointer_map_pages(data, usable_size, page_count)
    freeblock_total = gap_total = 0
    for page_no in range(1, page_count + 1):
        if page_no in skip:
            continue
        page = data[(page_no - 1) * page_size: page_no * page_size]
        header_offset = 100 if page_no == 1 else 0
        if page[header_offset] not in _BTREE_PAGE_TYPES:
            continue  # overflow page (or lock-byte page); not covered
        page_regions, freeblock_bytes, gap_bytes = _btree_page_regions(page, header_offset)
        regions.extend(page_regions)
        freeblock_total += freeblock_bytes
        gap_total += gap_bytes

    kept = [region for region in regions if region.strip(b"\x00")]
    blob = _REGION_SEPARATOR.join(kept)
    wal_present = Path(f"{sqlite_path}-wal").exists()
    return UnallocatedSpace(
        page_size=page_size,
        page_count=page_count,
        free_pages=len(free_pages),
        freeblock_bytes=freeblock_total,
        gap_bytes=gap_total,
        nonzero_bytes=sum(len(region) - region.count(0) for region in kept),
        blob=blob,
        wal_file_present=wal_present,
    )


def _windows(data: bytes, size: int) -> list[bytes]:
    """Start, middle and end windows, so a partly overwritten record still matches."""
    if len(data) <= size:
        return [data] if data else []
    middle = (len(data) - size) // 2
    return [data[:size], data[middle:middle + size], data[-size:]]


class KnownTexts:
    """Every chunk text a run wrote, to keep only windows that identify one text.

    Chunks overlap their neighbours, and a short final chunk can lie entirely
    inside its predecessor's overlap. A window that also occurs in another known
    text cannot be attributed, so it is not used.
    """

    def __init__(self, texts) -> None:
        self._joined = "\x00".join(set(t for t in texts if t))

    def distinctive(self, window: str, text: str) -> bool:
        return self._joined.count(window) <= text.count(window)


def _text_needles(text: str, known: KnownTexts | None = None) -> list[bytes]:
    """Up to three text windows, raw (metadata segment) and JSON-escaped (WAL).

    Without ``known``, windows sit at the quarter points, away from both ends,
    where neighbouring chunks overlap. With ``known``, candidate windows are
    taken across the text and only those that occur in no other known text are
    kept.
    """
    if len(text) <= 4 * _TEXT_WINDOW:
        starts = [max(0, (len(text) - _TEXT_WINDOW) // 2)]
    else:
        starts = [len(text) * q // 4 - _TEXT_WINDOW // 2 for q in (1, 2, 3)]
    if known is not None:
        step = max(1, _TEXT_WINDOW // 2)
        candidates = starts + [s for s in range(0, max(1, len(text) - _TEXT_WINDOW + 1), step) if s not in starts]
        starts = [s for s in candidates if known.distinctive(text[s:s + _TEXT_WINDOW], text)][:3]
    needles: list[bytes] = []
    for start in sorted(starts):
        window = text[start:start + _TEXT_WINDOW]
        if not window.strip():
            continue
        for encoded in (
            window.encode("utf-8"),
            json.dumps(window, ensure_ascii=False)[1:-1].encode("utf-8"),
            json.dumps(window, ensure_ascii=True)[1:-1].encode("utf-8"),
        ):
            if encoded not in needles:
                needles.append(encoded)
    return needles


def probe_item(
    space: UnallocatedSpace,
    doc_id: str,
    vector_blob: bytes | None,
    text: str | None,
    known: KnownTexts | None = None,
) -> dict[str, bool]:
    """Report which known bytes of one item are present in unallocated space."""
    blob = space.blob
    if not blob:
        return {"id": False, "vector": False, "text": False}
    return {
        "id": bool(doc_id) and doc_id.encode("utf-8") in blob,
        "vector": bool(vector_blob)
        and any(window in blob for window in _windows(vector_blob, _VECTOR_WINDOW)),
        "text": bool(text) and any(needle in blob for needle in _text_needles(text, known)),
    }


def remnant_class(found: dict[str, bool]) -> str:
    """Collapse a probe result into one remnant class (most informative first)."""
    if found["vector"] and found["text"]:
        return REMNANT_VECTOR_TEXT
    if found["vector"]:
        return REMNANT_VECTOR
    if found["text"]:
        return REMNANT_TEXT
    if found["id"]:
        return REMNANT_ID_ONLY
    return REMNANT_NONE
