"""Read-only inspection of surviving SQLite records in unused regions.

Schema-guided probe, not a production recovery parser. Candidate discovery does
not use deleted IDs, document contents, expected lengths, or target vectors.
Reference: https://www.sqlite.org/fileformat2.html#record_format
"""
from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass


@dataclass(frozen=True)
class Region:
    start: int
    end: int
    kind: str
    page: int


def varint(data: bytes, pos: int, limit: int) -> tuple[int, int]:
    value = 0
    for n in range(9):
        if pos >= limit:
            raise ValueError('truncated varint')
        b = data[pos]
        pos += 1
        if n == 8:
            return (value << 8) | b, pos
        value = (value << 7) | (b & 127)
        if b < 128:
            return value, pos
    raise ValueError('invalid varint')


def serial_size(code: int) -> int:
    if code < 0 or code in (10, 11):
        raise ValueError('invalid serial type')
    if code < 12:
        return (0, 1, 2, 3, 4, 6, 8, 8, 0, 0)[code]
    return (code - 12) // 2


def header(data: bytes, pos: int, limit: int) -> tuple[int, list[int]]:
    size, cursor = varint(data, pos, limit)
    if not 3 <= size <= 32 or pos + size > limit:
        raise ValueError('unsupported header size')
    types = []
    while cursor < pos + size:
        code, cursor = varint(data, cursor, pos + size)
        serial_size(code)
        types.append(code)
    if cursor != pos + size:
        raise ValueError('header boundary')
    return size, types


def unused_regions(data: bytes) -> tuple[int, int, list[Region]]:
    assert data.startswith(b'SQLite format 3\x00')
    page_size = int.from_bytes(data[16:18], 'big') or 65536
    if page_size == 1:
        page_size = 65536
    usable = page_size - data[20]
    assert int.from_bytes(data[56:60], 'big') == 1, 'UTF-8 database required'
    assert int.from_bytes(data[52:56], 'big') == 0, 'auto-vacuum is not supported by this probe'
    count = len(data) // page_size
    regions, free = [], set()
    trunk = int.from_bytes(data[32:36], 'big')
    while trunk:
        assert 1 <= trunk <= count and trunk not in free
        free.add(trunk)
        base = (trunk - 1) * page_size
        nxt, n = struct.unpack_from('>II', data, base)
        assert n <= (usable - 8) // 4
        regions.append(Region(base + 8 + 4 * n, base + usable, 'freelist_trunk_tail', trunk))
        for leaf in struct.unpack_from(f'>{n}I', data, base + 8):
            assert 1 <= leaf <= count and leaf not in free
            free.add(leaf)
            start = (leaf - 1) * page_size
            regions.append(Region(start, start + usable, 'freelist_leaf', leaf))
        trunk = nxt
    assert len(free) == int.from_bytes(data[36:40], 'big')
    for page in range(1, count + 1):
        if page in free:
            continue
        base = (page - 1) * page_size
        off = base + (100 if page == 1 else 0)
        typ = data[off]
        if typ not in (2, 5, 10, 13):
            continue
        first, n, start = struct.unpack_from('>HHH', data, off + 1)
        start = start or 65536
        head = 12 if typ in (2, 5) else 8
        gap = off + head + 2 * n
        if gap < base + start:
            regions.append(Region(gap, base + start, 'btree_gap', page))
        seen = set()
        while first:
            assert first not in seen and first + 4 <= usable
            seen.add(first)
            nxt, length = struct.unpack_from('>HH', data, base + first)
            assert length >= 4 and first + length <= usable
            regions.append(Region(base + first + 4, base + first + length, 'freeblock_body', page))
            first = nxt
    return page_size, usable, sorted(regions, key=lambda r: r.start)


def schema_kind(types: list[int]) -> str | None:
    text = lambda t: t >= 13 and t % 2 == 1
    integer = lambda t: t in (1, 2, 3, 4, 5, 6, 8, 9)
    if len(types) == 6 and integer(types[0]) and types[1] == 43 and text(types[2]) and types[3:] == [0, 0, 0]:
        return 'embedding_metadata'
    if len(types) == 3 and types[0] == 43 and text(types[1]) and integer(types[2]):
        return 'embedding_metadata_string_index'
    if len(types) == 2 and types[0] == 0 and text(types[1]):
        return 'fts_content_compatible'
    if (len(types) == 8 and types[0] == 0 and types[1] == 51 and integer(types[2])
            and text(types[3]) and text(types[4]) and types[5] >= 12 and types[5] % 2 == 0
            and text(types[6]) and text(types[7])):
        return 'embeddings_queue'
    return None


def decode(payload: bytes, size: int, types: list[int]) -> tuple[list, list[tuple[int, int]]]:
    values, spans = [], []
    cursor = size
    for code in types:
        length = serial_size(code)
        raw = payload[cursor:cursor + length]
        if len(raw) != length:
            raise ValueError('truncated field')
        spans.append((cursor, cursor + length))
        cursor += length
        if code == 0:
            value = None
        elif code in (8, 9):
            value = code - 8
        elif code < 7:
            value = int.from_bytes(raw, 'big', signed=True)
        elif code == 7:
            value = struct.unpack('>d', raw)[0]
        elif code % 2:
            value = raw.decode('utf-8')
        else:
            value = raw
        values.append(value)
    if cursor != len(payload):
        raise ValueError('payload boundary')
    return values, spans


def local_size(total: int, usable: int, index: bool) -> int:
    maximum = ((usable - 12) * 64 // 255) - 23 if index else usable - 35
    minimum = ((usable - 12) * 32 // 255) - 23
    if total <= maximum:
        return total
    k = minimum + ((total - minimum) % (usable - 4))
    return k if k <= maximum else minimum


def cell_prefixes(data: bytes, pos: int, total: int, lower: int, index: bool) -> list[dict]:
    result = []
    for start in range(max(lower, pos - 18), pos):
        try:
            value, cursor = varint(data, start, pos)
            if value != total:
                continue
            rowid = None
            if not index:
                rowid, cursor = varint(data, cursor, pos)
                if not 0 < rowid < 2**63:
                    continue
            if cursor == pos:
                result.append({'offset': start, 'rowid': rowid})
        except ValueError:
            pass
    return result


def read_payload(data: bytes, pos: int, total: int, region: Region, page_size: int,
                 usable: int, index: bool, free_leaves: set[int]) -> tuple[bytes, list[dict]]:
    local = local_size(total, usable, index)
    if pos + local + (4 if local < total else 0) > region.end:
        raise ValueError('local payload leaves unused region')
    payload = data[pos:pos + local]
    segments = [{'offset': pos, 'length': local, 'payload_offset': 0}]
    if local == total:
        return payload, segments
    page = int.from_bytes(data[pos + local:pos + local + 4], 'big')
    seen = {region.page}
    while len(payload) < total:
        if page not in free_leaves or page in seen:
            raise ValueError('unpreserved or reused overflow page')
        seen.add(page)
        base = (page - 1) * page_size
        length = min(total - len(payload), usable - 4)
        segments.append({'offset': base + 4, 'length': length, 'payload_offset': len(payload)})
        payload += data[base + 4:base + 4 + length]
        page = int.from_bytes(data[base:base + 4], 'big')
    if page:
        raise ValueError('overflow chain does not terminate')
    return payload, segments


def scan_records(data: bytes) -> tuple[list[dict], list[Region]]:
    page_size, usable, regions = unused_regions(data)
    free_leaves = {r.page for r in regions if r.kind == 'freelist_leaf'}
    records = []
    for region in regions:
        # All supported headers are short, one-byte header-size values.
        for match in re.finditer(b'[\x03-\x20]', data[region.start:region.end]):
            pos = region.start + match.start()
            try:
                size, types = header(data, pos, region.end)
                kind = schema_kind(types)
                if not kind:
                    continue
                total = size + sum(serial_size(t) for t in types)
                if not size < total < 1024 * 1024:
                    continue
                index = kind.endswith('_index')
                payload, segments = read_payload(data, pos, total, region, page_size, usable, index, free_leaves)
                values, spans = decode(payload, size, types)
                prefixes = cell_prefixes(data, pos, total, region.start, index)
                document, item_id, vector, internal_id = None, None, None, None
                if kind == 'embedding_metadata':
                    internal_id, key, document = values[:3]
                    if internal_id <= 0 or key != 'chroma:document':
                        continue
                elif kind.endswith('_index'):
                    key, document, internal_id = values
                    if key != 'chroma:document' or internal_id <= 0:
                        continue
                elif kind == 'fts_content_compatible':
                    # A current freeblock can overwrite the entire cell prefix
                    # while preserving the record header at start+4. Require
                    # exact freeblock-body boundaries if the prefix is lost.
                    bounded_freeblock = (region.kind == 'freeblock_body' and pos == region.start
                                         and pos + total == region.end and len(segments) == 1)
                    if not prefixes and not bounded_freeblock:
                        continue
                    document = values[1]
                else:
                    _, created, op, topic, item_id, vector, encoding, metadata = values
                    if (not re.fullmatch(r'\d{4}-\d\d-\d\d \d\d:\d\d:\d\d', created)
                            or op != 0 or not topic.startswith('persistent://') or not item_id
                            or encoding != 'FLOAT32' or len(vector) != 1536):
                        continue
                    obj = json.loads(metadata)
                    if not isinstance(obj, dict):
                        continue
                    document = obj.get('chroma:document')
                if not isinstance(document, str) or not document:
                    continue
                records.append({'header_offset': pos, 'header_size': size, 'serial_types': types,
                                'payload_size': total, 'kind': kind, 'region_kind': region.kind,
                                'page': region.page, 'cell_prefixes': prefixes, 'segments': segments,
                                'field_spans': spans, 'document': document, 'item_id': item_id,
                                'internal_id': internal_id, 'vector': vector})
            except (ValueError, UnicodeError, struct.error):
                continue
    return records, regions
