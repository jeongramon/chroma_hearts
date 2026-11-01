"""Full-vector byte search for the PDF audit experiments.

Search files separately, never concatenate file boundaries. Manifest vectors are
search needles, not parser output. SDK-active duplicates are flagged. HNSW header
bounds and mapping, and WAL rows read on a disposable copy, distinguish occupied
inactive records from unused bytes. Raw full matches outside these structures
remain a separate denominator. No carving of fragmented vectors is attempted.
"""
from __future__ import annotations

import pickle
import sqlite3
import struct
from pathlib import Path

from chroma_hearts.parser.source_copy import working_copy
from _experiment_common import _vector


def offsets(data: bytes, needle: bytes) -> list[int]:
    if not needle:
        raise ValueError('Empty search pattern')
    found, start = [], 0
    while (pos := data.find(needle, start)) >= 0:
        found.append(pos)
        start = pos + 1
    return found


def classify_hnsw(data: bytes, header: bytes, position: int, needle: bytes,
                   mapping: dict, live: dict[str, bytes]) -> str:
    base = struct.unpack_from('<Q', header, 4)[0]
    count, stride, label_offset, vector_offset = struct.unpack_from('<QQQQ', header, 0x14)
    delta = position - base - vector_offset
    if stride == 0 or delta < 0 or delta % stride or delta // stride >= count:
        return 'hnsw_unused_or_unaligned'
    if label_offset - vector_offset != len(needle):
        return 'hnsw_dimension_mismatch'
    block = position - vector_offset
    if block + stride > len(data) or label_offset + 8 > stride:
        return 'hnsw_truncated'
    label = struct.unpack_from('<Q', data, block + label_offset)[0]
    item_id = mapping.get(label)
    return 'hnsw_active' if item_id in live and live[item_id] == needle else 'hnsw_inactive'


def scan_full_vectors(store: Path, manifest: dict) -> dict:
    files = {str(p.relative_to(store)): p.read_bytes() for p in sorted(store.rglob('*')) if p.is_file()}
    live = {item_id: _vector(item['text']) for item_id, item in manifest['live'].items()}
    live_values = set(live.values())
    wal_inactive: set[bytes] = set()
    with working_copy(store, prefix='full_vector_reference') as cp:
        with sqlite3.connect(f'file:{cp / "chroma.sqlite3"}?mode=ro', uri=True) as conn:
            for item_id, blob in conn.execute('SELECT id, vector FROM embeddings_queue WHERE vector IS NOT NULL'):
                blob = bytes(blob)
                if live.get(item_id) != blob:
                    wal_inactive.add(blob)
    mappings = {}
    for name, data in files.items():
        if name.endswith('/index_metadata.pickle'):
            mappings[str(Path(name).parent)] = pickle.loads(data).get('label_to_id', {})
    cache = {}
    events = []
    for event, target in enumerate(manifest['deleted']):
        vector = _vector(target['text'])
        if vector not in cache:
            hits, partial_files = [], []
            windows = [vector[:128], vector[len(vector)//2:len(vector)//2+128], vector[-128:]]
            for name, data in files.items():
                positions = offsets(data, vector)
                if not positions and any(window in data for window in windows):
                    partial_files.append(name)
                for pos in positions:
                    kind = 'raw_full'
                    directory = str(Path(name).parent)
                    if name.endswith('/data_level0.bin'):
                        kind = classify_hnsw(data, files[directory + '/header.bin'], pos, vector,
                                             mappings.get(directory, {}), live)
                    hits.append({'file': name, 'offset': pos, 'kind': kind})
            structural = vector in wal_inactive or any(h['kind'] == 'hnsw_inactive' for h in hits)
            shared = vector in live_values
            residual = structural or (bool(hits) and not shared)
            cache[vector] = {
                'raw_full_hits': hits, 'partial_vector_files': partial_files,
                'full_vector_in_wal_row': vector in wal_inactive,
                'full_structural_residual': structural,
                'full_residual_confirmed': residual,
                'raw_full_presence': bool(hits), 'shares_active_vector': shared,
                'partial_vector_only': not hits and vector not in wal_inactive and bool(partial_files),
                'ambiguous_active_only': bool(hits) and shared and not structural,
            }
        events.append({'event': event, 'id': target['id'], 'category': target['category'], **cache[vector]})
    keys = ['raw_full_presence', 'full_structural_residual', 'full_residual_confirmed',
            'shares_active_vector', 'partial_vector_only', 'ambiguous_active_only']
    return {'deleted_events': len(events), 'counts': {k: sum(e[k] for e in events) for k in keys},
            'events': events,
            'scope': 'Contiguous exact full-vector matches per file plus complete inactive WAL BLOBs on a copy. '
                     'Occupied inactive HNSW records and WAL rows form the structural denominator. '
                     'Same-byte events may share one physical remnant; active-only matches are not residue. '
                     'Fragmented raw vectors are not carved; partial patterns are reported separately.'}
