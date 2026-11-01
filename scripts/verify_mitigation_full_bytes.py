"""Independent known-target full-value search of final mitigation stores.

No SDK or recovery output is used. Raw searches include every byte of every
remaining file, including unused HNSW capacity. Native SQLite values are read
only on disposable copies. The separate experimental SQLite record probe and
SQLite's native decoder supplement contiguous searches for overflow records.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import time
from collections import Counter
from pathlib import Path

from _audit_full_scan import offsets
from _audit_full_text import whole_text_encodings
from _sqlite_record_probe import scan_records
from _experiment_common import _vector
from _sqlite_native import native_decode
from chroma_hearts.parser.source_copy import working_copy

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'sample/experiment_results'
OUTPUT = RESULTS / 'mitigation_full_bytes_paper.json'


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_hashes(store: Path) -> dict:
    return {str(p.relative_to(store)): digest(p.read_bytes())
            for p in sorted(store.rglob('*')) if p.is_file()}


def run_case(row: dict, manifest_path: Path) -> dict:
    started = time.monotonic()
    store = Path(row['store'])
    files = {str(p.relative_to(store)): p.read_bytes()
             for p in sorted(store.rglob('*')) if p.is_file()}
    hashes = {name: digest(data) for name, data in files.items()}
    assert hashes == row['procedure']['output_sha256'], store
    manifest = json.loads(manifest_path.read_text())
    targets = manifest['deleted']
    assert len(targets) == row['targets']
    assert targets
    vectors = [_vector(t['text']) for t in targets]
    vector_lookup, text_lookup = {}, {}
    for i, target in enumerate(targets):
        vector_lookup.setdefault(vectors[i], []).append(i)
        text_lookup.setdefault(target['text'], []).append(i)
    events = [{'id': t['id'], 'vector_sha256': digest(v),
               'text_sha256': digest(t['text'].encode()),
               'raw_vector_hits': [], 'raw_text_hits': [],
               'native_vector_hits': [], 'native_text_hits': [],
               'unused_record_vector_hits': [], 'unused_record_text_hits': []}
              for t, v in zip(targets, vectors)]
    for i, target in enumerate(targets):
        for name, data in files.items():
            events[i]['raw_vector_hits'].extend(
                {'file': name, 'offset': pos, 'length': len(vectors[i])}
                for pos in offsets(data, vectors[i]))
            for encoding, needle in whole_text_encodings(target['text']).items():
                events[i]['raw_text_hits'].extend(
                    {'file': name, 'offset': pos, 'length': len(needle), 'encoding': encoding}
                    for pos in offsets(data, needle))

    def match_values(values, prefix, location):
        for column, value in enumerate(values):
            if isinstance(value, bytes):
                for i in vector_lookup.get(value, []):
                    events[i][prefix + '_vector_hits'].append({**location, 'column': column})
            elif isinstance(value, str):
                documents = [value]
                if value.startswith('{'):
                    try:
                        obj = json.loads(value)
                        if isinstance(obj, dict) and isinstance(obj.get('chroma:document'), str):
                            documents.append(obj['chroma:document'])
                    except ValueError:
                        pass
                for document in set(documents):
                    for i in text_lookup.get(document, []):
                        events[i][prefix + '_text_hits'].append({**location, 'column': column})

    with working_copy(store, prefix='mitigation_full_bytes') as cp:
        with sqlite3.connect(f'file:{cp / "chroma.sqlite3"}?mode=ro', uri=True) as conn:
            assert conn.execute('PRAGMA quick_check').fetchall() == [('ok',)]
            freelist = conn.execute('PRAGMA freelist_count').fetchone()[0]
            tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_schema WHERE type='table'")]
            for table in tables:
                quoted = '"' + table.replace('"', '""') + '"'
                for ordinal, values in enumerate(conn.execute('SELECT * FROM ' + quoted)):
                    match_values(values, 'native', {'table': table, 'row_ordinal': ordinal})

    records, regions = scan_records(files['chroma.sqlite3'])
    templates = {}
    for record in records:
        payload = b''.join(files['chroma.sqlite3'][s['offset']:s['offset'] + s['length']]
                           for s in record['segments'])
        assert len(payload) == record['payload_size']
        values = native_decode(payload, len(record['serial_types']), templates)
        match_values(values, 'unused_record', {'header_offset': record['header_offset'],
                     'kind': record['kind'], 'segments': record['segments']})
    keys = [k for k in events[0] if k.endswith('_hits')]
    counts = {key.replace('_hits', '_targets'): sum(bool(e[key]) for e in events) for key in keys}
    counts['vector_targets_any'] = sum(any(e[k] for k in keys if '_vector_' in k) for e in events)
    counts['text_targets_any'] = sum(any(e[k] for k in keys if '_text_' in k) for e in events)
    assert file_hashes(store) == hashes, 'Evidence modified'
    return {'seed': row['seed'], 'stage': row['stage'], 'deletion': row['deletion'],
            'positive_control': row['stage'] in ('C0', 'D0'), 'targets': len(targets),
            'store': str(store), 'file_sha256': hashes, 'manifest_sha256': digest(manifest_path.read_bytes()),
            'files_scanned': len(files), 'bytes_scanned': sum(map(len, files.values())),
            'sqlite_freelist_count': freelist, 'unused_regions': dict(Counter(r.kind for r in regions)),
            'unused_records_native_decoded': len(records), 'counts': counts,
            'hit_occurrences': {key: sum(len(e[key]) for e in events) for key in keys},
            'events': events, 'evidence_unchanged': True, 'seconds': time.monotonic() - started}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mitigation', type=Path, default=RESULTS/'mitigation_paper.json')
    ap.add_argument('--workload', type=Path, default=RESULTS/'workload_audit_1.5.9_paper.json')
    ap.add_argument('--output', type=Path, default=OUTPUT)
    args = ap.parse_args()
    source_path, workload_path = args.mitigation.resolve(), args.workload.resolve()
    source = json.loads(source_path.read_text())
    workload_root = Path(json.loads(workload_path.read_text())['root'])
    runs = []
    for row in source['runs']:
        if row['stage'] not in ('C3', 'D2') and not (row['seed'] == source['seeds'][0] and row['stage'] in ('C0', 'D0')):
            continue
        manifest = workload_root / f"seed{row['seed']}" / row['deletion'] / 'manifest.json'
        result = run_case(row, manifest)
        runs.append(result)
        print(json.dumps({k: result[k] for k in ('seed', 'stage', 'targets', 'counts', 'seconds')}), flush=True)
    controls = [r for r in runs if r['positive_control']]
    assert len(controls) == 2
    assert all(r['counts']['vector_targets_any'] and r['counts']['text_targets_any'] for r in controls)
    final = [r for r in runs if not r['positive_control']]
    assert len(final) == 2 * len(source['seeds'])
    paths = [Path(__file__), ROOT/'scripts/_audit_full_scan.py', ROOT/'scripts/_audit_full_text.py',
             ROOT/'scripts/_sqlite_record_probe.py', ROOT/'scripts/_sqlite_native.py',
             ROOT/'scripts/_experiment_common.py', ROOT/'src/chroma_hearts/generator/embedding.py',
             ROOT/'src/chroma_hearts/parser/source_copy.py']
    report = {'method': 'Exact full 384-float32 vector and whole UTF-8/JSON text search per file, '
              'plus native SQLite values and independently probed unused records decoded by SQLite. '
              'No recovery output, SDK access, similarity tolerance, or partial-fragment scoring.',
              'scope_limit': 'Remaining files only, not removed files or filesystem unallocated space. '
              'Arbitrary broken overflow chains and partial-only residues are not reconstructed.',
              'input_sha256': {str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in (source_path, workload_path)},
              'code_sha256': {str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in paths},
              'positive_controls_passed': True, 'runs': runs,
              'all_final_full_targets_absent': all(not r['counts']['vector_targets_any'] and
                                                  not r['counts']['text_targets_any'] for r in final)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print('REPORT', args.output, 'all_final_full_targets_absent=', report['all_final_full_targets_absent'], flush=True)
    if not report['all_final_full_targets_absent']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
