"""Validate every stored recovery output against complete insertion histories.

This supplements target-level recall: examine both merged output rows and each
source-evidence row, without filtering to known deleted targets or successful
matches. Active data are valid outputs too. Unknown IDs remain unknown.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from _experiment_common import REPO_ROOT, RESULTS_DIR
from chroma_hearts.parser.source_copy import working_copy
from _experiment_common import fingerprint
from _experiment_common import _vector

TAG = 'recovery_output_values_paper'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_rows(rows: list[dict], history: dict) -> dict:
    vector_ids, text_ids = defaultdict(set), defaultdict(set)
    for item_id, target in history.items():
        vector_ids[_vector(target['text'])].add(item_id)
        text_ids[target['text']].add(item_id)
    counts = Counter(rows=len(rows), vector_outputs=0, text_outputs=0,
                     paired_outputs=0, id_linked_data_outputs=0, outputs_without_id=0,
                     invalid_vectors=0, invalid_texts=0, invalid_metadata=0,
                     inconsistent_vector_text_pairs=0, incorrect_id_links=0,
                     rows_with_errors=0)
    by_method, errors = defaultdict(Counter), []
    for row in rows:
        method = row.get('method', 'merged')
        by_method[method]['rows'] += 1
        bad, candidates = [], []
        vector = row['vector']
        if vector is not None:
            counts['vector_outputs'] += 1
            by_method[method]['vector_outputs'] += 1
            matching = vector_ids.get(bytes(vector), set())
            candidates.append(matching)
            if not matching:
                counts['invalid_vectors'] += 1
                bad.append('vector_not_in_insertion_history')
        text = None
        if row['metadata'] is not None:
            try:
                metadata = json.loads(row['metadata'])
                if not isinstance(metadata, dict):
                    raise ValueError('Metadata is not an object')
                if 'chroma:document' in metadata:
                    text = metadata['chroma:document']
                    if not isinstance(text, str):
                        raise ValueError('Document is not a string')
            except (TypeError, ValueError):
                counts['invalid_metadata'] += 1
                bad.append('invalid_metadata_or_document_type')
        if isinstance(text, str):
            counts['text_outputs'] += 1
            by_method[method]['text_outputs'] += 1
            matching = text_ids.get(text, set())
            candidates.append(matching)
            if not matching:
                counts['invalid_texts'] += 1
                bad.append('whole_text_not_in_insertion_history')
        if vector is not None and isinstance(text, str):
            counts['paired_outputs'] += 1
            if all(candidates) and not set.intersection(*candidates):
                counts['inconsistent_vector_text_pairs'] += 1
                bad.append('vector_text_belong_to_different_items')
        item_id = row['id']
        if candidates:
            if item_id is None:
                counts['outputs_without_id'] += 1
            else:
                counts['id_linked_data_outputs'] += 1
                if item_id not in history or any(item_id not in ids for ids in candidates):
                    counts['incorrect_id_links'] += 1
                    bad.append('id_does_not_match_output_data')
        if bad:
            counts['rows_with_errors'] += 1
            by_method[method]['rows_with_errors'] += 1
            errors.append({'rowid': row['output_rowid'], 'method': method,
                           'id': item_id, 'errors': bad,
                           'vector_sha256': hashlib.sha256(bytes(vector)).hexdigest() if vector is not None else None,
                           'text_sha256': hashlib.sha256(text.encode()).hexdigest() if isinstance(text, str) else None})
    return {'counts': dict(counts), 'by_method': {k: dict(v) for k, v in by_method.items()}, 'errors': errors}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workload', type=Path, default=RESULTS_DIR/'workload_audit_1.5.9_paper.json')
    ap.add_argument('--recovery', type=Path, help='Separate historical recovery result, if needed')
    ap.add_argument('--output', type=Path, default=RESULTS_DIR/(TAG+'.json'))
    args = ap.parse_args()
    workload_path = args.workload.resolve()
    reference_path = (args.recovery or workload_path).resolve()
    reference = json.loads(reference_path.read_text())
    workload = json.loads(workload_path.read_text())
    root = Path(workload['root'])
    originals = {(r['seed'], r['deletion']): r for r in workload['runs']}
    runs = []
    for row in reference['runs']:
        seed, mode = row['seed'], row['deletion']
        case = root / f'seed{seed}' / mode
        manifest_path = case / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        history = {}
        for target in manifest['all_deleted'] + list(manifest['live'].values()):
            item_id = target['id']
            if item_id in history:
                assert target['text'] == history[item_id]['text']
            history[item_id] = target
        assert len(history) == originals[seed, mode]['operations']['add']
        store = case / 'chroma_db'
        before = fingerprint(store)
        assert before == originals[seed, mode]['checks']['evidence_sha256']
        report_dir = Path(row['report'])
        output_db = report_dir / 'embeddings.sqlite3'
        output_before = sha(output_db)
        audits = {}
        with working_copy(report_dir, prefix='output_values_validation') as cp:
            with sqlite3.connect(f'file:{cp / "embeddings.sqlite3"}?mode=ro', uri=True) as conn:
                conn.row_factory = sqlite3.Row
                for table in ('embeddings', 'recovery_evidence'):
                    rows = [dict(r) for r in conn.execute(f'SELECT rowid AS output_rowid,* FROM {table}')]
                    audits[table] = audit_rows(rows, history)
        assert sha(output_db) == output_before
        assert fingerprint(store) == before
        result = {'seed': seed, 'deletion': mode, 'inserted_items': len(history),
                  'active_items': len(manifest['live']), 'deleted_targets': len(manifest['deleted']),
                  'manifest_sha256': sha(manifest_path), 'recovery_db_sha256': output_before,
                  'recovery_db': str(output_db), 'evidence_sha256': before,
                  'source_and_output_unchanged': True, **audits}
        runs.append(result)
        print(json.dumps({'seed': seed, 'deletion': mode,
                          **{t: audits[t]['counts'] for t in audits}}), flush=True)
    assert len(runs) == len(workload['runs'])
    aggregate = {}
    for table in ('embeddings', 'recovery_evidence'):
        counts = Counter()
        for row in runs:
            counts.update(row[table]['counts'])
        aggregate[table] = dict(counts)
    paths = [Path(__file__).resolve(), REPO_ROOT/'scripts/_experiment_common.py',
             REPO_ROOT/'src/chroma_hearts/generator/embedding.py',
             REPO_ROOT/'src/chroma_hearts/parser/source_copy.py']
    result = {'method': 'Every merged row and every source-evidence row in the existing section-6.2 '
              'production output, including SDK, WAL, HNSW and SQLite carving, compared against '
              'all inserted items (active and deleted). Exact full float32 vector bytes and whole text; '
              'check vector-text consistency and known ID association. No target-success filtering.',
              'scope': 'Output-value validity in the existing ten workloads, not general deletion-attribution '
              'precision. Counts include overlapping sources and duplicate evidence records; '
              'missing IDs are not inferred and rows without data are not treated as recovered values.',
              'input_sha256': {str(p.relative_to(REPO_ROOT)): sha(p) for p in (reference_path, workload_path)},
              'code_sha256': {str(p.relative_to(REPO_ROOT)): sha(p) for p in paths},
              'runs': runs, 'aggregate': aggregate,
              'all_outputs_valid': all(not a['rows_with_errors'] for a in aggregate.values())}
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print('REPORT', output, 'all_outputs_valid=', result['all_outputs_valid'], flush=True)
    if not result['all_outputs_valid']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
