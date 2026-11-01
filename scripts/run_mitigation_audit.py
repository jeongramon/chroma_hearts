"""Audit staged mitigation on the ten preserved section-6.2 PDF stores.

No byte scanner: the production parser supplies all recovered evidence. Known
deleted items enter only post-extraction grading. Each stage is copied from its
predecessor; originals and completed stages remain immutable. Run serially so
procedure timing is not contaminated by concurrent experiment workers.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sqlite3
import statistics
import subprocess
import sys
import time

from _experiment_common import REPO_ROOT, RESULTS_DIR, spawn_call, subprocess_env, write_csv
from _experiment_common import fingerprint
from _experiment_common import save_json
from _experiment_common import _vector
from chroma_hearts import mitigation
from chroma_hearts.evaluator.recovery_paths import PATHS, RecoveryIndex, save_targets, summarize
from chroma_hearts.parser.source_copy import working_copy

STAGES = {'item_delete': ('C0', 'C1', 'C2', 'C3'),
          'collection_drop': ('D0', 'D1', 'D2')}


def copy_store(source: Path, destination: Path) -> None:
    # All SQLite / SDK access is confined to this derivative or parser copies.
    with working_copy(source, prefix='mitigation_stage') as copy:
        shutil.copytree(copy, destination)


def apply_stage(store: Path, stage: str) -> dict:
    actions = {}
    started = time.perf_counter()
    with mitigation.DiskPeak(store) as peak:
        if stage == 'C1':
            actions['flush'] = asdict(mitigation.flush_pending(store, 'pdf_workload'))
        if stage == 'C2':
            actions['rebuild'] = asdict(spawn_call(mitigation.rebuild_collection, store, 'pdf_workload'))
        if stage in ('C2', 'D1'):
            actions['removed_segment_dirs'] = mitigation.remove_unreferenced_segments(store)
        if stage in ('C3', 'D2'):
            actions['vacuum_bytes_reclaimed'] = mitigation.vacuum_sqlite(store)
    elapsed = time.perf_counter() - started
    return {'actions': actions, 'step_seconds': elapsed,
            'stage_start_bytes': peak.start, 'stage_peak_bytes': peak.peak,
            'step_peak_extra_bytes': peak.extra,
            'stage_end_bytes': sum(p.stat().st_size for p in store.rglob('*') if p.is_file()),
            'output_sha256': fingerprint(store)}


def verify_active(index: RecoveryIndex, manifest: dict) -> dict:
    sdk = [row for row in index.rows if row['method'] == 'sdk']
    assert len(sdk) == len({row['id'] for row in sdk})
    actual = {row['id']: row for row in sdk}
    assert actual.keys() == manifest['live'].keys(), 'Active ID set changed'
    digest = {}
    for key, item in manifest['live'].items():
        row = actual[key]
        assert row['vector_hash'] == hashlib.sha256(_vector(item['text'])).hexdigest(), key
        assert row['document'] == item['text'], key
        metadata = json.loads(row['metadata'] or '{}')
        metadata.pop('chroma:document', None)
        assert metadata == item['metadata'], (key, metadata, item['metadata'])
        digest[key] = {'vector_sha256': row['vector_hash'],
                       'text_sha256': hashlib.sha256(item['text'].encode()).hexdigest(),
                       'metadata': metadata}
    return {'items': len(actual), 'ids_documents_metadata_vectors_exact': True,
            'snapshot_sha256': hashlib.sha256(json.dumps(digest, sort_keys=True).encode()).hexdigest()}


def audit(case: Path, report: Path, manifest: dict, mode: str) -> tuple[dict, dict]:
    store = case/'chroma_db'
    before = fingerprint(store)
    if not report.exists():
        with (case/'parser.log').open('w') as log:
            subprocess.run([sys.executable, '-m', 'chroma_hearts.parser', '-i', str(case),
                            '-o', str(report.parent)], check=True, cwd=REPO_ROOT,
                           env=subprocess_env(), stdout=log, stderr=subprocess.STDOUT)
    assert json.loads((report/'sqlite_carving.json').read_text())['status'] == 'ok'
    index = RecoveryIndex(report/'embeddings.sqlite3')
    active = verify_active(index, manifest)
    events = []
    for target in manifest['deleted']:
        vh = hashlib.sha256(_vector(target['text'])).hexdigest()
        grade = index.grade(target['id'], vh, target['text'])
        grade.update(event_type=mode, target_id=target['id'], vector_hash=vh,
                     text_sha256=hashlib.sha256(target['text'].encode()).hexdigest())
        events.append(grade)
    counts = summarize(events)
    assert sum(counts[k] for k in (*PATHS, 'none')) == counts['targets']
    assert counts['unclassified_data'] == 0
    with sqlite3.connect(f'file:{report / "embeddings.sqlite3"}?mode=ro', uri=True) as conn:
        assert conn.execute('SELECT count(*) FROM recovery_evidence e LEFT JOIN embeddings x '
                            'ON x.rowid=e.embedding_rowid WHERE x.rowid IS NULL').fetchone()[0] == 0
    save_targets(report/'recovery_targets.sqlite3', events)
    save_json(report/'target_results.json', events)
    assert fingerprint(store) == before, 'Audit modified its source'
    return counts, active


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tag', default='mitigation_paper')
    ap.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    ap.add_argument('--workload', type=Path, default=RESULTS_DIR/'workload_audit_1.5.9_paper.json')
    ap.add_argument('--recovery', type=Path, help='Separate historical recovery result, if needed')
    args = ap.parse_args()
    assert Path(args.tag).name == args.tag and args.tag not in ('.', '..')
    assert len(set(args.seeds)) == len(args.seeds)
    source_report = args.workload.resolve()
    source = json.loads(source_report.read_text())
    source_root = Path(source['root'])
    reference = json.loads((args.recovery or source_report).read_text())
    baseline_counts = {(r['seed'], r['deletion']): r['summary'] for r in reference['runs']}
    root = REPO_ROOT/'sample/experiments'/args.tag
    root.mkdir(parents=True, exist_ok=True)
    code_files = [*sorted((REPO_ROOT/'src/chroma_hearts/parser').glob('*.py')),
                  REPO_ROOT/'src/chroma_hearts/evaluator/recovery_paths.py',
                  REPO_ROOT/'src/chroma_hearts/mitigation.py',
                  REPO_ROOT/'scripts/_experiment_common.py', Path(__file__).resolve()]
    provenance = {'source_report_sha256': hashlib.sha256(source_report.read_bytes()).hexdigest(),
                  'corpus_sha256': hashlib.sha256((source_root/'corpus.json').read_bytes()).hexdigest(),
                  'code_sha256': {str(p.relative_to(REPO_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in code_files},
                  'environment': {'python': sys.version, 'platform': platform.platform(),
                                  **{p: importlib.metadata.version(p)
                                     for p in ('chromadb', 'chroma-hnswlib', 'numpy')}}}
    provenance_path = root/'provenance.json'
    if provenance_path.exists():
        assert json.loads(provenance_path.read_text()) == provenance, 'Inputs or code changed on resume'
    else:
        save_json(provenance_path, provenance)
    runs = []
    for seed in args.seeds:
        for mode, stages in STAGES.items():
            original = source_root/f'seed{seed}'/mode
            manifest = json.loads((original/'manifest.json').read_text())
            source_hashes = json.loads((original/'checks.json').read_text())['evidence_sha256']
            assert fingerprint(original/'chroma_db') == source_hashes
            previous = original/'chroma_db'
            previous_hashes = source_hashes
            cumulative_time, cumulative_peak = 0.0, 0
            start_bytes = sum(p.stat().st_size for p in previous.rglob('*') if p.is_file())
            for stage in stages:
                case = root/f'seed{seed}'/stage
                case.mkdir(parents=True, exist_ok=True)
                store = case/'chroma_db'
                procedure_path = case/'procedure.json'
                if not procedure_path.exists():
                    assert not store.exists(), f'Incomplete stage requires inspection: {case}'
                    assert fingerprint(previous) == previous_hashes
                    copy_store(previous, store)
                    if stage in ('C0', 'D0'):
                        procedure = {'actions': {}, 'step_seconds': 0.0,
                                     'stage_start_bytes': start_bytes, 'stage_peak_bytes': start_bytes,
                                     'step_peak_extra_bytes': 0, 'stage_end_bytes': start_bytes,
                                     'output_sha256': fingerprint(store)}
                    else:
                        procedure = apply_stage(store, stage)
                    assert fingerprint(previous) == previous_hashes
                    save_json(procedure_path, procedure)
                procedure = json.loads(procedure_path.read_text())
                assert fingerprint(store) == procedure['output_sha256']
                cumulative_time += procedure['step_seconds']
                cumulative_peak = max(cumulative_peak, procedure['stage_peak_bytes']-start_bytes)
                report = root/f'seed{seed}'/'reports'/stage
                counts, active = audit(case, report, manifest, mode)
                if stage in ('C0', 'D0'):
                    assert counts == baseline_counts[(seed, mode)], 'Section-6.2 baseline changed'
                row = {'seed': seed, 'deletion': mode, 'stage': stage, **counts,
                       'active': active, 'procedure': procedure,
                       'cumulative_seconds': cumulative_time,
                       'cumulative_peak_extra_bytes': cumulative_peak,
                       'store': str(store), 'report': str(report),
                       'original_unchanged': True, 'audit_source_unchanged': True}
                save_json(case/'result.json', row)
                runs.append(row)
                print(json.dumps({k: v for k, v in row.items()
                                  if k not in ('procedure', 'store', 'report')}, ensure_ascii=False), flush=True)
                previous, previous_hashes = store, procedure['output_sha256']
            assert fingerprint(original/'chroma_db') == source_hashes
    aggregate = []
    keys = ('targets', *PATHS, 'none', 'vector_recovered', 'text_recovered')
    for mode, stages in STAGES.items():
        for stage in stages:
            rows = [r for r in runs if r['stage'] == stage]
            row = {'deletion': mode, 'stage': stage, 'repetitions': len(rows),
                   **{key: sum(r[key] for r in rows) for key in keys},
                   'active_items': sum(r['active']['items'] for r in rows),
                   'active_exact': all(r['active']['ids_documents_metadata_vectors_exact'] for r in rows)}
            row.update({key+'_rate': row[key]/row['targets'] for key in keys if key != 'targets'})
            for metric in ('cumulative_seconds', 'cumulative_peak_extra_bytes'):
                values = [r[metric] for r in rows]
                row.update({metric+'_mean': statistics.mean(values),
                            metric+'_min': min(values), metric+'_max': max(values)})
            aggregate.append(row)
    result = {'tag': args.tag, 'seeds': args.seeds, 'source_section_6_2': str(source_root),
              'no_reinsertions': True, 'scanner_used': False, 'runs': runs, 'aggregate': aggregate,
              'provenance': provenance,
              'timing': 'Serial steps; includes subprocess startup and internal copy validation. '
                        'Excludes snapshot copying and independent audits. Cumulative time sums steps; '
                        'peak extra is the largest sampled store size minus the baseline size. '
                        '20 ms disk sampling; not an application downtime measurement.'}
    save_json(RESULTS_DIR/f'{args.tag}.json', result)
    write_csv(aggregate, RESULTS_DIR/f'{args.tag}.csv', list(aggregate[0]))
    print(json.dumps(aggregate, indent=2), flush=True)


if __name__ == '__main__':
    main()
