"""Check saved C0-C3 settings, active data and search in fresh processes.

Each SDK process receives a disposable working copy, never an evidence store.
Query every live and deleted chunk using the saved embedding-function settings.
This is functional validation, not an ANN ranking-invariance experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

from _experiment_common import REPO_ROOT, RESULTS_DIR, subprocess_env
from chroma_hearts.parser.source_copy import working_copy
from _experiment_common import fingerprint

TAG = 'mitigation_search_paper'


def save(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def worker(store: Path, manifest_path: Path, output: Path) -> None:
    import chromadb
    import numpy as np
    from chromadb.config import Settings
    from chroma_hearts.generator.embedding import HashEmbeddingFunction  # registers saved EF
    from _experiment_common import _vector

    manifest = json.loads(manifest_path.read_text())
    live = manifest['live']
    deleted_ids = {t['id'] for t in manifest['deleted']}
    assert live and not set(live) & deleted_ids
    client = chromadb.PersistentClient(path=str(store), settings=Settings(anonymized_telemetry=False))
    col = client.get_collection('pdf_workload')
    configuration = col.configuration_json
    assert configuration['hnsw']['space'] == 'l2'
    assert configuration['hnsw']['sync_threshold'] == 1000
    data = col.get(include=['documents', 'metadatas', 'embeddings'])
    assert len(data['ids']) == len(set(data['ids'])) == len(live) == col.count()
    assert set(data['ids']) == set(live)
    snapshots = {}
    for i, item_id in enumerate(data['ids']):
        expected = live[item_id]
        blob = np.asarray(data['embeddings'][i], dtype='<f4').tobytes()
        assert blob == _vector(expected['text']), item_id
        assert data['documents'][i] == expected['text'], item_id
        assert data['metadatas'][i] == expected['metadata'], item_id
        snapshots[item_id] = {'document_sha256': hashlib.sha256(expected['text'].encode()).hexdigest(),
                              'vector_sha256': hashlib.sha256(blob).hexdigest(),
                              'metadata': data['metadatas'][i]}
    queries = ([{'id': key, 'text': value['text'], 'kind': 'live'} for key, value in sorted(live.items())]
               + [{'id': t['id'], 'text': t['text'], 'kind': 'deleted'} for t in manifest['deleted']])
    answers, self_first = [], 0
    for start in range(0, len(queries), 64):
        batch = queries[start:start + 64]
        result = col.query(query_texts=[q['text'] for q in batch], n_results=10,
                           include=['documents', 'metadatas', 'distances'])
        assert len(result['ids']) == len(batch)
        for j, query in enumerate(batch):
            ids, distances = result['ids'][j], result['distances'][j]
            assert len(ids) == len(set(ids)) == 10
            assert set(ids) <= set(live) and not set(ids) & deleted_ids
            assert len(distances) == 10 and np.isfinite(distances).all()
            for k, item_id in enumerate(ids):
                assert result['documents'][j][k] == live[item_id]['text']
                assert result['metadatas'][j][k] == live[item_id]['metadata']
            self_first += query['kind'] == 'live' and ids[0] == query['id']
            answers.append({'query_id': query['id'], 'query_kind': query['kind'],
                            'result_ids': ids, 'distances': distances})
    save(output, {'configuration': configuration, 'collection_metadata': col.metadata,
                  'collection_name': col.name, 'active_items': len(live),
                  'active_snapshot_sha256': hashlib.sha256(json.dumps(snapshots, sort_keys=True).encode()).hexdigest(),
                  'active_data_exact': True, 'queries': len(queries),
                  'live_queries': len(live), 'deleted_queries': len(manifest['deleted']),
                  'k': 10, 'returned_results': len(queries) * 10,
                  'invalid_or_deleted_result_ids': 0, 'query_payloads_exact': True,
                  'live_self_rank1': self_first, 'answers': answers})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--worker', action='store_true')
    ap.add_argument('--store', type=Path)
    ap.add_argument('--manifest', type=Path)
    ap.add_argument('--output', type=Path, default=RESULTS_DIR/(TAG+'.json'))
    ap.add_argument('--mitigation', type=Path, default=RESULTS_DIR/'mitigation_paper.json')
    ap.add_argument('--workload', type=Path, default=RESULTS_DIR/'workload_audit_1.5.9_paper.json')
    ap.add_argument('--tag', default=TAG)
    args = ap.parse_args()
    if args.worker:
        worker(args.store, args.manifest, args.output)
        return
    source_path = args.mitigation.resolve()
    workload_path = args.workload.resolve()
    source = json.loads(source_path.read_text())
    workload_root = Path(json.loads(workload_path.read_text())['root'])
    outroot = REPO_ROOT / 'sample/experiments' / args.tag
    outroot.mkdir(parents=True, exist_ok=True)
    baselines, runs, manifests = {}, [], {}
    for row in source['runs']:
        if row['stage'] not in ('C0', 'C1', 'C2', 'C3'):
            continue
        seed, stage, store = row['seed'], row['stage'], Path(row['store'])
        manifest = workload_root / f'seed{seed}' / 'item_delete/manifest.json'
        manifests[str(manifest)] = sha(manifest)
        before = fingerprint(store)
        assert before == row['procedure']['output_sha256']
        output = outroot / f'seed{seed}_{stage}.json'
        with working_copy(store, prefix='mitigation_search') as cp:
            with output.with_suffix('.log').open('w') as log:
                subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker',
                                '--store', str(cp), '--manifest', str(manifest), '--output', str(output)],
                               cwd=REPO_ROOT, env=subprocess_env(), check=True,
                               stdout=log, stderr=subprocess.STDOUT)
        assert fingerprint(store) == before
        result = json.loads(output.read_text())
        if stage == 'C0':
            baselines[seed] = result
        baseline = baselines[seed]
        for key in ('configuration', 'collection_metadata', 'collection_name', 'active_snapshot_sha256'):
            assert result[key] == baseline[key], (seed, stage, key, result[key], baseline[key])
        summary = {k: v for k, v in result.items() if k != 'answers'}
        summary.update(seed=seed, stage=stage, settings_equal_C0=True, snapshot_equal_C0=True,
                       fresh_process=True, original_unchanged=True,
                       evidence_sha256=before, detail_path=str(output), detail_sha256=sha(output))
        runs.append(summary)
        print(json.dumps({k: summary[k] for k in ('seed', 'stage', 'active_items', 'queries',
                         'settings_equal_C0', 'live_self_rank1', 'invalid_or_deleted_result_ids')}), flush=True)
    assert len(runs) == 4 * len(source['seeds'])
    paths = [Path(__file__).resolve(), REPO_ROOT/'src/chroma_hearts/generator/embedding.py',
             REPO_ROOT/'src/chroma_hearts/parser/source_copy.py', REPO_ROOT/'scripts/_experiment_common.py']
    report = {'method': 'Fresh process per saved C0-C3 working copy; SDK get exact active-data checks; '
              'compare full SDK collection configuration and collection metadata to C0; '
              'top-10 query_texts for every active and deleted chunk using the saved embedding function.',
              'scope': 'Functional search check, not semantic retrieval quality or invariant ANN ranking. '
              'No multiple-collection experiment.',
              'input_sha256': {str(p): sha(p) for p in (source_path, workload_path)},
              'manifest_sha256': manifests,
              'code_sha256': {str(p.relative_to(REPO_ROOT)): sha(p) for p in paths},
              'environment': {'python': sys.version, 'platform': platform.platform(),
                              **{p: importlib.metadata.version(p) for p in ('chromadb', 'numpy')}},
              'runs': runs, 'status': 'passed'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save(args.output, report)
    print('PASSED', args.output, flush=True)


if __name__ == '__main__':
    main()
