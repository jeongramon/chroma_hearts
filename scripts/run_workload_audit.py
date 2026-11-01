"""Section 6.2: paired arXiv workloads, independent reference, and recovery recall."""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from _experiment_common import (REPO_ROOT, RESULTS_DIR, chromadb_version, prepare_pdf_experiment,
                                spawn_call, save_json, fingerprint, _vector, run)
from _audit_full_scan import scan_full_vectors
from _audit_full_text import scan_full_texts, verify_full_text_hits
from _sqlite_record_probe import scan_records
from _sqlite_native import native_decode
from chroma_hearts.evaluator.recovery_paths import RecoveryIndex, save_targets, summarize

def make_plan(size: int, iterations: int, seed: int) -> list[dict]:
    """Choose operations and targets without ever inserting a chunk twice."""
    rng = random.Random(seed)
    unused, active, plan = list(range(size)), [], []
    for iteration in range(1, iterations + 1):
        operation = rng.choices(["add", "delete", "search"], weights=[60, 30, 10])[0]
        entry = {"iteration": iteration, "operation": operation, "executed": False}
        if operation == "add":
            if not unused:
                raise ValueError("Corpus exhausted; shorten the workload, never recycle chunks")
            item = unused.pop(rng.randrange(len(unused)))
            active.append(item)
            entry.update(item=item, executed=True)
        elif operation == "delete" and active:
            entry.update(item=active.pop(rng.randrange(len(active))), executed=True)
        elif operation == "search" and active:
            entry.update(item=rng.randrange(size), executed=True)
        plan.append(entry)
    return plan


def write_workload(store: str, corpus: list[dict], plan: list[dict], drop: bool) -> dict:
    import chromadb
    from chromadb.config import Settings
    from chroma_hearts.generator.embedding import HashEmbeddingFunction

    client = chromadb.PersistentClient(path=store, settings=Settings(anonymized_telemetry=False))
    collection = client.create_collection("pdf_workload", embedding_function=HashEmbeddingFunction())
    live, deleted = {}, []
    for event in plan:
        if not event["executed"]:
            continue
        item = corpus[event["item"]]
        if event["operation"] == "add":
            collection.add(ids=[item["id"]], documents=[item["text"]], metadatas=[item["metadata"]])
            live[item["id"]] = item
        elif event["operation"] == "delete":
            collection.delete(ids=[item["id"]])
            deleted.append({**live.pop(item["id"]), "category": "item_delete"})
        else:
            collection.query(query_texts=[item["text"]], n_results=min(3, len(live)), include=[])
    assert collection.count() == len(live)
    before_drop = list(live.values())
    targets = deleted
    if drop:
        targets = [{**item, "category": "collection_drop"} for item in before_drop]
        client.delete_collection(collection.name)
        live = {}
    return {"deleted": targets, "live": live, "all_deleted": deleted + (targets if drop else []),
            "individual_deletes": len(deleted), "active_before_drop": len(before_drop),
            "configuration": {"space": "l2", "sync_threshold": 1000, "defaults": True}}


def extract_api(store: str) -> dict:
    """Use only HNSW files and public label/vector API; no SDK, SQLite or ID map."""
    import hnswlib
    import numpy as np
    from chroma_hearts.parser.source_copy import working_copy

    vectors, errors = [], []
    removed, enumerated = 0, 0
    with working_copy(Path(store), prefix="vector_only_api") as cp:
        for mapping in cp.rglob("index_metadata.pickle"):
            mapping.unlink()  # Enforce that this baseline cannot consult ID mappings.
            removed += 1
        for directory in sorted(cp.iterdir()):
            if not (directory / "header.bin").exists():
                continue
            index = hnswlib.Index(space="l2", dim=384)
            index.load_index(str(directory), is_persistent_index=True)
            labels = index.get_ids_list()
            enumerated += len(labels)
            for label in labels:
                try:
                    vector = np.asarray(index.get_items([label])[0], dtype="<f4").tobytes()
                    vectors.append(vector.hex())
                except RuntimeError as error:
                    errors.append({"segment": directory.name, "label": int(label), "error": str(error)})
    return {"vectors": vectors, "library_errors": errors, "labels_enumerated": enumerated,
            "id_mapping_used": False, "mapping_files_removed_from_copy": removed}


def evaluate_case(case: Path, manifest: dict, report_base: Path) -> dict:
    """Extract blind, then score against an independently decoded reference."""
    store = case / 'chroma_db'
    before = fingerprint(store)
    api = spawn_call(extract_api, str(store))
    run([sys.executable, '-m', 'chroma_hearts.parser',
         '-i', str(case), '-o', str(report_base)], case / 'parser.log')
    report = report_base / case.name
    index = RecoveryIndex(report / 'embeddings.sqlite3')
    assert json.loads((report / 'sqlite_carving.json').read_text())['status'] == 'ok'

    vectors = scan_full_vectors(store, manifest)
    texts = scan_full_texts(store, manifest)
    assert verify_full_text_hits(store, manifest, texts)
    # This experimental probe is separate from the production carver. Validate
    # its reconstructed full payloads with SQLite itself before using the values.
    data = (store / 'chroma.sqlite3').read_bytes()
    records, _ = scan_records(data)
    native_vectors, native_texts, templates = set(), set(), {}
    for record in records:
        payload = b''.join(data[s['offset']:s['offset'] + s['length']] for s in record['segments'])
        values = native_decode(payload, len(record['serial_types']), templates)
        kind = record['kind']
        if kind == 'embedding_metadata':
            assert values[:3] == (record['internal_id'], 'chroma:document', record['document'])
            assert values[3:] == (None, None, None)
        elif kind.endswith('_index'):
            assert values == ('chroma:document', record['document'], record['internal_id'])
        elif kind == 'fts_content_compatible':
            assert values == (None, record['document'])
        else:
            assert values[4] == record['item_id'] and values[5] == record['vector']
            assert json.loads(values[7])['chroma:document'] == record['document']
            native_vectors.add(values[5])
        native_texts.add(record['document'])

    events, nv, nt, av = [], 0, 0, 0
    api_vectors = set(api['vectors'])
    for target, vr, tr in zip(manifest['deleted'], vectors['events'], texts['events'], strict=True):
        blob = _vector(target['text'])
        rv = vr['full_residual_confirmed'] or blob in native_vectors
        rt = tr['full_text_residual_confirmed'] or target['text'] in native_texts
        grade = index.grade(target['id'], hashlib.sha256(blob).hexdigest(), target['text'])
        assert not grade['vector_recovered'] or rv, target['id']
        assert not grade['text_recovered'] or rt, target['id']
        nv += rv
        nt += rt
        av += blob.hex() in api_vectors
        grade.update(target_id=target['id'], event_type=case.name,
                     vector_hash=hashlib.sha256(blob).hexdigest(),
                     text_sha256=hashlib.sha256(target['text'].encode()).hexdigest(),
                     reference_vector=rv, reference_text=rt)
        events.append(grade)
    save_targets(report / 'recovery_targets.sqlite3', events)
    save_json(report / 'target_reference_results.json', events)
    save_json(case / 'reference.json', vectors)
    save_json(case / 'text_reference.json', texts)
    save_json(case / 'api.json', api)
    assert fingerprint(store) == before
    checks = {'evidence_sha256': before, 'evidence_unchanged': True}
    save_json(case / 'checks.json', checks)
    summary = summarize(events)
    assert not summary['unclassified_data']
    return {'deletion': case.name, 'summary': summary, 'reference_vectors': nv,
            'reference_texts': nt, 'api_vectors': av, 'report': str(report),
            'checks': checks, 'evidence_unchanged': True}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input-dir', type=Path, required=True)
    ap.add_argument('--tag', default='paper')
    ap.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument('--iterations', type=int, default=1500)
    args = ap.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or args.iterations < 1:
        ap.error('Use distinct seeds and a positive iteration count')
    version = chromadb_version()
    if version != '1.5.9':
        ap.error('The controlled paper protocol requires chromadb 1.5.9')
    root = prepare_pdf_experiment(args, 'workload_audit', version)
    from chroma_hearts.generator.document_processing import load_documents_from_dir
    ids, documents, metadata = load_documents_from_dir(args.input_dir.resolve())
    assert len(documents) == len(set(documents)), 'This protocol requires unique chunks'
    corpus = [dict(id=i, text=t, metadata=m) for i, t, m in zip(ids, documents, metadata)]
    save_json(root / 'corpus.json', corpus)
    runs = []
    for seed in args.seeds:
        plan = make_plan(len(corpus), args.iterations, seed)
        base = root / f'seed{seed}'
        save_json(base / 'operations.json', plan)
        operations = {op: sum(e['executed'] and e['operation'] == op for e in plan)
                      for op in ('add', 'delete', 'search')}
        pair = []
        for mode in ('item_delete', 'collection_drop'):
            case = base / mode
            (case / 'chroma_db').mkdir(parents=True)
            manifest = spawn_call(write_workload, str(case / 'chroma_db'), corpus, plan,
                                  mode == 'collection_drop')
            save_json(case / 'manifest.json', manifest)
            row = evaluate_case(case, manifest, base / 'reports')
            row.update(seed=seed, operations=operations)
            runs.append(row)
            pair.append(manifest)
            save_json(case / 'result.json', row)
            print('[DONE]', seed, mode, row['summary'], flush=True)
        assert set(pair[0]['live']) == {t['id'] for t in pair[1]['deleted']}
    aggregate = []
    for mode in ('item_delete', 'collection_drop'):
        group = [r for r in runs if r['deletion'] == mode]
        for metric, ref, recovered in [('vector', 'reference_vectors', 'vector_recovered'),
                                       ('original_text', 'reference_texts', 'text_recovered')]:
            n = sum(r[ref] for r in group)
            ch = sum(r['summary'][recovered] for r in group)
            api = sum(r['api_vectors'] for r in group) if metric == 'vector' else None
            aggregate.append(dict(deletion=mode, metric=metric, targets=n, chroma_hearts=ch,
                                  get_items=api, chroma_hearts_recall=ch/n if n else None,
                                  get_items_recall=api/n if api is not None and n else None))
    result = dict(tag=args.tag, root=str(root), seeds=args.seeds, iterations=args.iterations,
                  paired_plan=True, no_reinsertions=True, runs=runs, aggregate=aggregate)
    output = RESULTS_DIR / f'workload_audit_{version}_{args.tag}.json'
    save_json(output, result)
    print('[RESULT]', output, json.dumps(aggregate, indent=2), flush=True)


if __name__ == '__main__':
    main()
