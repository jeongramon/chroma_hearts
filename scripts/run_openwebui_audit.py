"""Verify the knowledge-base deletion cases with declared cosine normalization.

Parse without deleted targets. Historical segment metrics can be supplied for
orphan directories; target lists enter only the subsequent evaluation.
"""
import argparse
from pathlib import Path
import hashlib
import json
import shutil
import sqlite3
import sys

from _snapshots import run_recovery_pipeline, extract_before_snapshot
from _experiment_common import spawn_call
from _experiment_common import fingerprint
from chroma_hearts.parser.hnsw_extractor import load_segment_spaces
from chroma_hearts.parser.source_copy import working_copy
from chroma_hearts.evaluator.recovery_paths import PATHS, RecoveryIndex, save_targets, summarize

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT/'sample/openwebui_knowledge_base'
OUTPUT = ROOT/'sample/experiments/openwebui_paper'
KB_NAME = 'd32554c5-316d-4074-a93c-dd6fd09b3d0c'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input-dir', type=Path, default=SOURCE)
    ap.add_argument('--output-dir', type=Path, default=OUTPUT)
    ap.add_argument('--result', type=Path, default=ROOT/'sample/experiment_results/openwebui_paper.json')
    ap.add_argument('--collection-name', default=KB_NAME)
    args = ap.parse_args()
    source_dir, output_dir = args.input_dir.resolve(), args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    before_snapshot = output_dir/'before_snapshot.sqlite3'
    spawn_call(extract_before_snapshot, source_dir/'baseline_add_only/vector_db', before_snapshot)
    before_store=source_dir/'baseline_add_only/vector_db'
    hashes={mode:fingerprint(source_dir/mode/'vector_db') for mode in ('baseline_add_only','delete_half','delete_all')}
    with working_copy(before_store,prefix='case_before_metadata') as cp:
        spaces={directory.name: 'cosine' for mode in ('baseline_add_only','delete_half','delete_all')
                for directory in (source_dir/mode/'vector_db').iterdir() if (directory/'header.bin').exists()}
        with sqlite3.connect(f'file:{cp/"chroma.sqlite3"}?mode=ro',uri=True) as c:
            kb_id=c.execute('SELECT id FROM collections WHERE name=?',(args.collection_name,)).fetchone()[0]
            kb_segments=[r[0] for r in c.execute('SELECT id FROM segments WHERE collection=?',(kb_id,))]
            before_pk={pk:(item,doc) for pk,item,doc in c.execute(
                "SELECT e.id,e.embedding_id,m.string_value FROM embeddings e JOIN segments s ON s.id=e.segment_id JOIN embedding_metadata m ON m.id=e.id AND m.key='chroma:document' WHERE s.collection=?",(kb_id,))}
            before_wal={item:(bytes(vector),json.loads(metadata or '{}').get('chroma:document'))
                        for item,vector,metadata in c.execute(
                            'SELECT id,vector,metadata FROM embeddings_queue WHERE vector IS NOT NULL AND topic LIKE ?',('%'+kb_id,))}
    with sqlite3.connect(f'file:{before_snapshot}?mode=ro',uri=True) as c:
        c.row_factory=sqlite3.Row
        targets=[dict(r) for r in c.execute('SELECT * FROM before_snapshot WHERE collection_id=?',(kb_id,))]
    assert len(targets)==2434
    assert len({r['document'] for r in targets})==2434
    runs=[]
    for mode in ('delete_half','delete_all'):
        after=output_dir/mode
        if not (after/'verified.json').exists():
            run_recovery_pipeline(source_dir/mode/'vector_db',after,segment_spaces=spaces)
        index=RecoveryIndex(after/'embeddings.sqlite3')
        # Evaluation-only source attribution. Keep unknown Chroma IDs unknown;
        # a preserved SQLite PK and full text identify the before-record's
        # collection, provided that PK is not reused by an active after-record.
        with working_copy(source_dir/mode/'vector_db',prefix='case_after_metadata') as cp:
            with sqlite3.connect(f'file:{cp/"chroma.sqlite3"}?mode=ro',uri=True) as c:
                active_pks={r[0] for r in c.execute('SELECT id FROM embeddings')}
        with sqlite3.connect(f'file:{after/"embeddings.sqlite3"}?mode=ro',uri=True) as c:
            internal_ids=dict(c.execute('SELECT rowid,sqlite_embedding_pk FROM embeddings WHERE source_sqlite_carved=1'))
        attribution=[]
        for row in index.rows:
            if row['method']!='sqlite_carving' or row['collection_id'] is not None:
                continue
            pk=internal_ids.get(row['embedding_rowid'])
            ref=before_pk.get(pk)
            if ref and pk not in active_pks and row['document']==ref[1]:
                row['collection_id']=kb_id
                attribution.append({'evidence_id':row['evidence_id'],'sqlite_embedding_pk':pk,
                                    'reference_target_id':ref[0],'collection_id':kb_id})
        (after/'source_attribution.json').write_text(json.dumps(attribution,indent=2)+'\n')
        unbound={row['evidence_id'] for row in index.rows
                 if row['method']=='sqlite_carving' and row['collection_id'] is None}
        for lookup in (index.by_vector,index.by_text):
            for key in lookup:
                lookup[key]=[row for row in lookup[key] if row['evidence_id'] not in unbound]
        (after/'unattributed_carved_evidence.json').write_text(json.dumps(sorted(unbound))+'\n')
        active={(r['collection_id'],r['id']) for r in index.rows if r['method']=='sdk'}
        deleted=[t for t in targets if (kb_id,t['id']) not in active]
        details=[]
        for t in deleted:
            hashes_for_target={t['vector_hash']}
            if t['id'] in before_wal:
                vector,document=before_wal[t['id']]
                assert document==t['document']
                hashes_for_target.add(hashlib.sha256(vector).hexdigest())
            grades=[index.grade(t['id'],vh,t['document'],kb_id) for vh in hashes_for_target]
            order=(*PATHS,'none')
            selected=min((x['representative_path'] for x in grades),key=order.index)
            g={'representative_path':selected,
               **{k:any(x[k] for x in grades) for k in ('vector_recovered','text_recovered','id_recovered')},
               **{k:sorted({v for x in grades for v in x[k]}) for k in
                  ('source_paths','matched_evidence_ids','excluded_active_or_ambiguous_evidence_ids')},
               'unclassified_data':selected=='none' and any(x['source_paths'] for x in grades),
               'reference_vector_hashes':sorted(hashes_for_target)}
            g.update(event_type=mode,target_id=t['id'],collection_id=kb_id,vector_hash=t['vector_hash'],
                     text_sha256=hashlib.sha256(t['document'].encode()).hexdigest())
            details.append(g)
        save_targets(after/'recovery_targets.sqlite3',details)
        (after/'target_results.json').write_text(json.dumps(details,indent=2)+'\n')
        counts=summarize(details)
        result={'deletion':mode,**counts,'knowledge_collection_id':kb_id,'report':str(after),
                'active_knowledge_items':sum(c==kb_id for c,_ in active),
                'scope_linked_carved_records':len(attribution),
                'unattributed_carved_records_excluded':len(unbound),
                'metric':'Exact full-vector bytes against before-SDK and before-WAL representations; stored cosine norms restored. Exact full document.',
                'evidence_unchanged':fingerprint(source_dir/mode/'vector_db')==hashes[mode]}
        assert result['evidence_unchanged']
        (after/'verified.json').write_text(json.dumps(result,indent=2)+'\n')
        runs.append(result)
        print('RESULT',json.dumps(result),flush=True)
    assert fingerprint(before_store)==hashes['baseline_add_only']
    output={'runs':runs,'declared_segment_spaces':spaces,'before_kb_segments':kb_segments,
            'source_hashes':hashes,'code_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [Path(__file__).resolve(),ROOT/'src/chroma_hearts/parser/hnsw_extractor.py',
                          ROOT/'src/chroma_hearts/evaluator/recovery_paths.py']}}
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(output,indent=2)+'\n')


if __name__=='__main__':
    main()
