"""Four-path priority, full-text-only recovery, provenance, and active copies."""
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from chroma_hearts.parser.sdk_extractor import EMBEDDINGS_TABLE_SQL
from chroma_hearts.parser.recovery_store import ensure_schema, add_evidence
from chroma_hearts.parser.sqlite_extractor import extract_sqlite_carved
from chroma_hearts.evaluator.recovery_paths import RecoveryIndex, summarize
from chroma_hearts.evaluator.main import _expected_blob, evaluate_run_verified, save_verified_metrics
from test_sqlite_record_probe import fixture


class RecoveryPathTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.db = self.root / 'embeddings.sqlite3'
        with sqlite3.connect(self.db) as conn:
            conn.execute(EMBEDDINGS_TABLE_SQL)
            ensure_schema(conn)

    def insert(self, method, item_id, text, *, vector=True, document=True, collection='collection'):
        blob = _expected_blob(text, 8) if vector else None
        meta = json.dumps({'chroma:document': text}) if document else '{}'
        flag = {'sdk': 'source_chroma_sdk', 'wal': 'source_wal', 'hnsw': 'source_hnsw',
                'sqlite_carving': 'source_sqlite_carved'}[method]
        with sqlite3.connect(self.db) as conn:
            rowid = conn.execute(f'INSERT INTO embeddings(id,collection_id,vector,metadata,{flag}) VALUES (?,?,?,?,1)',
                                 (item_id, collection, blob, meta)).lastrowid
            add_evidence(conn, rowid, method, item_id=item_id, collection_id=collection, vector=blob, metadata=meta)

    def grade(self, target, text, collection=None):
        return RecoveryIndex(self.db).grade(target, hashlib.sha256(_expected_blob(text,8)).hexdigest(), text, collection)

    def test_hnsw_orphan_keeps_priority_over_complete_carved_wal(self):
        self.insert('hnsw', None, 'orphan document', document=False)
        self.insert('sqlite_carving', 'deleted', 'orphan document')
        result = self.grade('deleted', 'orphan document')
        self.assertEqual(result['representative_path'], 'orphan_vector')
        self.assertTrue(result['text_recovered'])
        self.assertTrue(result['id_recovered'])
        self.assertEqual(result['source_paths'], ['hnsw', 'sqlite_carving'])
        self.assertEqual(len(result['matched_evidence_ids']), 2)

    def test_full_priority_and_text_only_counts_are_separate(self):
        self.insert('wal', 'full', 'wal document')
        self.insert('hnsw', 'full', 'wal document', document=False)
        self.insert('sqlite_carving', 'full', 'wal document')
        self.insert('hnsw', 'identified', 'mapped document', document=False)
        self.insert('sqlite_carving', None, 'mapped document', vector=False)
        self.insert('sqlite_carving', None, 'text only document', vector=False)
        rows = [self.grade('full','wal document'), self.grade('identified','mapped document'),
                self.grade('text','text only document')]
        self.assertEqual([r['representative_path'] for r in rows],
                         ['wal_full_entry','identified_vector','sqlite_carved_data'])
        summary = summarize(rows)
        self.assertEqual(summary['vector_recovered'], 2)
        self.assertEqual(summary['text_recovered'], 3)
        self.assertEqual(summary['targets'], 3)

    def test_active_copy_and_wrong_id_cannot_be_deleted_evidence(self):
        self.insert('sdk', 'live', 'shared content')
        self.insert('sqlite_carving', None, 'shared content', vector=False)
        self.insert('wal', 'other-id', 'target content')
        self.assertEqual(self.grade('deleted','shared content')['representative_path'], 'none')
        self.assertEqual(self.grade('deleted','target content')['representative_path'], 'none')
        self.insert('sqlite_carving', 'deleted', 'shared content')
        self.assertEqual(self.grade('deleted','shared content')['representative_path'], 'sqlite_carved_data')

    def test_collection_identity_does_not_collapse_equal_ids(self):
        self.insert('sdk', 'same-id', 'shared', collection='live')
        self.insert('sqlite_carving', 'same-id', 'shared', collection='dropped')
        self.assertEqual(self.grade('same-id','shared','dropped')['representative_path'], 'sqlite_carved_data')
        self.assertEqual(self.grade('same-id','shared','live')['representative_path'], 'none')

    def test_carver_keeps_unknown_ids_null_and_records_offsets_without_header_repair(self):
        source = self.root / 'source'; source.mkdir()
        data, pos = fixture('original document ' * 50, prefix=False)
        (source/'chroma.sqlite3').write_bytes(data)
        result = extract_sqlite_carved(source, self.root)
        self.assertEqual(result['records'], 1)
        self.assertEqual((source/'chroma.sqlite3').read_bytes(),data)
        with sqlite3.connect(self.db) as conn:
            row = conn.execute('SELECT id,vector,source_wal,source_sqlite_carved FROM embeddings').fetchone()
            self.assertEqual(row,(None,None,0,1))
            self.assertEqual(conn.execute('SELECT record_offset FROM recovery_evidence').fetchone()[0],pos)
        corrupt = bytearray(data); corrupt[pos] = 0
        (source/'chroma.sqlite3').write_bytes(corrupt)
        self.assertEqual(extract_sqlite_carved(source,self.root)['records'],0)

    def test_verified_evaluator_partitions_events_and_persists_all_sources(self):
        targets = [('a','first document'),('b','second document'),('c','third document'),('d','absent')]
        self.insert('wal','a',targets[0][1])
        self.insert('hnsw',None,targets[1][1],document=False)
        self.insert('sqlite_carving','b',targets[1][1])
        self.insert('sqlite_carving',None,targets[2][1],vector=False)
        ops = self.root/'operations.sqlite3'
        with sqlite3.connect(ops) as conn:
            conn.execute('CREATE TABLE operations (iteration INTEGER,target_collection TEXT,operation TEXT,result TEXT,doc_id TEXT,document TEXT)')
            for i,(item,text) in enumerate(targets):
                conn.execute('INSERT INTO operations VALUES (?,?,?,?,?,?)',(i*2,'collection','add','1',item,text))
                conn.execute('INSERT INTO operations VALUES (?,?,?,?,?,?)',(i*2+1,'collection','delete','1',item,None))
        m=evaluate_run_verified(ops,self.db,'test','hash',8)
        self.assertEqual((m.delete_event_full_chunk,m.delete_event_identified_only,m.delete_event_orphan_only,
                          m.delete_event_sqlite_carved,m.delete_event_unrecovered),(1,0,1,1,1))
        self.assertEqual(m.delete_event_vector_recovered,2)
        self.assertEqual(m.delete_event_text_recovered,3)
        save_verified_metrics(m,self.root/'evaluation.sqlite3')
        with sqlite3.connect(self.root/'recovery_targets.sqlite3') as conn:
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM target_recovery').fetchone()[0],4)


if __name__ == '__main__':
    unittest.main()
