"""Whole original chunks, partial matches, and active-copy ambiguity."""
import json
import random
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from _audit_full_text import find_full_text, scan_full_texts, verify_full_text_hits


class FullTextTests(unittest.TestCase):
    def make_store(self, root):
        with sqlite3.connect(root / 'chroma.sqlite3') as conn:
            conn.execute('CREATE TABLE embeddings_queue (seq_id INTEGER, id TEXT, metadata TEXT)')
            conn.execute('CREATE TABLE embeddings (id INTEGER, embedding_id TEXT)')
            conn.execute('CREATE TABLE embedding_metadata (id INTEGER, key TEXT, string_value TEXT)')

    def test_partial_and_cross_file_matches_do_not_count(self):
        text = 'A distinct complete chunk ' + ''.join(random.Random(2).choices('abcdef0123456789', k=800))
        data = text.encode()
        self.assertEqual(find_full_text({'first': data[:400], 'second': data[400:]}, text), [])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_store(root)
            (root / 'partial.bin').write_bytes(data[:400])
            manifest = {'live': {}, 'deleted': [{'id': 'gone', 'text': text}]}
            event = scan_full_texts(root, manifest)['events'][0]
            self.assertFalse(event['full_text_residual_confirmed'])
            self.assertTrue(event['partial_only'])

    def test_unicode_json_and_active_overlap(self):
        text = '한국어 "문서"\n원문\\end'
        for ascii_only in (False, True):
            data = json.dumps(text, ensure_ascii=ascii_only).encode()
            hits = find_full_text({'json': data}, text)
            self.assertTrue(hits)
            self.assertEqual(hits[0]['offset'], 0)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_store(root)
            (root / 'whole.bin').write_bytes(text.encode())
            manifest = {'live': {'active': {'text': 'prefix ' + text}},
                        'deleted': [{'id': 'gone', 'text': text}]}
            result = scan_full_texts(root, manifest)
            self.assertFalse(result['events'][0]['full_text_residual_confirmed'])
            self.assertTrue(result['events'][0]['ambiguous_active_overlap'])
            with sqlite3.connect(root / 'chroma.sqlite3') as conn:
                conn.execute('INSERT INTO embeddings_queue VALUES (1,?,?)',
                             ('gone', json.dumps({'chroma:document': text})))
            result = scan_full_texts(root, manifest)
            self.assertTrue(result['events'][0]['full_text_residual_confirmed'])
            self.assertGreater(verify_full_text_hits(root, manifest, result), 0)

    def test_sqlite_complete_value_can_span_pages(self):
        text = ''.join(random.Random(7).choices('abcdef0123456789', k=20000))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_store(root)
            with sqlite3.connect(root / 'chroma.sqlite3') as conn:
                conn.execute('INSERT INTO embeddings_queue VALUES (1,?,?)',
                             ('gone', json.dumps({'chroma:document': text})))
            manifest = {'live': {}, 'deleted': [{'id': 'gone', 'text': text}]}
            result = scan_full_texts(root, manifest)
            self.assertEqual(result['events'][0]['raw_full_hits'], [])
            self.assertTrue(result['events'][0]['full_text_residual_confirmed'])
            self.assertEqual(verify_full_text_hits(root, manifest, result), 1)


if __name__ == '__main__':
    unittest.main()
