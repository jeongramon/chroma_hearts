"""Whole-vector reference checks: partial bytes, file bounds and active duplicates."""
import sqlite3
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from _audit_full_scan import classify_hnsw, scan_full_vectors
from _experiment_common import _vector


class FullScanTests(unittest.TestCase):
    def test_full_vector_vs_partial_and_active_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with sqlite3.connect(root/'chroma.sqlite3') as conn:
                conn.execute('CREATE TABLE embeddings_queue (id TEXT, vector BLOB)')
            text = 'known PDF chunk used for full byte verification'
            vector = _vector(text)
            (root/'part_a.bin').write_bytes(vector[:768])
            (root/'part_b.bin').write_bytes(vector[768:])
            manifest = {'live': {}, 'deleted': [{'id':'gone', 'text':text, 'category':'test'}]}
            result = scan_full_vectors(root,manifest)['events'][0]
            self.assertFalse(result['raw_full_presence'])
            self.assertTrue(result['partial_vector_only'])
            (root/'whole.bin').write_bytes(b'prefix'+vector+b'suffix')
            result = scan_full_vectors(root,manifest)['events'][0]
            self.assertTrue(result['full_residual_confirmed'])
            self.assertEqual(result['raw_full_hits'],[{'file':'whole.bin','offset':6,'kind':'raw_full'}])
            manifest['live']['active'] = {'text':text}
            result = scan_full_vectors(root,manifest)['events'][0]
            self.assertFalse(result['full_residual_confirmed'])
            self.assertTrue(result['ambiguous_active_only'])

    def test_occupied_inactive_vs_active_and_unused_hnsw(self):
        header = bytearray(64)
        struct.pack_into('<Q',header,4,0)
        struct.pack_into('<QQQQ',header,0x14,1,20,12,4)
        needle = b'vector!!'
        block = b'HEAD'+needle+struct.pack('<Q',9)
        data = block+block
        self.assertEqual(classify_hnsw(data,header,4,needle,{9:'live'},{'live':needle}), 'hnsw_active')
        self.assertEqual(classify_hnsw(data,header,4,needle,{}, {'live':needle}), 'hnsw_inactive')
        self.assertEqual(classify_hnsw(data,header,24,needle,{},{}), 'hnsw_unused_or_unaligned')
        self.assertEqual(classify_hnsw(data,header,5,needle,{},{}), 'hnsw_unused_or_unaligned')


if __name__ == '__main__':
    unittest.main()
