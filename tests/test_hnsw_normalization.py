"""Stored cosine norms restore vectors without changing raw evidence or L2 data."""
import json
from pathlib import Path
import pickle
import sqlite3
import struct
import tempfile
import unittest

from chroma_hearts.parser.hnsw_extractor import extract_hnsw_from_copy
from chroma_hearts.parser.sdk_extractor import EMBEDDINGS_TABLE_SQL


class HnswNormalizationTests(unittest.TestCase):
    def run_fixture(self, space, orphan=False, supplied=None):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); source=root/'source'; report=root/'report'
            segment=source/'segment'; segment.mkdir(parents=True); report.mkdir()
            header=bytearray(0x5c)
            for off,fmt,value in [(0,'I',1),(4,'Q',0),(12,'Q',1),(20,'Q',1),
                                  (28,'Q',28),(36,'Q',20),(44,'Q',12),
                                  (52,'I',0),(56,'I',0),(60,'Q',1),(68,'Q',2),
                                  (76,'d',1.0),(84,'Q',100)]:
                struct.pack_into('<'+fmt,header,off,value)
            (segment/'header.bin').write_bytes(header)
            raw=struct.pack('<ff',0.6,0.8)
            block=bytearray(28);block[12:20]=raw;struct.pack_into('<Q',block,20,19)
            (segment/'data_level0.bin').write_bytes(block)
            (segment/'length.bin').write_bytes(struct.pack('<f',5.0))
            (segment/'index_metadata.pickle').write_bytes(pickle.dumps({'label_to_id':{19:'item'}}))
            with sqlite3.connect(report/'sysdb.sqlite3') as c:
                c.execute('CREATE TABLE collections(id TEXT,name TEXT,config_json_str TEXT)')
                c.execute('CREATE TABLE segments(id TEXT,collection TEXT)')
                if not orphan:
                    c.execute('INSERT INTO collections VALUES (?,?,?)',('collection','name',json.dumps({'hnsw':{'space':space}})))
                    c.execute('INSERT INTO segments VALUES (?,?)',('segment','collection'))
            with sqlite3.connect(report/'embeddings.sqlite3') as c:
                c.execute(EMBEDDINGS_TABLE_SQL)
            extract_hnsw_from_copy(source,report,segment_spaces=supplied)
            with sqlite3.connect(report/'embeddings.sqlite3') as c:
                recovered=c.execute('SELECT vector FROM embeddings').fetchone()[0]
                vector,details=c.execute('SELECT vector,details FROM recovery_evidence').fetchone()
            self.assertEqual((segment/'data_level0.bin').read_bytes(),block)
            self.assertEqual(recovered,vector)
            return raw,recovered,json.loads(details)

    def test_cosine_uses_element_index_norm_and_keeps_raw_bytes(self):
        raw,recovered,details=self.run_fixture('cosine')
        self.assertEqual(recovered,struct.pack('<ff',3.0,4.0))
        self.assertEqual(details['raw_vector_hex'],raw.hex())
        self.assertEqual(details['norm_byte_offset'],0)
        self.assertEqual(details['hnsw_label'],19)

    def test_l2_does_not_treat_length_file_as_norms(self):
        raw,recovered,details=self.run_fixture('l2')
        self.assertEqual(raw,recovered)
        self.assertNotIn('original_norm',details)

    def test_unknown_orphan_metric_is_not_guessed(self):
        raw,recovered,details=self.run_fixture(None,orphan=True)
        self.assertEqual(raw,recovered)
        self.assertNotIn('original_norm',details)

    def test_orphan_can_use_explicit_historical_metric(self):
        raw,recovered,details=self.run_fixture(None,orphan=True,supplied={'segment':'cosine'})
        self.assertEqual(recovered,struct.pack('<ff',3.0,4.0))
        self.assertEqual(details['space'],'cosine')


if __name__=='__main__':
    unittest.main()
