"""Random plans, independent form labels, and the mapping-free library baseline."""
import json
import pickle
import sqlite3
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from run_workload_audit import make_plan, extract_api
from _experiment_common import fingerprint
from _experiment_common import _vector


class WorkloadAuditTests(unittest.TestCase):
    def test_paired_plan_has_unique_adds_and_only_deletes_active_items(self):
        for seed in range(1, 6):
            plan = make_plan(1161, 1500, seed)
            self.assertEqual(plan, make_plan(1161, 1500, seed))
            added, active = set(), set()
            for event in plan:
                if not event["executed"]:
                    continue
                item = event["item"]
                if event["operation"] == "add":
                    self.assertNotIn(item, added)
                    added.add(item)
                    active.add(item)
                elif event["operation"] == "delete":
                    self.assertIn(item, active)
                    active.remove(item)
            self.assertLessEqual(len(added), 1161)
        with self.assertRaises(ValueError):
            make_plan(1, 100, 1)


    def test_library_works_without_sqlite_or_readable_id_mapping(self):
        import hnswlib
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp)
            directory = store / "segment"
            directory.mkdir()
            index = hnswlib.Index(space="l2", dim=384)
            index.init_index(max_elements=3, is_persistent_index=True, persistence_location=str(directory))
            matrix = np.zeros((3, 384), dtype=np.float32)
            matrix[0, 0], matrix[1, 1], matrix[2, 2] = 1, 1, 1
            index.add_items(matrix, [10, 11, 12])
            index.mark_deleted(12)
            index.persist_dirty()
            del index
            (directory / "index_metadata.pickle").write_bytes(b"This is deliberately not a pickle")
            before = fingerprint(store)
            result = extract_api(str(store))
            self.assertEqual(set(result["vectors"]), {v.tobytes().hex() for v in matrix[:2]})
            self.assertEqual(len(result["library_errors"]), 1)
            self.assertFalse(result["id_mapping_used"])
            self.assertEqual(result["mapping_files_removed_from_copy"], 1)
            self.assertEqual(fingerprint(store), before)


if __name__ == "__main__":
    unittest.main()
