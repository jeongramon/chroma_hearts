"""Regression tests for SDK writes during forensic recovery.

Run with: python3 -m unittest discover -s tests -v
"""

from argparse import Namespace
import hashlib
from importlib import import_module
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from chroma_hearts.parser import sdk_extractor

parser_main = import_module("chroma_hearts.parser.main")


def snapshot(directory: Path) -> dict[str, tuple[str, int]]:
    """Capture file names, contents, and modification times without opening DBs."""
    return {
        str(path.relative_to(directory)): (
            hashlib.sha256(path.read_bytes()).hexdigest(),
            path.stat().st_mtime_ns,
        )
        for path in directory.rglob("*")
        if path.is_file()
    }


class SDKIsolationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "run" / "chroma_db"
        self.source.mkdir(parents=True)
        with sqlite3.connect(self.source / "chroma.sqlite3") as conn:
            for name in sdk_extractor.SYSDB_TABLES:
                columns = "id TEXT, collection TEXT" if name == "segments" else "id TEXT"
                conn.execute(f"CREATE TABLE {name} ({columns})")
            conn.executescript("""
                CREATE TABLE embeddings (
                    id INTEGER, embedding_id TEXT, seq_id BLOB,
                    created_at TIMESTAMP, segment_id TEXT
                );
                CREATE TABLE embeddings_queue (id TEXT);
                INSERT INTO embeddings_queue VALUES ('deleted-vector');
            """)
        segment = self.source / "segment"
        segment.mkdir()
        (segment / "data_level0.bin").write_bytes(b"deleted HNSW vector")
        (segment / "index_metadata.pickle").write_bytes(b"captured label map")
        self.original = snapshot(self.source)
        self.sdk_paths: list[Path] = []

    def mutating_client(self, *, path: str, settings: object) -> SimpleNamespace:
        """Simulate SDK initialization purging queue rows and rewriting HNSW."""
        sdk_path = Path(path)
        self.sdk_paths.append(sdk_path)
        with sqlite3.connect(sdk_path / "chroma.sqlite3") as conn:
            conn.execute("DELETE FROM embeddings_queue")
        (sdk_path / "segment" / "data_level0.bin").write_bytes(b"SDK rewrote index")
        (sdk_path / "segment" / "index_metadata.pickle").unlink()
        (sdk_path / "sdk-created-file").write_text("new artifact")
        return SimpleNamespace(list_collections=lambda: [])

    def assert_source_preserved_and_copies_removed(self) -> None:
        self.assertEqual(snapshot(self.source), self.original)
        self.assertTrue(self.sdk_paths)
        for path in self.sdk_paths:
            self.assertNotEqual(path, self.source)
            self.assertFalse(path.parent.exists(), f"Temporary copy remains: {path}")

    def test_public_entry_points_preserve_supplied_store(self) -> None:
        for extract in (
            sdk_extractor.extract_via_sdk,
            sdk_extractor.extract_via_sdk_from_copy,
        ):
            with self.subTest(entry_point=extract.__name__):
                report = self.root / extract.__name__
                with patch.object(
                    sdk_extractor.chromadb, "PersistentClient", side_effect=self.mutating_client
                ):
                    self.assertEqual(extract(self.source, report), 0)
                self.assertTrue((report / "sysdb.sqlite3").is_file())
                self.assertTrue((report / "embeddings.sqlite3").is_file())
                self.assert_source_preserved_and_copies_removed()

    def test_sdk_failure_preserves_source_and_removes_copy(self) -> None:
        def failing_client(**kwargs: object) -> None:
            self.mutating_client(**kwargs)
            raise RuntimeError("SDK initialization failed")

        with patch.object(sdk_extractor.chromadb, "PersistentClient", side_effect=failing_client):
            with self.assertRaisesRegex(RuntimeError, "SDK initialization failed"):
                sdk_extractor.extract_via_sdk(self.source, self.root / "report")
        self.assert_source_preserved_and_copies_removed()

    def test_pipeline_preserves_artifacts_for_wal_and_hnsw(self) -> None:
        recovery_paths: list[Path] = []

        def inspect_artifacts(source: Path, output_dir: Path) -> dict:
            recovery_paths.append(source)
            self.assertNotEqual(source, self.source)
            self.assertNotIn(source, self.sdk_paths)
            self.assertEqual(snapshot(source), self.original)
            self.assertTrue((output_dir / "embeddings.sqlite3").exists())
            return {'unreferenced_segment_dirs': []}

        report = self.root / "report"
        args = Namespace(
            chroma_db_path=self.source,
            chroma_run_dir=self.source.parent,
            output_dir=report,
            report_base=report.parent,
            run_label="run",
        )
        with (
            patch.object(sdk_extractor.chromadb, "PersistentClient", side_effect=self.mutating_client),
            patch.object(parser_main, "merge_wal_from_copy", side_effect=inspect_artifacts),
            patch.object(parser_main, "extract_hnsw_from_copy", side_effect=inspect_artifacts),
            patch.object(parser_main, "report_persistence_status", return_value={
                'collections': 0, 'target_sqlite': str(report / 'persistence.sqlite3'), 'rows': [],
            }),
        ):
            parser_main.main(args)
        self.assertEqual(len(recovery_paths), 2)
        self.assertTrue(all(not path.parent.exists() for path in recovery_paths))
        self.assert_source_preserved_and_copies_removed()


if __name__ == "__main__":
    unittest.main()
