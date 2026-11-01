"""SQLite-backed operation log for sample ChromaDB generation."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


OPERATIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS operations (
    iteration INTEGER NOT NULL,
    target_collection TEXT NOT NULL,
    operation TEXT NOT NULL,
    result TEXT NOT NULL,
    collection_count INTEGER NOT NULL,
    doc_id TEXT,
    document TEXT
);
CREATE INDEX IF NOT EXISTS idx_operations_iteration ON operations(iteration);
CREATE INDEX IF NOT EXISTS idx_operations_operation ON operations(operation);
CREATE INDEX IF NOT EXISTS idx_operations_doc_id ON operations(doc_id);
CREATE TABLE IF NOT EXISTS run_config (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class OperationLog:
    """Persist every generator operation for later recovery-rate analysis."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path.resolve()
        self.conn = sqlite3.connect(self.db_path)
        self.conn.executescript(OPERATIONS_SCHEMA)

    def record(
        self,
        iteration: int,
        target_collection: str,
        operation: str,
        result: Any,
        collection_count: int,
        doc_id: str | None = None,
        document: str | None = None,
    ) -> None:
        """Append one operation row.

        ``doc_id`` / ``document`` capture the affected Chroma id and the exact
        document text embedded for add/update/upsert (delete records only the
        id). This is the ground truth the evaluator replays to reconstruct and
        byte-verify recovered vectors under hash embeddings.
        """
        self.conn.execute(
            """
            INSERT INTO operations (
                iteration, target_collection, operation, result, collection_count,
                doc_id, document
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                iteration,
                target_collection,
                operation,
                str(result),
                collection_count,
                doc_id,
                document,
            ),
        )

    def record_config(self, key: str, value: Any) -> None:
        """Record one run setting (e.g. the sync_threshold collections were created with)."""
        self.conn.execute(
            "INSERT OR REPLACE INTO run_config (key, value) VALUES (?, ?)", (key, str(value))
        )

    def close(self) -> None:
        """Commit and close the database connection."""
        self.conn.commit()
        self.conn.close()

    def __enter__(self) -> OperationLog:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
