"""
Persistence status (pipeline step 4): how far each collection is from its next
HNSW persist.

Full Entry and Identified Vector residue lasts until the vector segment next
persists. Chroma persists once the log records applied since the last persist
reach the collection's ``sync_threshold``, checked when a write call (or the
first read's backfill) finishes. Persistence is driven by record count, not
time, so this step reports the pending state of every collection in the
snapshot: how many records wait beyond the vector segment's ``max_seq_id``,
how old they are, and how many more records the collection needs before the
next persist clears them.

No operation log is needed, so this also works on stores from real
deployments.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

DEFAULT_SYNC_THRESHOLD = 1000
_SYNC_THRESHOLD_RE = re.compile(r'"sync_threshold"\s*:\s*(\d+)')

OPERATION_NAMES = {0: "add", 1: "update", 2: "upsert", 3: "delete"}


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _as_int(value: Any) -> int:
    """seq_id values are integers, or big-endian bytes in older schemas."""
    if value is None:
        return 0
    if isinstance(value, (bytes, bytearray)):
        return int.from_bytes(value, "big")
    return int(value)


def collection_sync_thresholds(conn: sqlite3.Connection) -> dict[str, tuple[str, int]]:
    """Map collection id -> (name, sync_threshold) from a sysdb ``collections`` table.

    The threshold is read from ``schema_str`` (1.x), then ``config_json_str``,
    then legacy ``hnsw:sync_threshold`` collection metadata, else Chroma's default.
    """
    columns = _columns(conn, "collections")
    config_columns = [c for c in ("schema_str", "config_json_str") if c in columns]
    select = ", ".join(["id", "name", *config_columns])
    legacy: dict[str, int] = {}
    if "collection_metadata" in {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}:
        for collection_id, value in conn.execute(
            "SELECT collection_id, int_value FROM collection_metadata WHERE key = 'hnsw:sync_threshold'"
        ):
            if value is not None:
                legacy[str(collection_id)] = int(value)

    result: dict[str, tuple[str, int]] = {}
    for row in conn.execute(f"SELECT {select} FROM collections"):
        collection_id, name, *configs = row
        threshold = None
        for config in configs:
            match = _SYNC_THRESHOLD_RE.search(config or "")
            if match:
                threshold = int(match.group(1))
                break
        if threshold is None:
            threshold = legacy.get(str(collection_id), DEFAULT_SYNC_THRESHOLD)
        result[str(collection_id)] = (str(name), threshold)
    return result


PERSISTENCE_TABLE_SQL = """
CREATE TABLE persistence_status (
    collection_id TEXT PRIMARY KEY,
    collection_name TEXT,
    vector_segment_id TEXT,
    sync_threshold INTEGER,
    vector_max_seq_id INTEGER,
    metadata_max_seq_id INTEGER,
    wal_rows INTEGER,
    pending_records INTEGER,
    pending_by_operation TEXT,
    records_to_next_persist INTEGER,
    oldest_pending_created_at TEXT,
    newest_pending_created_at TEXT
);
"""


def report_persistence_status(source_chroma_db_path: Path, output_dir: Path) -> dict[str, Any]:
    """Write ``persistence.sqlite3`` for a working copy of a persisted store.

    ``source_chroma_db_path`` must already be a working copy; the database is
    only read.
    """
    source_sqlite = Path(source_chroma_db_path).resolve() / "chroma.sqlite3"
    target = Path(output_dir).resolve() / "persistence.sqlite3"
    if target.exists():
        target.unlink()

    src = sqlite3.connect(f"file:{source_sqlite}?mode=ro", uri=True)
    try:
        thresholds = collection_sync_thresholds(src)
        segments: dict[str, dict[str, str]] = {}
        for segment_id, collection_id, scope in src.execute(
            "SELECT id, collection, scope FROM segments"
        ):
            segments.setdefault(str(collection_id), {})[str(scope)] = str(segment_id)
        max_seq = {str(s): _as_int(v) for s, v in src.execute("SELECT segment_id, seq_id FROM max_seq_id")}

        rows = []
        for collection_id, (name, threshold) in thresholds.items():
            vector_segment = segments.get(collection_id, {}).get("VECTOR")
            metadata_segment = segments.get(collection_id, {}).get("METADATA")
            vector_max = max_seq.get(vector_segment or "", 0)
            metadata_max = max_seq.get(metadata_segment or "", 0)
            topic_like = f"%/{collection_id}"
            wal_rows = src.execute(
                "SELECT COUNT(*) FROM embeddings_queue WHERE topic LIKE ?", (topic_like,)
            ).fetchone()[0]
            pending = src.execute(
                """
                SELECT operation, COUNT(*), MIN(created_at), MAX(created_at)
                FROM embeddings_queue
                WHERE topic LIKE ? AND seq_id > ?
                GROUP BY operation
                """,
                (topic_like, vector_max),
            ).fetchall()
            by_operation = {OPERATION_NAMES.get(int(op), str(op)): n for op, n, _, _ in pending}
            pending_records = sum(by_operation.values())
            oldest = min((r[2] for r in pending if r[2]), default=None)
            newest = max((r[3] for r in pending if r[3]), default=None)
            rows.append(
                (
                    collection_id,
                    name,
                    vector_segment,
                    threshold,
                    vector_max,
                    metadata_max,
                    wal_rows,
                    pending_records,
                    json.dumps(by_operation, sort_keys=True),
                    max(threshold - pending_records, 0),
                    oldest,
                    newest,
                )
            )
    finally:
        src.close()

    with sqlite3.connect(target) as dst:
        dst.executescript(PERSISTENCE_TABLE_SQL)
        dst.executemany(
            f"INSERT INTO persistence_status VALUES ({', '.join('?' * 12)})", rows
        )

    return {
        "step": "persistence_status",
        "target_sqlite": str(target),
        "collections": len(rows),
        "collections_with_pending_records": sum(1 for r in rows if r[7]),
        "status": "ok",
        "rows": rows,
    }
