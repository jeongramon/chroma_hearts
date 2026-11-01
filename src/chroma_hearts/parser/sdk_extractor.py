"""
SDK extractor (pipeline step 1):
1) Copy source Chroma DB directory to a temporary location,
2) Extract sysdb tables into sysdb.sqlite3,
3) Extract embeddings via Chroma SDK into embeddings.sqlite3,
4) Remove the temporary copy.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import chromadb
from chromadb.config import Settings

from chroma_hearts.generator import embedding as _embedding  # noqa: F401

from .source_copy import working_copy
from .recovery_store import capture_sdk_evidence

SYSDB_TABLES = [
    "tenants",
    "databases",
    "collections",
    "collection_metadata",
    "segments",
    "segment_metadata",
]

# Hard-coded from current `chroma.sqlite3` `embeddings_queue` reference schema.
EMBEDDINGS_TABLE_SQL = """
CREATE TABLE embeddings (
    recovery_id INTEGER PRIMARY KEY,
    seq_id BLOB,
    created_at TIMESTAMP,
    operation TEXT,
    topic TEXT,
    id TEXT,
    sqlite_embedding_pk INTEGER,
    hnsw_label INTEGER,
    segment_id TEXT,
    vector_segment_id TEXT,
    collection_id TEXT,
    collection_name TEXT,
    vector BLOB,
    encoding TEXT,
    metadata TEXT,
    source_chroma_sdk BOOLEAN DEFAULT 0,
    source_wal BOOLEAN DEFAULT 0,
    source_hnsw BOOLEAN DEFAULT 0,
    delete_mark BOOLEAN DEFAULT 0,
    source_sqlite_carved BOOLEAN DEFAULT 0
);
"""


def _copy_table_exact(
    source_conn: sqlite3.Connection,
    target_conn: sqlite3.Connection,
    table_name: str,
) -> int:
    """Copy one SQLite table schema + all rows exactly."""
    schema_row = source_conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    if schema_row is None or schema_row[0] is None:
        raise ValueError(f"Table not found in source sqlite: {table_name}")

    target_conn.execute(schema_row[0])
    rows = source_conn.execute(f"SELECT * FROM {table_name}").fetchall()
    if not rows:
        return 0

    placeholders = ",".join(["?"] * len(rows[0]))
    target_conn.executemany(
        f"INSERT INTO {table_name} VALUES ({placeholders})",
        rows,
    )
    return len(rows)


def _to_float32_blob(vector: Any) -> bytes:
    """Convert vector-like value to float32 BLOB bytes."""
    import array

    if hasattr(vector, "tolist"):
        values = vector.tolist()
    else:
        values = list(vector)
    return array.array("f", [float(v) for v in values]).tobytes()


def _batched_get_ids(collection: Any, batch_size: int = 1000) -> Iterable[List[str]]:
    """Yield collection ids in pages to avoid loading everything at once."""
    offset = 0
    while True:
        page = collection.get(limit=batch_size, offset=offset, include=[])
        ids = page.get("ids") or []
        if not ids:
            break
        yield ids
        offset += len(ids)


def _extract_embeddings_with_sdk(
    copy_db_path: Path,
    embeddings_sqlite_path: Path,
) -> Dict[str, Any]:
    """Extract vectors and metadata via Chroma SDK and store into embeddings.sqlite3."""
    client = chromadb.PersistentClient(
        path=str(copy_db_path),
        settings=Settings(anonymized_telemetry=False),
    )

    with sqlite3.connect(str(embeddings_sqlite_path)) as conn:
        conn.execute(EMBEDDINGS_TABLE_SQL)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_id ON embeddings(id)")
        inserted_rows = 0
        collection_stats: List[Dict[str, Any]] = []

        for collection in client.list_collections():
            collection_rows = 0
            for id_batch in _batched_get_ids(collection):
                payload = collection.get(
                    ids=id_batch,
                    include=["embeddings", "metadatas", "documents"],
                )
                ids = payload.get("ids")
                embeddings = payload.get("embeddings")
                documents = payload.get("documents") or []
                metadatas = payload.get("metadatas")
                if ids is None:
                    ids = []
                if embeddings is None:
                    embeddings = []
                if metadatas is None:
                    metadatas = []

                insert_rows = []
                for idx, item_id in enumerate(ids):
                    vector_blob = None
                    if idx < len(embeddings) and embeddings[idx] is not None:
                        vector_blob = sqlite3.Binary(_to_float32_blob(embeddings[idx]))
                    metadata_text = None
                    if idx < len(metadatas) and metadatas[idx] is not None:
                        metadata_text = json.dumps(metadatas[idx], ensure_ascii=False, default=str)

                    if idx < len(documents) and documents[idx] is not None:
                        obj = json.loads(metadata_text or "{}")
                        obj["chroma:document"] = documents[idx]
                        metadata_text = json.dumps(obj, ensure_ascii=False)

                    insert_rows.append(
                        (
                            None,  # seq_id unknown
                            None,  # created_at unknown
                            "Add",  # operation fixed
                            None,  # topic unknown
                            item_id,
                            None,  # sqlite_embedding_pk unknown
                            None,  # hnsw_label unknown
                            None,  # segment_id unknown
                            None,  # vector_segment_id unknown until HNSW extraction
                            str(collection.id),
                            collection.name,
                            vector_blob,
                            None,  # encoding unknown
                            metadata_text,
                            True,  # source_chroma_sdk
                            False,  # source_wal
                            False,  # source_hnsw
                            False,  # delete_mark default
                        )
                    )

                if insert_rows:
                    conn.executemany(
                        """
                        INSERT INTO embeddings
                        (
                            seq_id,
                            created_at,
                            operation,
                            topic,
                            id,
                            sqlite_embedding_pk,
                            hnsw_label,
                            segment_id,
                            vector_segment_id,
                            collection_id,
                            collection_name,
                            vector,
                            encoding,
                            metadata,
                            source_chroma_sdk,
                            source_wal,
                            source_hnsw,
                            delete_mark
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        insert_rows,
                    )
                    inserted_rows += len(insert_rows)
                    collection_rows += len(insert_rows)

            collection_stats.append(
                {
                    "name": collection.name,
                    "id": str(collection.id),
                    "count": collection.count(),
                    "inserted_rows": collection_rows,
                }
            )

        conn.commit()

    return {
        "embeddings_sqlite_path": str(embeddings_sqlite_path),
        "inserted_rows": inserted_rows,
        "collections": collection_stats,
    }


def _sync_embeddings_from_source(
    source_sqlite_path: Path,
    embeddings_sqlite_path: Path,
) -> None:
    """
    Sync columns from source `embeddings` table into embeddings table.

    Matching rule:
    - source.embedding_id == target.embeddings.id
    """
    with sqlite3.connect(str(source_sqlite_path)) as src_conn, sqlite3.connect(
        str(embeddings_sqlite_path)
    ) as dst_conn:
        src_rows = src_conn.execute(
            """
            SELECT
                embeddings.id,
                embeddings.embedding_id,
                embeddings.seq_id,
                embeddings.created_at,
                embeddings.segment_id,
                segments.collection
            FROM embeddings
            JOIN segments ON embeddings.segment_id = segments.id
            """
        ).fetchall()

        dst_conn.executemany(
            """
            UPDATE embeddings
            SET
                seq_id = ?,
                created_at = ?,
                sqlite_embedding_pk = ?,
                segment_id = ?
            WHERE id = ?
            AND collection_id = ?
            """,
            [
                (
                    row[2],  # seq_id
                    row[3],  # created_at
                    row[0],  # source sqlite embeddings.id
                    row[4],  # segment_id
                    row[1],  # embedding_id -> target.id
                    row[5],  # collection id
                )
                for row in src_rows
            ],
        )
        dst_conn.commit()


def extract_via_sdk(
    source_chroma_db_path: Path,
    output_dir: Optional[Path] = None,
) -> int:
    """
    Build reports using a dedicated temporary copy, then remove that copy.

    Always copy, even when the caller supplies a pipeline working copy. SDK
    initialization and reads can mutate WAL/HNSW artifacts that later recovery
    stages still need in their captured state.

    Args:
        source_chroma_db_path: Path to the persisted Chroma directory (e.g. chroma_db/).
        output_dir: Output directory that will contain:
            - sysdb.sqlite3
            - embeddings.sqlite3
    """
    if output_dir is None:
        raise ValueError("output_dir is required for step 1 extraction.")
    with working_copy(source_chroma_db_path, prefix="chroma_hearts_sdk") as copy_db:
        return _extract_via_sdk_on_working_copy(copy_db, output_dir)


def extract_via_sdk_from_copy(
    source_chroma_db_path: Path,
    output_dir: Optional[Path] = None,
) -> int:
    """Compatibility entry point; isolate SDK reads in another temporary copy.

    Preserve the supplied directory for subsequent WAL/HNSW extraction, even
    when it is already a working copy. New callers should use extract_via_sdk.
    """
    return extract_via_sdk(source_chroma_db_path, output_dir=output_dir)


def _extract_via_sdk_on_working_copy(
    copy_db: Path,
    output_dir: Path,
) -> int:
    """Extract reports from the dedicated, disposable SDK directory."""
    source_sqlite_path = copy_db / "chroma.sqlite3"

    out = output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    # 1) sysdb extraction
    sysdb_path = out / "sysdb.sqlite3"
    if sysdb_path.exists():
        sysdb_path.unlink()
    with sqlite3.connect(str(source_sqlite_path)) as src_conn, sqlite3.connect(
        str(sysdb_path)
    ) as dst_conn:
        table_counts: Dict[str, int] = {}
        for table_name in SYSDB_TABLES:
            table_counts[table_name] = _copy_table_exact(src_conn, dst_conn, table_name)
        dst_conn.commit()
    # 2) embeddings extraction via Chroma SDK
    embeddings_path = out / "embeddings.sqlite3"
    if embeddings_path.exists():
        embeddings_path.unlink()
    embeddings_result = _extract_embeddings_with_sdk(copy_db, embeddings_path)
    _sync_embeddings_from_source(source_sqlite_path, embeddings_path)
    with sqlite3.connect(embeddings_path) as conn:
        capture_sdk_evidence(conn)
    return int(embeddings_result.get("inserted_rows", 0))
