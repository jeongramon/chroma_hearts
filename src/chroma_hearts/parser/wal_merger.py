"""
WAL merger (pipeline step 2):
Merge `chroma.sqlite3`.`embeddings_queue` (write-ahead log) into the report.
"""

import pickle
import sqlite3
from pathlib import Path
from typing import Any, Dict

from .source_copy import working_copy
from .recovery_store import ensure_schema, add_evidence

OPERATION_MAP = {
    0: "Add",
    1: "Update",
    2: "Upsert",
    3: "Delete",
}


def _normalize_operation_text(operation_code: Any) -> str:
    """Convert operation code to canonical operation text."""
    try:
        op_code = int(operation_code)
    except (TypeError, ValueError):
        return str(operation_code)
    return OPERATION_MAP.get(op_code, str(op_code))


def _load_id_hnsw_label_pairs_from_pickle(pickle_path: Path) -> list[tuple[int, str]]:
    """Read HNSW label / Chroma id pairs from one index_metadata.pickle file."""
    with pickle_path.open("rb") as handle:
        payload = pickle.load(handle)

    pairs: list[tuple[int, str]] = []
    id_to_label = payload.get("id_to_label", {}) if isinstance(payload, dict) else {}
    label_to_id = payload.get("label_to_id", {}) if isinstance(payload, dict) else {}

    # Primary source: id -> label mapping.
    for item_id, label in id_to_label.items():
        try:
            pairs.append((int(label), str(item_id)))
        except (TypeError, ValueError):
            continue

    # Fallback source: label -> id mapping.
    for label, item_id in label_to_id.items():
        try:
            pairs.append((int(label), str(item_id)))
        except (TypeError, ValueError):
            continue

    return pairs


def _load_segment_collection_maps(
    sysdb_sqlite_path: Path,
) -> tuple[dict[str, str], dict[str, str]]:
    """Load vector segment -> collection and collection -> name mappings."""
    uri = f"file:{sysdb_sqlite_path}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        segment_to_collection = {
            str(segment_id): str(collection_id)
            for segment_id, collection_id in conn.execute(
                "SELECT id, collection FROM segments"
            ).fetchall()
            if segment_id is not None and collection_id is not None
        }
        collection_to_name = {
            str(collection_id): str(name)
            for collection_id, name in conn.execute(
                "SELECT id, name FROM collections"
            ).fetchall()
            if collection_id is not None and name is not None
        }
    return segment_to_collection, collection_to_name


def _collection_id_from_topic(topic: Any) -> str | None:
    """Extract collection id from a Chroma persistent topic string."""
    if topic is None:
        return None
    topic_text = str(topic)
    if "/" not in topic_text:
        return None
    collection_id = topic_text.rsplit("/", 1)[-1]
    return collection_id or None


def _supplement_hnsw_label_id_from_indexes(
    source_chroma_db_path: Path,
    dst_conn: sqlite3.Connection,
    segment_to_collection: dict[str, str],
    collection_to_name: dict[str, str],
) -> int:
    """Supplement missing (hnsw_label, id) rows from index metadata pickles."""

    existing_ids = {
        (row[0], row[1])
        for row in dst_conn.execute("SELECT id, collection_id FROM embeddings WHERE source_sqlite_carved=0").fetchall()
        if row and row[0] is not None
    }

    insert_rows: list[tuple[int, str, str, str | None, str | None]] = []
    for pickle_path in sorted(source_chroma_db_path.glob("*/index_metadata.pickle")):
        vector_segment_id = pickle_path.parent.name
        collection_id = segment_to_collection.get(vector_segment_id)
        collection_name = (
            collection_to_name.get(collection_id)
            if collection_id is not None
            else None
        )
        for hnsw_label, item_id in _load_id_hnsw_label_pairs_from_pickle(pickle_path):
            existing_key = (item_id, collection_id)
            if existing_key in existing_ids:
                continue
            existing_ids.add(existing_key)
            insert_rows.append(
                (hnsw_label, item_id, vector_segment_id, collection_id, collection_name)
            )

    if insert_rows:
        dst_conn.executemany(
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
            [
                (
                    None,
                    None,
                    None,
                    None,
                    item_id,
                    None,
                    hnsw_label,
                    None,
                    vector_segment_id,
                    collection_id,
                    collection_name,
                    None,
                    None,
                    None,
                    False,
                    False,
                    False,
                    False,
                )
                for hnsw_label, item_id, vector_segment_id, collection_id, collection_name
                in insert_rows
            ],
        )
    return len(insert_rows)


def merge_wal(source_chroma_db_path: Path, output_dir: Path) -> Dict[str, Any]:
    """Merge the source WAL into the report, working on a throwaway copy.

    The source is copied first because reading ``embeddings_queue`` means opening
    ``chroma.sqlite3``, and SQLite can checkpoint or truncate the write-ahead log
    on open — which would destroy the very records this stage recovers. Callers
    that already hold a working copy should call :func:`merge_wal_from_copy`
    instead to avoid copying twice.
    """
    with working_copy(source_chroma_db_path, prefix="chroma_hearts_wal") as copy_db:
        return merge_wal_from_copy(copy_db, output_dir)


def merge_wal_from_copy(source_chroma_db_path: Path, output_dir: Path) -> Dict[str, Any]:
    """
    Merge `chroma.sqlite3`.`embeddings_queue` into report `embeddings.sqlite3`.

    ``source_chroma_db_path`` must already be a working copy — this function
    opens it read-write. Use :func:`merge_wal` to copy first.

    Rules:
    - Operation codes are converted as: 0=Add, 1=Update, 2=Upsert, 3=Delete.
    - Each WAL record is preserved as its own row, keeping its ``seq_id`` and
      ``operation``. A vector store holds many vector versions per id over time
      (every update/upsert supersedes the previous one), so collapsing to one row
      per id would silently drop every version but the last — including vectors
      that are still physically present in the queue.
    - Duplicates are suppressed only on exact ``(collection_id, id, vector)``
      identity: if a row already carries this id's exact vector bytes (e.g. from
      the live SDK snapshot, or an earlier identical WAL record), the record is
      folded into that row (``source_wal=1`` plus any missing fields) instead of
      inserting a copy. Two records with *different* vector bytes are never merged.
    - A Delete tombstone (vector NULL, metadata "{}") is kept as its own row with
      ``delete_mark=1``; because it is a separate row it can no longer overwrite
      the document/metadata recovered from an earlier Add record for the same id.
    """
    source_sqlite = source_chroma_db_path.resolve() / "chroma.sqlite3"
    target_sqlite = output_dir.resolve() / "embeddings.sqlite3"
    sysdb_sqlite = output_dir.resolve() / "sysdb.sqlite3"

    if not source_sqlite.exists():
        raise FileNotFoundError(f"Source sqlite not found: {source_sqlite}")
    if not target_sqlite.exists():
        raise FileNotFoundError(f"Target embeddings sqlite not found: {target_sqlite}")
    if not sysdb_sqlite.exists():
        raise FileNotFoundError(f"sysdb sqlite not found: {sysdb_sqlite}")

    segment_to_collection, collection_to_name = _load_segment_collection_maps(sysdb_sqlite)

    with sqlite3.connect(str(source_sqlite)) as src_conn, sqlite3.connect(
        str(target_sqlite)
    ) as dst_conn:
        ensure_schema(dst_conn)
        supplemented_hnsw_labels = _supplement_hnsw_label_id_from_indexes(
            source_chroma_db_path.resolve(),
            dst_conn,
            segment_to_collection,
            collection_to_name,
        )

        queue_rows = src_conn.execute(
            """
            SELECT seq_id, created_at, operation, topic, id, vector, encoding, metadata
            FROM embeddings_queue
            ORDER BY seq_id ASC
            """
        ).fetchall()

        updated_count = 0
        inserted_count = 0

        for row in queue_rows:
            seq_id, created_at, operation_code, topic, item_id, vector, encoding, metadata = row
            operation = _normalize_operation_text(operation_code)
            is_tombstone = vector is None
            collection_id = _collection_id_from_topic(topic)
            collection_name = (
                collection_to_name.get(collection_id)
                if collection_id is not None
                else None
            )

            # Dedup only on exact (collection_id, id, vector) identity: if this
            # record's vector bytes already exist for this id, fold the WAL
            # provenance into that row rather than duplicating it. Tombstones
            # (no vector) are never folded — each is kept as its own row.
            match_rowid: int | None = None
            if not is_tombstone:
                candidates = dst_conn.execute(
                    """
                    SELECT rowid, vector
                    FROM embeddings
                    WHERE id = ?
                    AND source_sqlite_carved = 0
                    AND (collection_id = ? OR collection_id IS NULL)
                    """,
                    (item_id, collection_id),
                ).fetchall()
                for rowid, existing_vector in candidates:
                    if existing_vector is not None and bytes(existing_vector) == bytes(vector):
                        match_rowid = rowid
                        break

            if match_rowid is not None:
                # Same vector already recovered elsewhere (SDK snapshot or an
                # identical earlier record): record WAL provenance, back-fill any
                # missing fields, but never change the vector.
                merge_metadata = metadata
                if merge_metadata is not None and merge_metadata.strip() in ("", "{}"):
                    merge_metadata = None
                dst_conn.execute(
                    """
                    UPDATE embeddings
                    SET
                        seq_id = COALESCE(seq_id, ?),
                        created_at = COALESCE(created_at, ?),
                        operation = COALESCE(operation, ?),
                        topic = COALESCE(topic, ?),
                        encoding = COALESCE(encoding, ?),
                        metadata = COALESCE(metadata, ?),
                        collection_id = COALESCE(collection_id, ?),
                        collection_name = COALESCE(collection_name, ?),
                        source_wal = 1
                    WHERE rowid = ?
                    """,
                    (
                        seq_id,
                        created_at,
                        operation,
                        topic,
                        encoding,
                        merge_metadata,
                        collection_id,
                        collection_name,
                        match_rowid,
                    ),
                )
                updated_count += 1
            else:
                # New vector version, or a Delete tombstone: keep as its own row.
                dst_conn.execute(
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
                    (
                        seq_id,
                        created_at,
                        operation,
                        topic,
                        item_id,
                        None,  # sqlite_embedding_pk unknown for pure WAL insert
                        None,  # hnsw_label unknown for pure WAL insert
                        None,  # segment_id not provided by WAL queue
                        None,  # vector_segment_id unknown until HNSW extraction
                        collection_id,
                        collection_name,
                        vector,
                        encoding,
                        metadata,
                        False,  # source_chroma_sdk
                        True,  # source_wal
                        False,  # source_hnsw
                        bool(is_tombstone),  # delete_mark set on Delete tombstones
                    ),
                )
                match_rowid = dst_conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                inserted_count += 1

            add_evidence(dst_conn, match_rowid, 'wal', source_file='chroma.sqlite3',
                         source_table='embeddings_queue', source_rowid=seq_id,
                         item_id=item_id, collection_id=collection_id, vector=vector,
                         metadata=metadata, details={'operation': operation, 'topic': topic,
                                                     'created_at': created_at, 'encoding': encoding})

        dst_conn.commit()

    return {
        "step": "wal_merge",
        "source_chroma_db_path": str(source_chroma_db_path.resolve()),
        "target_embeddings_sqlite": str(target_sqlite),
        "supplemented_hnsw_label_rows": supplemented_hnsw_labels,
        "queue_rows": len(queue_rows),
        "updated_rows": updated_count,
        "inserted_rows": inserted_count,
        "status": "ok",
    }
