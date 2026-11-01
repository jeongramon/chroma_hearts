#!/usr/bin/env python3
"""Snapshot extraction helpers for the Open WebUI case study.

The parser preserves SDK, native WAL, HNSW, and intact SQLite-carved evidence.
One representative path is selected per deleted (collection, ID) target:
WAL Full Entry > Identified Vector > Orphan Vector > SQLite Carved Data.
Vector and original-text recovery totals include all matching evidence.
No missing record headers are repaired and no cosine matches enter exact recall.
"""

from __future__ import annotations

import array
import hashlib
import json
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

import chromadb  # noqa: E402
from chromadb.config import Settings  # noqa: E402

from chroma_hearts.parser.hnsw_extractor import extract_hnsw_from_copy  # noqa: E402
from chroma_hearts.parser.sdk_extractor import extract_via_sdk  # noqa: E402
from chroma_hearts.parser.source_copy import working_copy
from chroma_hearts.parser.sqlite_extractor import extract_sqlite_carved
from chroma_hearts.parser.wal_merger import merge_wal_from_copy  # noqa: E402



BEFORE_SNAPSHOT_SCHEMA = """
CREATE TABLE IF NOT EXISTS before_snapshot (
    id TEXT,
    collection_id TEXT,
    collection_name TEXT,
    vector BLOB,
    vector_hash TEXT,
    document TEXT,
    metadata TEXT,
    seq_id INTEGER,
    hnsw_durable_at_capture BOOLEAN
);
CREATE INDEX IF NOT EXISTS idx_before_id ON before_snapshot(id);
"""


def _load_id_seq_map(source_sqlite_path: Path) -> dict[str, int]:
    """chroma embedding_id -> seq_id, from the metadata segment's ``embeddings`` table.

    ``seq_id`` here is the durable WAL position at which this id's *current*
    add/update/upsert was committed to the (synchronously-durable) metadata
    segment — independent of whether the vector segment's HNSW binary has
    caught up to that same WAL position yet.
    """
    with sqlite3.connect(str(source_sqlite_path)) as conn:
        return {
            str(embedding_id): int(seq_id)
            for embedding_id, seq_id in conn.execute(
                "SELECT embedding_id, seq_id FROM embeddings"
            ).fetchall()
        }


def _load_vector_checkpoint_map(source_sqlite_path: Path) -> dict[str, int]:
    """collection_id -> vector segment's ``max_seq_id`` WAL catch-up checkpoint.

    Chroma's local HNSW vector segment lazily/asynchronously applies
    ``embeddings_queue`` (WAL) rows into the on-disk binary index and records
    how far it has caught up in ``max_seq_id``. A collection with no row in
    ``max_seq_id`` for its vector segment has never synced at all (checkpoint
    treated as 0).
    """
    with sqlite3.connect(str(source_sqlite_path)) as conn:
        collection_to_vector_segment = {
            str(collection_id): str(segment_id)
            for segment_id, collection_id in conn.execute(
                "SELECT id, collection FROM segments WHERE scope = 'VECTOR'"
            ).fetchall()
        }
        segment_checkpoints = {
            str(segment_id): int(seq_id)
            for segment_id, seq_id in conn.execute(
                "SELECT segment_id, seq_id FROM max_seq_id"
            ).fetchall()
        }
    return {
        collection_id: segment_checkpoints.get(segment_id, 0)
        for collection_id, segment_id in collection_to_vector_segment.items()
    }


def _to_float32_blob(vector: Any) -> bytes:
    """Convert vector-like value to float32 BLOB bytes."""
    values = vector.tolist() if hasattr(vector, "tolist") else list(vector)
    return array.array("f", [float(v) for v in values]).tobytes()


def _batched_get_ids(collection: Any, batch_size: int = 1000):
    """Yield collection ids in pages to avoid loading everything at once."""
    offset = 0
    while True:
        page = collection.get(limit=batch_size, offset=offset, include=[])
        ids = page.get("ids") or []
        if not ids:
            break
        yield ids
        offset += len(ids)


def extract_before_snapshot(before_db_path: Path, output_path: Path) -> int:
    """Full live snapshot of the *before* DB via SDK, including document text.

    Works on a throwaway copy so the original is never opened by a
    PersistentClient or by SQLite in read-write mode.
    """
    with working_copy(before_db_path, prefix="chroma_diff_before") as copy_db:
        client = chromadb.PersistentClient(
            path=str(copy_db), settings=Settings(anonymized_telemetry=False)
        )

        source_sqlite_path = copy_db / "chroma.sqlite3"
        id_seq_map = _load_id_seq_map(source_sqlite_path)
        vector_checkpoint_map = _load_vector_checkpoint_map(source_sqlite_path)

        if output_path.exists():
            output_path.unlink()
        conn = sqlite3.connect(str(output_path))
        conn.executescript(BEFORE_SNAPSHOT_SCHEMA)

        inserted = 0
        for collection in client.list_collections():
            for id_batch in _batched_get_ids(collection):
                payload = collection.get(
                    ids=id_batch, include=["embeddings", "metadatas", "documents"]
                )
                ids = payload.get("ids") or []
                embeddings = payload.get("embeddings")
                metadatas = payload.get("metadatas") or []
                documents = payload.get("documents") or []

                collection_id = str(collection.id)
                vector_checkpoint = vector_checkpoint_map.get(collection_id, 0)

                rows = []
                for idx, item_id in enumerate(ids):
                    vector_blob = None
                    vector_hash = None
                    if embeddings is not None and idx < len(embeddings) and embeddings[idx] is not None:
                        blob = _to_float32_blob(embeddings[idx])
                        vector_blob = sqlite3.Binary(blob)
                        vector_hash = hashlib.sha256(blob).hexdigest()
                    metadata_text = None
                    if idx < len(metadatas) and metadatas[idx] is not None:
                        metadata_text = json.dumps(metadatas[idx], ensure_ascii=False, default=str)
                    document_text = documents[idx] if idx < len(documents) else None
                    seq_id = id_seq_map.get(str(item_id))
                    hnsw_durable = (
                        seq_id is not None and seq_id <= vector_checkpoint
                    )
                    rows.append(
                        (
                            item_id,
                            collection_id,
                            collection.name,
                            vector_blob,
                            vector_hash,
                            document_text,
                            metadata_text,
                            seq_id,
                            hnsw_durable,
                        )
                    )

                if rows:
                    conn.executemany(
                        """
                        INSERT INTO before_snapshot
                        (id, collection_id, collection_name, vector, vector_hash, document, metadata, seq_id, hnsw_durable_at_capture)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        rows,
                    )
                    inserted += len(rows)

        conn.commit()
        conn.close()
        return inserted


def run_recovery_pipeline(after_db_path: Path, output_dir: Path,
                          segment_spaces: dict[str, str] | None = None) -> None:
    """Run SDK, carving, native WAL, and HNSW extraction on isolated copies."""
    after_db_path = after_db_path.resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Preserve a snapshot for WAL/HNSW; SDK extraction creates a separate copy.
    with working_copy(after_db_path, prefix="chroma_diff_after") as copy_db:
        inserted = extract_via_sdk(copy_db, output_dir=output_dir)
        print(f"[INFO] Step 1 (SDK snapshot of after-DB) rows: {inserted}")

        carving = extract_sqlite_carved(copy_db, output_dir)
        print(f"[INFO] SQLite record carving: {carving}")
        wal_result = merge_wal_from_copy(copy_db, output_dir)
        print(f"[INFO] Step 2 (WAL merge) result: {wal_result}")

        hnsw_result = extract_hnsw_from_copy(copy_db, output_dir, segment_spaces=segment_spaces)
        print(f"[INFO] Step 3 (HNSW binary parse) result: {hnsw_result}")
