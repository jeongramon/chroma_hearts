"""Source-specific evidence, kept separately from merged result rows."""
from __future__ import annotations

import json
import sqlite3


def ensure_schema(conn: sqlite3.Connection) -> None:
    columns = {r[1] for r in conn.execute('PRAGMA table_info(embeddings)')}
    if 'source_sqlite_carved' not in columns:
        conn.execute('ALTER TABLE embeddings ADD COLUMN source_sqlite_carved BOOLEAN DEFAULT 0')
    conn.execute('''CREATE TABLE IF NOT EXISTS recovery_evidence (
        evidence_id INTEGER PRIMARY KEY,
        embedding_rowid INTEGER NOT NULL,
        method TEXT NOT NULL,
        source_file TEXT,
        source_table TEXT,
        source_rowid INTEGER,
        region_type TEXT,
        page_no INTEGER,
        record_offset INTEGER,
        byte_ranges TEXT,
        id TEXT,
        collection_id TEXT,
        vector BLOB,
        metadata TEXT,
        details TEXT
    )''')
    conn.execute('CREATE INDEX IF NOT EXISTS evidence_embedding ON recovery_evidence(embedding_rowid)')


def add_evidence(conn: sqlite3.Connection, rowid: int, method: str, *,
                 source_file=None, source_table=None, source_rowid=None, region_type=None,
                 page_no=None, record_offset=None, byte_ranges=None, item_id=None,
                 collection_id=None, vector=None, metadata=None, details=None) -> None:
    conn.execute('''INSERT INTO recovery_evidence
        (embedding_rowid,method,source_file,source_table,source_rowid,region_type,page_no,
         record_offset,byte_ranges,id,collection_id,vector,metadata,details)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
        (rowid, method, source_file, source_table, source_rowid, region_type, page_no,
         record_offset, json.dumps(byte_ranges) if byte_ranges is not None else None,
         item_id, collection_id, vector, metadata,
         json.dumps(details, ensure_ascii=False) if details is not None else None))


def capture_sdk_evidence(conn: sqlite3.Connection) -> None:
    ensure_schema(conn)
    conn.execute("DELETE FROM recovery_evidence WHERE method='sdk'")
    for rowid, item_id, collection, vector, metadata in conn.execute(
            'SELECT rowid,id,collection_id,vector,metadata FROM embeddings WHERE source_chroma_sdk=1').fetchall():
        add_evidence(conn, rowid, 'sdk', source_file='chroma.sqlite3', item_id=item_id,
                     collection_id=collection, vector=vector, metadata=metadata)
