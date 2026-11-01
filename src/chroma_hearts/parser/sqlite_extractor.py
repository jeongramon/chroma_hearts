"""Recover intact SQLite records without a target manifest or header repair."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from .recovery_store import ensure_schema, add_evidence
from .sqlite_carver import scan_records
from .wal_merger import _collection_id_from_topic, _normalize_operation_text


def extract_sqlite_carved(source: Path, output_dir: Path) -> dict:
    source_file = source / 'chroma.sqlite3'
    data = source_file.read_bytes()  # Raw reads only: no SQLite/SDK open on evidence.
    result = {'source_sha256': hashlib.sha256(data).hexdigest(), 'header_repair': False,
              'records': 0, 'status': 'ok'}
    try:
        records, _ = scan_records(data)
    except (ValueError, AssertionError, IndexError) as exc:
        result.update(status='unsupported_or_invalid', reason=str(exc))
        (output_dir / 'sqlite_carving.json').write_text(json.dumps(result, indent=2) + '\n')
        return result
    with sqlite3.connect(output_dir / 'embeddings.sqlite3') as conn:
        ensure_schema(conn)
        # Carved observations stay separate: an extra ID or document must not
        # promote an HNSW orphan into a native WAL Full Entry.
        conn.execute("DELETE FROM recovery_evidence WHERE method='sqlite_carving'")
        conn.execute('''DELETE FROM embeddings WHERE source_sqlite_carved=1
                        AND source_chroma_sdk=0 AND source_wal=0 AND source_hnsw=0''')
        for record in records:
            kind = record['kind']
            rowids = {p['rowid'] for p in record['cell_prefixes'] if p['rowid'] is not None}
            source_rowid = next(iter(rowids)) if len(rowids) == 1 else None
            seq_id = created = operation = topic = item_id = encoding = collection = None
            internal_id = None
            if kind == 'embeddings_queue':
                _, created, op, topic, item_id, vector, encoding, metadata = record['values']
                seq_id = source_rowid
                operation = _normalize_operation_text(op)
                collection = _collection_id_from_topic(topic)
                table = 'embeddings_queue'
            else:
                vector = None
                metadata = json.dumps({'chroma:document': record['document']}, ensure_ascii=False)
                table = {'embedding_metadata': 'embedding_metadata',
                         'embedding_metadata_string_index': 'embedding_metadata_string_value',
                         'fts_content_compatible': 'embedding_fulltext_search_content'}[kind]
                if kind == 'embedding_metadata':
                    internal_id = record['internal_id']
                elif kind == 'fts_content_compatible':
                    internal_id = source_rowid
            cursor = conn.execute('''INSERT INTO embeddings
                (seq_id,created_at,operation,topic,id,sqlite_embedding_pk,collection_id,vector,
                 encoding,metadata,source_sqlite_carved)
                VALUES (?,?,?,?,?,?,?,?,?,?,1)''',
                (seq_id, created, operation, topic, item_id, internal_id, collection, vector, encoding, metadata))
            add_evidence(conn, cursor.lastrowid, 'sqlite_carving', source_file='chroma.sqlite3',
                         source_table=table, source_rowid=source_rowid, region_type=record['region_kind'],
                         page_no=record['page'], record_offset=record['header_offset'],
                         byte_ranges=record['segments'], item_id=item_id, collection_id=collection,
                         vector=vector, metadata=metadata,
                         details={'header_size': record['header_size'], 'serial_types': record['serial_types'],
                                  'header_hex': data[record['header_offset']:record['header_offset'] + record['header_size']].hex(),
                                  'payload_size': record['payload_size'], 'cell_prefixes': record['cell_prefixes'],
                                  'field_spans': record['field_spans'], 'schema_signature': kind,
                                  'source_table_inferred': True, 'header_repaired': False})
            result['records'] += 1
    (output_dir / 'sqlite_carving.json').write_text(json.dumps(result, indent=2) + '\n')
    return result
