"""Target-level representative paths and independent vector/text recovery.

Known targets are evaluation inputs only. They never enter the parser/carver or
change recovered identities. Evidence retains every source, including overlap.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3

PATHS = ('wal_full_entry', 'identified_vector', 'orphan_vector', 'sqlite_carved_data')
LABELS = dict(zip(PATHS, ('WAL Full Entry', 'Identified Vector', 'Orphan Vector', 'SQLite Carved Data')))


def document(metadata):
    try:
        value = json.loads(metadata or '{}')
        text = value.get('chroma:document') if isinstance(value, dict) else None
        return text if isinstance(text, str) else None
    except (ValueError, TypeError):
        return None


def load_evidence(path: Path) -> list[dict]:
    with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as conn:
        conn.row_factory = sqlite3.Row
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        result = [dict(r) for r in conn.execute('SELECT * FROM recovery_evidence')] if 'recovery_evidence' in tables else []
        covered = {(r['embedding_rowid'], r['method']) for r in result}
        next_id = max((r['evidence_id'] for r in result), default=0)
        # Historical or partially augmented output compatibility.

        for row in conn.execute('SELECT rowid AS output_rowid,* FROM embeddings'):
            r = dict(row)
            for flag, method in (('source_chroma_sdk', 'sdk'), ('source_wal', 'wal'),
                                 ('source_hnsw', 'hnsw'), ('source_sqlite_carved', 'sqlite_carving')):
                if r.get(flag) and (r['output_rowid'], method) not in covered:
                    next_id += 1
                    result.append({'evidence_id': next_id, 'embedding_rowid': r['output_rowid'],
                                   'method': method, 'id': r['id'], 'collection_id': r.get('collection_id'),
                                   'vector': r['vector'], 'metadata': r['metadata'] if method != 'hnsw' else None})
        return result


class RecoveryIndex:
    def __init__(self, path: Path):
        self.rows = load_evidence(path)
        self.by_vector, self.by_text = defaultdict(list), defaultdict(list)
        self.active_id, self.active_vector, self.active_text = defaultdict(list), defaultdict(list), defaultdict(list)
        for row in self.rows:
            row['vector_hash'] = hashlib.sha256(bytes(row['vector'])).hexdigest() if row['vector'] is not None else None
            row['document'] = document(row['metadata'])
            if row['method'] == 'sdk':
                self.active_id[row['id']].append(row)
                self.active_vector[row['vector_hash']].append(row)
                self.active_text[row['document']].append(row)
            else:
                if row['vector_hash']:
                    self.by_vector[row['vector_hash']].append(row)
                if row['document'] is not None:
                    self.by_text[row['document']].append(row)

    def _active_or_ambiguous(self, row: dict) -> bool:
        if row['id'] is not None:
            candidates = self.active_id[row['id']]
        elif row['vector_hash']:
            candidates = self.active_vector[row['vector_hash']]
        else:
            candidates = self.active_text[row['document']]
        for active in candidates:
            if row['collection_id'] and row['collection_id'] != active['collection_id']:
                continue
            if row['vector_hash'] and row['vector_hash'] == active['vector_hash']:
                return True
            if row['document'] is not None and row['document'] == active['document']:
                return True
        return False

    def grade(self, item_id: str | None, vector_hash: str | None, text: str | None,
              collection_id: str | None = None) -> dict:
        candidates = {r['evidence_id']: r for r in self.by_vector.get(vector_hash, [])}
        if text is not None:
            candidates.update({r['evidence_id']: r for r in self.by_text.get(text, [])})
        matches, excluded = [], []
        for row in candidates.values():
            if item_id is not None and row['id'] is not None and row['id'] != item_id:
                continue
            if collection_id and row['collection_id'] and row['collection_id'] != collection_id:
                continue
            if self._active_or_ambiguous(row):
                excluded.append(row['evidence_id'])
                continue
            matches.append(row)
        vector_rows = [r for r in matches if vector_hash and r['vector_hash'] == vector_hash]
        text_rows = [r for r in matches if text is not None and r['document'] == text]
        wal = [r for r in vector_rows if r['method'] == 'wal' and r['id'] == item_id
               and r['document'] is not None and (text is None or r['document'] == text)]
        identified = [r for r in vector_rows if r['method'] == 'hnsw' and r['id'] is not None
                      and r['id'] == item_id]
        orphan = [r for r in vector_rows if r['method'] == 'hnsw' and r['id'] is None]
        carved = [r for r in matches if r['method'] == 'sqlite_carving']
        selected = next((name for name, rows in zip(PATHS, (wal, identified, orphan, carved)) if rows), 'none')
        return {'representative_path': selected, 'vector_recovered': bool(vector_rows),
                'text_recovered': bool(text_rows), 'id_recovered': any(r['id'] == item_id for r in matches if item_id is not None),
                'source_paths': sorted({r['method'] for r in matches}),
                'matched_evidence_ids': sorted(r['evidence_id'] for r in matches),
                'excluded_active_or_ambiguous_evidence_ids': sorted(excluded),
                'unclassified_data': selected == 'none' and bool(matches)}


def summarize(rows: list[dict]) -> dict:
    counts = Counter(r['representative_path'] for r in rows)
    return {'targets': len(rows), **{p: counts[p] for p in (*PATHS, 'none')},
            'vector_recovered': sum(r['vector_recovered'] for r in rows),
            'text_recovered': sum(r['text_recovered'] for r in rows),
            'unclassified_data': sum(r.get('unclassified_data', False) for r in rows)}


def save_targets(path: Path, rows: list[dict]) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute('DROP TABLE IF EXISTS target_recovery')
        conn.execute('''CREATE TABLE target_recovery (
            event_index INTEGER PRIMARY KEY, event_type TEXT, target_id TEXT, collection_id TEXT,
            vector_hash TEXT, text_sha256 TEXT, representative_path TEXT,
            vector_recovered BOOLEAN, text_recovered BOOLEAN, id_recovered BOOLEAN,
            source_paths TEXT, matched_evidence_ids TEXT, excluded_active_or_ambiguous_evidence_ids TEXT,
            unclassified_data BOOLEAN)''')
        for i, row in enumerate(rows):
            conn.execute('INSERT INTO target_recovery VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                         (i, row.get('event_type'), row.get('target_id'), row.get('collection_id'), row.get('vector_hash'), row.get('text_sha256'),
                          row['representative_path'], row['vector_recovered'], row['text_recovered'], row['id_recovered'],
                          json.dumps(row['source_paths']), json.dumps(row['matched_evidence_ids']),
                          json.dumps(row['excluded_active_or_ambiguous_evidence_ids']), row.get('unclassified_data', False)))
        conn.execute('DROP TABLE IF EXISTS recovery_path_summary')
        conn.execute('CREATE TABLE recovery_path_summary (event_type TEXT PRIMARY KEY, metrics TEXT)')
        for kind in sorted({r.get('event_type', 'target') for r in rows}):
            conn.execute('INSERT INTO recovery_path_summary VALUES (?,?)',
                         (kind, json.dumps(summarize([r for r in rows if r.get('event_type', 'target') == kind]))))
