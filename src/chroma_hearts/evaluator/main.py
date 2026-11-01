"""
Evaluate recovery rate by comparing operations.sqlite3 (ground truth)
with recovered embeddings.sqlite3.

Two evaluation paths:

* **Verified path** (hash embeddings + operation log carrying ``doc_id`` /
  ``document``): the deterministic hash embedding lets us reconstruct the exact
  float32 vector bytes for any known document text. We replay the operation log
  to learn, per Chroma id, which vectors were deleted (net) vs merely superseded
  by update/upsert, then **byte-verify** each recovered vector against the
  reconstructed ground truth. This correctly excludes update/upsert leftovers
  from the delete recovery rate and lets even id-less orphan vectors be
  attributed to a document by their bytes.

* **Legacy presence path** (default embeddings, or older logs without the
  ground-truth columns): falls back to counting recovered vectors by presence,
  as before. It cannot verify vector correctness.

The verified path also grades **collection drops** (every item alive in a
dropped collection counts as deleted at the drop) and, for deleted items found
in none of the three residue forms, checks the known bytes against SQLite
unallocated space (``sqlite_unallocated``). That check is knowledge-based, not
carving, and is reported separately as *SQLite remnants*.
"""

from __future__ import annotations

import array
import hashlib
import sqlite3
from argparse import Namespace
from dataclasses import asdict, dataclass, field, fields
import json
from pathlib import Path
from typing import Any

from chroma_hearts.config import get_embedding_settings
from chroma_hearts.config import get_evaluator_defaults as get_default_args
from chroma_hearts.generator.embedding import hash_text_to_vector
from . import residue_lifetime
from .recovery_paths import RecoveryIndex, save_targets
from .sqlite_unallocated import (
    KnownTexts,
    REMNANT_ID_ONLY,
    REMNANT_TEXT,
    REMNANT_VECTOR,
    REMNANT_VECTOR_TEXT,
    UnallocatedSpace,
    probe_item,
    read_unallocated_space,
    remnant_class,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _operation_summary(conn: sqlite3.Connection) -> dict[str, dict[str, int]]:
    """Per-operation attempts/successes from the operation log."""
    rows = conn.execute(
        """
        SELECT
            operation,
            COUNT(*) AS attempts,
            SUM(CASE WHEN result != '0' THEN 1 ELSE 0 END) AS successes
        FROM operations
        GROUP BY operation
        ORDER BY operation
        """
    ).fetchall()
    return {
        row[0]: {"attempts": int(row[1]), "successes": int(row[2] or 0)}
        for row in rows
    }


def _has_ground_truth_columns(ops_db_path: Path) -> bool:
    """True if the operation log records per-operation doc_id and document."""
    conn = sqlite3.connect(str(ops_db_path))
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(operations)")}
    finally:
        conn.close()
    return {"doc_id", "document"}.issubset(cols)


def _expected_blob(text: str, dimension: int) -> bytes:
    """The float32 vector bytes Chroma would store for ``text``.

    Mirrors how vectors are persisted: ``array('f', ...).tobytes()``. Verified
    to be byte-identical to Chroma's stored/recovered vectors for hash
    embeddings.
    """
    vector = hash_text_to_vector(text, dimension)
    return array.array("f", [float(v) for v in vector]).tobytes()


def _expected_blob_hash(text: str, dimension: int) -> str:
    """SHA-256 of :func:`_expected_blob`."""
    return hashlib.sha256(_expected_blob(text, dimension)).hexdigest()


def _percent(numerator: int, denominator: int) -> float:
    """Safe percentage helper."""
    if denominator <= 0:
        return 0.0
    return numerator / denominator * 100.0


# ---------------------------------------------------------------------------
# Verified path (hash embeddings + ground-truth operation log)
# ---------------------------------------------------------------------------


@dataclass
class VerifiedRecoveryMetrics:
    """Byte-verified recovery evaluation for one hash-embedding run."""

    run_label: str
    embedding_mode: str
    embedding_dimension: int
    operation_summary: str
    total_operations: int
    add_success: int
    update_events: int
    upsert_events: int
    delete_events: int
    distinct_deleted_ids: int
    final_alive_count: int
    total_recovered_rows: int
    # Verified recovery counts (numerators), scoped to distinct deleted ids.
    deleted_id_verified: int
    deleted_content_only: int
    deleted_vector_recovered: int
    deleted_full_chunk: int
    deleted_latest_recovered: int
    deleted_original_recovered: int
    # Rates against the distinct-deleted-id denominator.
    id_verified_rate_by_id: float
    content_only_rate_by_id: float
    combined_rate_by_id: float
    full_chunk_rate_by_id: float
    # Delete-event grading. Denominator is delete_events (the number of delete
    # operations); the numerator for each class counts the vector that was live
    # at that delete event and was actually recovered from the deletion residue.
    # By construction the four classes sum to delete_events.
    delete_event_full_chunk: int
    delete_event_identified_only: int
    delete_event_orphan_only: int
    delete_event_unrecovered: int
    delete_event_full_chunk_rate: float
    delete_event_identified_only_rate: float
    delete_event_orphan_only_rate: float
    delete_event_unrecovered_rate: float
    delete_event_vector_recovered: int
    delete_event_vector_recovered_rate: float
    # Cross-check: where the non-live recovered vectors actually come from.
    recovered_nonlive_vectors: int
    recovered_matched_deleted: int
    recovered_matched_superseded: int
    recovered_unexplained: int
    # Update/upsert-superseded vector accounting (event-based). Each update to a
    # live id replaces its vector; the old version is graded the same way as a
    # delete-event vector, or is PERMANENTLY LOST when its bytes survive nowhere
    # at all. By construction superseded_total = (update + upsert-on-existing)
    # events, and the four classes sum to superseded_total.
    superseded_total: int
    superseded_full_chunk: int
    superseded_identified_only: int
    superseded_orphan_only: int
    superseded_lost: int
    superseded_full_chunk_rate: float
    superseded_identified_only_rate: float
    superseded_orphan_only_rate: float
    superseded_lost_rate: float
    superseded_recovered: int
    superseded_recovered_rate: float
    # Of the recovered superseded vectors, how many survived only because an
    # identical byte sequence is live again (a re-add of the same text) rather
    # than as deletion residue. Reported for transparency; already counted in
    # the classes above.
    superseded_recovered_via_live_readd: int
    # SQLite remnants of delete events graded "Not recovered": the known bytes
    # still sit in unallocated space of chroma.sqlite3. The classes partition
    # delete_event_unrecovered (most informative first). text_in_unallocated
    # counts Identified/Orphan events whose chunk text also survives there.
    delete_event_remnant_vector_text: int
    delete_event_remnant_vector: int
    delete_event_remnant_text: int
    delete_event_remnant_id_only: int
    delete_event_not_found: int
    delete_event_text_in_unallocated: int
    # Collection drop. drop_items = items alive in a collection when it was
    # dropped; graded like delete events, and the classes sum to drop_items.
    drop_events: int
    drop_items: int
    drop_full_chunk: int
    drop_identified_only: int
    drop_orphan_only: int
    drop_unrecovered: int
    drop_vector_recovered: int
    drop_remnant_vector_text: int
    drop_remnant_vector: int
    drop_remnant_text: int
    drop_remnant_id_only: int
    drop_not_found: int
    drop_text_in_unallocated: int
    # What the unallocated-space scan covered (0s when it could not run).
    sqlite_unallocated_scanned: int
    sqlite_free_pages: int
    sqlite_freeblock_bytes: int
    sqlite_gap_bytes: int
    sqlite_unallocated_nonzero_bytes: int
    sqlite_wal_file_present: int
    # Four representative paths; content recovery is counted independently.
    delete_event_sqlite_carved: int = 0
    delete_event_sqlite_carved_rate: float = 0.0
    delete_event_text_recovered: int = 0
    delete_event_text_recovered_rate: float = 0.0
    drop_sqlite_carved: int = 0
    drop_text_recovered: int = 0
    superseded_sqlite_carved: int = 0


@dataclass
class _GroundTruth:
    """Reconstructed ground truth from replaying the operation log."""

    operation_summary: dict[str, dict[str, int]]
    total_operations: int
    add_success: int
    update_events: int
    upsert_events: int
    delete_events: int
    final_alive_count: int
    # Per distinct deleted id (net-deleted):
    # (doc_id, latest_hash, original_hash, all_version_hashes).
    deleted_ids: list[tuple[str, str | None, str | None, frozenset[str]]]
    # All expected vector hashes belonging to deleted ids.
    deleted_expected_hashes: set[str]
    # Dead (superseded) versions of ids that remain alive at the end.
    superseded_expected_hashes: set[str]
    # Event-based: (doc_id, prior-version vector hash) for every
    # update/upsert-on-existing event, in order. One entry per supersession,
    # so ``len(superseded_records) == update+upsert-existing events`` and the
    # identity ``update_events = full + identified + orphan + lost`` holds by
    # construction.
    superseded_records: list[tuple[str, str]]
    # Event-based: (doc_id, vector hash live at delete time) for every
    # successful delete operation, in order. One entry per delete event, so
    # ``len(delete_event_records) == delete_events``. A delete of an id whose
    # text was never observed contributes ``None`` and can never be graded as
    # recovered.
    delete_event_records: list[tuple[str, str | None]]
    # Event-based: (doc_id, vector hash live at drop time) for every item alive
    # in a collection when it was dropped.
    drop_events: int
    drop_event_records: list[tuple[str, str | None]]
    # Expected vector hash -> the text it was embedded from, for every hash in
    # the delete/drop records (the unallocated probe needs the text and bytes).
    text_by_hash: dict[str, str]
    # Every text the run embedded; text probes ignore windows two texts share.
    all_texts: set[str]


def _build_ground_truth(ops_db_path: Path, dimension: int) -> _GroundTruth:
    """Replay the operation log to reconstruct per-id vector history."""
    conn = sqlite3.connect(str(ops_db_path))
    try:
        summary = _operation_summary(conn)
        total_operations = conn.execute("SELECT COUNT(*) FROM operations").fetchone()[0]
        rows = conn.execute(
            """
            SELECT iteration, target_collection, operation, result, doc_id, document
            FROM operations
            ORDER BY iteration ASC
            """
        ).fetchall()
    finally:
        conn.close()

    original_text: dict[tuple[str, str], str] = {}
    latest_text: dict[tuple[str, str], str] = {}
    all_texts: dict[tuple[str, str], set[str]] = {}
    alive: dict[tuple[str, str], bool] = {}
    current_text: dict[tuple[str, str], str | None] = {}

    add_success = 0
    update_events = 0
    upsert_events = 0
    delete_events = 0
    superseded_records: list[tuple[str, str]] = []
    delete_event_records: list[tuple[str, str | None]] = []
    drop_events = 0
    drop_event_records: list[tuple[str, str | None]] = []
    text_by_hash: dict[str, str] = {}

    def live_hash(key: tuple[str, str]) -> str | None:
        text = current_text.get(key)
        if text is None:
            return None
        vec_hash = _expected_blob_hash(text, dimension)
        text_by_hash[vec_hash] = text
        return vec_hash

    for _iteration, collection, operation, result, doc_id, document in rows:
        if operation == "drop_collection":
            # Every item alive in the dropped collection is deleted here.
            drop_events += 1
            for key in [k for k, is_alive in alive.items() if is_alive and k[0] == str(collection)]:
                drop_event_records.append((key[1], live_hash(key)))
                alive[key] = False
                current_text[key] = None
            continue
        if str(result) == "0" or doc_id is None:
            continue
        key = (str(collection), str(doc_id))
        if operation == "add":
            add_success += 1
            alive[key] = True
            if document is not None:
                original_text.setdefault(key, document)
                latest_text[key] = document
                all_texts.setdefault(key, set()).add(document)
                current_text[key] = document
        elif operation in ("update", "upsert"):
            if operation == "update":
                update_events += 1
            else:
                upsert_events += 1
            # The version currently live for this id is being replaced: its
            # vector is superseded (soft-deleted in HNSW / left in an old WAL
            # Add record). Record it before overwriting the current version.
            prev = current_text.get(key)
            if prev is not None:
                superseded_records.append(
                    (str(doc_id), _expected_blob_hash(prev, dimension))
                )
            alive[key] = True
            if document is not None:
                original_text.setdefault(key, document)
                latest_text[key] = document
                all_texts.setdefault(key, set()).add(document)
                current_text[key] = document
        elif operation == "delete":
            delete_events += 1
            # The version live at delete time is exactly the vector that gets
            # soft-deleted into HNSW / left behind in the WAL. Record it per
            # event so the recovery rate is measured against delete operations
            # rather than against net-deleted ids.
            delete_event_records.append((str(doc_id), live_hash(key)))
            alive[key] = False
            current_text[key] = None
            # latest_text[key] is intentionally left as the text present at
            # delete time: that is the vector soft-deleted into HNSW.

    deleted_ids: list[tuple[str, str | None, str | None, frozenset[str]]] = []
    deleted_expected_hashes: set[str] = set()
    superseded_expected_hashes: set[str] = set()

    for key, is_alive in alive.items():
        latest = latest_text.get(key)
        original = original_text.get(key)
        latest_hash = _expected_blob_hash(latest, dimension) if latest else None
        original_hash = _expected_blob_hash(original, dimension) if original else None
        if not is_alive:
            # Every version this id ever held is now dead (nothing is live);
            # a recovered vector may be any of them (e.g. an intermediate update).
            all_hashes = {
                _expected_blob_hash(text, dimension)
                for text in all_texts.get(key, set())
            }
            if latest_hash:
                all_hashes.add(latest_hash)
            if original_hash:
                all_hashes.add(original_hash)
            deleted_ids.append(
                (key[1], latest_hash, original_hash, frozenset(all_hashes))
            )
            deleted_expected_hashes.update(all_hashes)
        else:
            # Superseded = every historical version except the current live one.
            for text in all_texts.get(key, set()):
                if text == latest:
                    continue
                superseded_expected_hashes.add(_expected_blob_hash(text, dimension))

    final_alive_count = sum(1 for is_alive in alive.values() if is_alive)

    return _GroundTruth(
        operation_summary=summary,
        total_operations=int(total_operations),
        add_success=add_success,
        update_events=update_events,
        upsert_events=upsert_events,
        delete_events=delete_events,
        final_alive_count=final_alive_count,
        deleted_ids=deleted_ids,
        deleted_expected_hashes=deleted_expected_hashes,
        superseded_expected_hashes=superseded_expected_hashes,
        superseded_records=superseded_records,
        delete_event_records=delete_event_records,
        drop_events=drop_events,
        drop_event_records=drop_event_records,
        text_by_hash=text_by_hash,
        all_texts={t for texts in all_texts.values() for t in texts},
    )


def _build_recovered_vector_map(
    emb_db_path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], set[str], int]:
    """Return (non-live recovery map, live map, all-source hash set, total rows).

    * ``recovered`` maps blob-hash -> {ids, has_chunk_doc} for ``source_chroma_sdk
      = 0`` rows only: a genuinely deleted id is absent from the live snapshot, so
      its residual vector must be recovered from WAL / HNSW rather than the live
      collection. This is the deletion residue, and everything is graded against
      it.
    * ``live`` is the same mapping built from ``source_chroma_sdk = 1`` rows. Used
      only for update-superseded accounting: a superseded version is not
      "permanently lost" when a later re-add makes the identical bytes live again.
    * ``all_hashes`` is the blob-hash set of **every** stored vector (live +
      WAL + HNSW).
    """
    index = RecoveryIndex(emb_db_path)
    with sqlite3.connect(f"file:{emb_db_path}?mode=ro", uri=True) as conn:
        total_recovered_rows = conn.execute('SELECT COUNT(*) FROM embeddings').fetchone()[0]
    recovered, live, all_hashes = {}, {}, set()
    for row in index.rows:
        blob_hash = row['vector_hash']
        if not blob_hash:
            continue
        all_hashes.add(blob_hash)
        if row['method'] != 'sdk' and index._active_or_ambiguous(row):
            continue
        target = live if row['method'] == 'sdk' else recovered
        info = target.setdefault(blob_hash, {'ids': set(), 'has_chunk_doc': False,
                                            'doc_ids': set(), 'wal_ids': set(), 'hnsw_ids': set(),
                                            'hnsw_orphan': False, 'carved': False})
        item_id = row['id']
        if item_id is not None:
            info['ids'].add(str(item_id))
        if row['document'] is not None:
            info['has_chunk_doc'] = True
            if item_id is not None:
                info['doc_ids'].add(str(item_id))
                if row['method'] in ('wal', 'sdk'):
                    info['wal_ids'].add(str(item_id))
        if row['method'] == 'hnsw':
            if item_id is None:
                info['hnsw_orphan'] = True
            else:
                info['hnsw_ids'].add(str(item_id))
        info['carved'] |= row['method'] == 'sqlite_carving'
    return recovered, live, all_hashes, int(total_recovered_rows)


def _grade_against(
    doc_id: str,
    vec_hash: str | None,
    vector_map: dict[str, dict[str, Any]],
) -> str:
    """Grade one expected vector against a recovered-vector map.

    Returns ``"full"`` (id matched and the row still carries the chunk text),
    ``"identified"`` (id matched, chunk text gone), ``"orphan"`` (the vector
    bytes are there but no row attributes them to this id) or ``"none"`` (the
    bytes are absent from this map entirely).
    """
    if not vec_hash or vec_hash not in vector_map:
        return 'none'
    info = vector_map[vec_hash]
    if doc_id in info['wal_ids']:
        return 'full'
    if doc_id in info['hnsw_ids']:
        return 'identified'
    if info['hnsw_orphan']:
        return 'orphan'
    if info['carved']:
        return 'carved'
    return 'none'


@dataclass
class _EventGrades:
    """Grades of one event list (delete or drop) against the deletion residue."""

    full: int = 0
    identified: int = 0
    orphan: int = 0
    unrecovered: int = 0
    carved: int = 0
    vector_count: int = 0
    text_count: int = 0
    details: list[dict] = field(default_factory=list)
    remnant_vector_text: int = 0
    remnant_vector: int = 0
    remnant_text: int = 0
    remnant_id_only: int = 0
    not_found: int = 0
    text_in_unallocated: int = 0

    @property
    def vector_recovered(self) -> int:
        return self.vector_count


def _grade_events(
    records: list[tuple[str, str | None]],
    recovered: dict[str, dict[str, Any]],
    truth: _GroundTruth,
    space: UnallocatedSpace | None,
    dimension: int,
    recovery_index: RecoveryIndex,
    event_type: str,
) -> _EventGrades:
    """Grade events into the three forms, then probe SQLite unallocated space.

    Unrecovered events are split into remnant classes; Identified/Orphan events
    are also checked for chunk text left in unallocated space, since that text is
    more than the form itself reveals.
    """
    grades = _EventGrades()
    known = KnownTexts(truth.all_texts)
    for doc_id, vec_hash in records:
        text = truth.text_by_hash.get(vec_hash) if vec_hash else None
        detail = recovery_index.grade(doc_id, vec_hash, text)
        detail.update(event_type=event_type, target_id=doc_id, vector_hash=vec_hash,
                      text_sha256=hashlib.sha256(text.encode()).hexdigest() if text is not None else None)
        grades.details.append(detail)
        grades.vector_count += detail['vector_recovered']
        grades.text_count += detail['text_recovered']
        grade = {'wal_full_entry': 'full', 'identified_vector': 'identified',
                 'orphan_vector': 'orphan', 'sqlite_carved_data': 'carved', 'none': 'none'}[detail['representative_path']]
        if grade == "full":
            grades.full += 1
            continue
        if grade == "identified":
            grades.identified += 1
        elif grade == "orphan":
            grades.orphan += 1
        elif grade == "carved":
            grades.carved += 1
        else:
            grades.unrecovered += 1

        if space is None:
            if grade == "none":
                grades.not_found += 1
            continue
        text = truth.text_by_hash.get(vec_hash) if vec_hash else None
        if grade == "none":
            found = probe_item(
                space, doc_id, _expected_blob(text, dimension) if text else None, text, known
            )
            cls = remnant_class(found)
            if cls == REMNANT_VECTOR_TEXT:
                grades.remnant_vector_text += 1
            elif cls == REMNANT_VECTOR:
                grades.remnant_vector += 1
            elif cls == REMNANT_TEXT:
                grades.remnant_text += 1
            elif cls == REMNANT_ID_ONLY:
                grades.remnant_id_only += 1
            else:
                grades.not_found += 1
        elif text and probe_item(space, doc_id, None, text, known)["text"]:
            grades.text_in_unallocated += 1
    return grades


def evaluate_run_verified(
    ops_db: Path,
    emb_db: Path,
    run_label: str,
    mode: str,
    dimension: int,
    chroma_sqlite: Path | None = None,
) -> VerifiedRecoveryMetrics:
    """Byte-verified recovery evaluation for a hash-embedding run.

    ``chroma_sqlite`` is the run's ``chroma.sqlite3``; when given, its unallocated
    space is read as raw bytes (never opened through SQLite) for the remnant check.
    """
    truth = _build_ground_truth(ops_db, dimension)
    recovered, live, all_hashes, total_recovered_rows = _build_recovered_vector_map(
        emb_db
    )

    id_verified = 0
    content_only = 0
    vector_recovered = 0
    full_chunk = 0
    latest_recovered = 0
    original_recovered = 0

    for doc_id, latest_hash, original_hash, id_version_hashes in truth.deleted_ids:
        found_latest = bool(latest_hash and latest_hash in recovered)
        found_original = bool(original_hash and original_hash in recovered)
        matched_hashes = [h for h in id_version_hashes if h in recovered]
        if found_latest:
            latest_recovered += 1
        if found_original:
            original_recovered += 1
        if not matched_hashes:
            continue
        vector_recovered += 1

        row_id_matches = False
        chunk_doc_present = False
        for candidate in matched_hashes:
            info = recovered[candidate]
            if doc_id in info["ids"]:
                row_id_matches = True
                if doc_id in info['doc_ids']:
                    chunk_doc_present = True
        if row_id_matches:
            id_verified += 1
            if chunk_doc_present:
                full_chunk += 1
        else:
            content_only += 1

    # Cross-check: explain every non-live recovered vector.
    nonlive_hashes = set(recovered.keys())
    matched_deleted = len(nonlive_hashes & truth.deleted_expected_hashes)
    remaining = nonlive_hashes - truth.deleted_expected_hashes
    matched_superseded = len(remaining & truth.superseded_expected_hashes)
    unexplained = len(remaining - truth.superseded_expected_hashes)

    space: UnallocatedSpace | None = None
    if chroma_sqlite is not None and chroma_sqlite.exists():
        space = read_unallocated_space(chroma_sqlite)

    # Delete-event grading: one entry per delete operation, graded on the vector
    # that was live when that delete ran. Denominator is the delete-operation
    # count, so the four classes partition truth.delete_events exactly.
    recovery_index = RecoveryIndex(emb_db)
    dels = _grade_events(truth.delete_event_records, recovered, truth, space, dimension, recovery_index, "delete")
    del_full, del_ident, del_orphan = dels.full, dels.identified, dels.orphan
    del_unrecovered, del_vector_recovered = dels.unrecovered, dels.vector_recovered
    drops = _grade_events(truth.drop_event_records, recovered, truth, space, dimension, recovery_index, "drop")
    save_targets(emb_db.parent / "recovery_targets.sqlite3", dels.details + drops.details)

    # Update-superseded vector accounting (event-based): of every prior version
    # replaced by an update, grade it exactly like a delete-event vector against
    # the deletion residue. A version whose bytes are absent from the residue but
    # live again (identical text re-added) is not permanently lost, so it is
    # graded against the live snapshot instead and counted separately. Anything
    # whose bytes survive nowhere is PERMANENTLY LOST.
    superseded_total = len(truth.superseded_records)
    sup_full = sup_ident = sup_orphan = sup_carved = sup_lost = 0
    sup_via_live = 0
    for doc_id, vec_hash in truth.superseded_records:
        grade = _grade_against(doc_id, vec_hash, recovered)
        if grade == "none":
            live_grade = _grade_against(doc_id, vec_hash, live)
            if live_grade != "none":
                grade = live_grade
                sup_via_live += 1
        if grade == "full":
            sup_full += 1
        elif grade == "identified":
            sup_ident += 1
        elif grade == "orphan":
            sup_orphan += 1
        elif grade == "carved":
            sup_carved += 1
        else:
            sup_lost += 1
    superseded_recovered = sup_full + sup_ident + sup_orphan + sup_carved

    n_ids = len(truth.deleted_ids)
    n_events = truth.delete_events

    return VerifiedRecoveryMetrics(
        run_label=run_label,
        embedding_mode=mode,
        embedding_dimension=dimension,
        operation_summary=json.dumps(
            truth.operation_summary, ensure_ascii=False, sort_keys=True
        ),
        total_operations=truth.total_operations,
        add_success=truth.add_success,
        update_events=truth.update_events,
        upsert_events=truth.upsert_events,
        delete_events=truth.delete_events,
        distinct_deleted_ids=n_ids,
        final_alive_count=truth.final_alive_count,
        total_recovered_rows=total_recovered_rows,
        deleted_id_verified=id_verified,
        deleted_content_only=content_only,
        deleted_vector_recovered=vector_recovered,
        deleted_full_chunk=full_chunk,
        deleted_latest_recovered=latest_recovered,
        deleted_original_recovered=original_recovered,
        id_verified_rate_by_id=_percent(id_verified, n_ids),
        content_only_rate_by_id=_percent(content_only, n_ids),
        combined_rate_by_id=_percent(vector_recovered, n_ids),
        full_chunk_rate_by_id=_percent(full_chunk, n_ids),
        delete_event_full_chunk=del_full,
        delete_event_identified_only=del_ident,
        delete_event_orphan_only=del_orphan,
        delete_event_unrecovered=del_unrecovered,
        delete_event_full_chunk_rate=_percent(del_full, n_events),
        delete_event_identified_only_rate=_percent(del_ident, n_events),
        delete_event_orphan_only_rate=_percent(del_orphan, n_events),
        delete_event_unrecovered_rate=_percent(del_unrecovered, n_events),
        delete_event_vector_recovered=del_vector_recovered,
        delete_event_vector_recovered_rate=_percent(del_vector_recovered, n_events),
        recovered_nonlive_vectors=len(nonlive_hashes),
        recovered_matched_deleted=matched_deleted,
        recovered_matched_superseded=matched_superseded,
        recovered_unexplained=unexplained,
        superseded_total=superseded_total,
        superseded_full_chunk=sup_full,
        superseded_identified_only=sup_ident,
        superseded_orphan_only=sup_orphan,
        superseded_lost=sup_lost,
        superseded_full_chunk_rate=_percent(sup_full, superseded_total),
        superseded_identified_only_rate=_percent(sup_ident, superseded_total),
        superseded_orphan_only_rate=_percent(sup_orphan, superseded_total),
        superseded_lost_rate=_percent(sup_lost, superseded_total),
        superseded_recovered=superseded_recovered,
        superseded_recovered_rate=_percent(superseded_recovered, superseded_total),
        superseded_recovered_via_live_readd=sup_via_live,
        delete_event_remnant_vector_text=dels.remnant_vector_text,
        delete_event_remnant_vector=dels.remnant_vector,
        delete_event_remnant_text=dels.remnant_text,
        delete_event_remnant_id_only=dels.remnant_id_only,
        delete_event_not_found=dels.not_found,
        delete_event_text_in_unallocated=dels.text_in_unallocated,
        drop_events=truth.drop_events,
        drop_items=len(truth.drop_event_records),
        drop_full_chunk=drops.full,
        drop_identified_only=drops.identified,
        drop_orphan_only=drops.orphan,
        drop_unrecovered=drops.unrecovered,
        drop_vector_recovered=drops.vector_recovered,
        drop_remnant_vector_text=drops.remnant_vector_text,
        drop_remnant_vector=drops.remnant_vector,
        drop_remnant_text=drops.remnant_text,
        drop_remnant_id_only=drops.remnant_id_only,
        drop_not_found=drops.not_found,
        drop_text_in_unallocated=drops.text_in_unallocated,
        sqlite_unallocated_scanned=int(space is not None),
        sqlite_free_pages=space.free_pages if space else 0,
        sqlite_freeblock_bytes=space.freeblock_bytes if space else 0,
        sqlite_gap_bytes=space.gap_bytes if space else 0,
        sqlite_unallocated_nonzero_bytes=space.nonzero_bytes if space else 0,
        sqlite_wal_file_present=int(space.wal_file_present) if space else 0,
        delete_event_sqlite_carved=dels.carved,
        delete_event_sqlite_carved_rate=_percent(dels.carved, n_events),
        delete_event_text_recovered=dels.text_count,
        delete_event_text_recovered_rate=_percent(dels.text_count, n_events),
        drop_sqlite_carved=drops.carved,
        drop_text_recovered=drops.text_count,
        superseded_sqlite_carved=sup_carved,
    )


def _print_remnants(prefix: str, m: VerifiedRecoveryMetrics, total: int) -> None:
    """Print the SQLite remnant split of one event group's unrecovered items."""
    if not m.sqlite_unallocated_scanned:
        return
    rows = [
        ("vector + chunk text", getattr(m, f"{prefix}_remnant_vector_text")),
        ("vector", getattr(m, f"{prefix}_remnant_vector")),
        ("chunk text", getattr(m, f"{prefix}_remnant_text")),
        ("id only", getattr(m, f"{prefix}_remnant_id_only")),
        ("not found anywhere", getattr(m, f"{prefix}_not_found")),
    ]
    print("      of which left in SQLite unallocated space (knowledge-based):")
    for label, count in rows:
        print(f"        {label:<21}: {count} ({_percent(count, total):.2f}%)")
    text_extra = getattr(m, f"{prefix}_text_in_unallocated")
    if text_extra:
        print(
            "      Identified/Orphan whose chunk text is also in unallocated "
            f"space: {text_extra}"
        )


def print_verified_metrics(m: VerifiedRecoveryMetrics) -> None:
    """Human-readable summary for the verified path."""
    print(f"\n{'='*66}")
    print(f"  Verified Recovery Evaluation: {m.run_label}")
    print(f"  (hash embedding, dimension={m.embedding_dimension}, byte-verified)")
    print(f"{'='*66}")
    print("  Operation summary:")
    for operation, stats in json.loads(m.operation_summary).items():
        print(
            f"    {operation:<14}: "
            f"{stats['successes']} success / {stats['attempts']} attempts"
        )
    print(f"  Total operations       : {m.total_operations}")
    print(f"  Final alive count      : {m.final_alive_count}")
    print(
        "  Mutations              : "
        f"update={m.update_events}, upsert={m.upsert_events}, "
        f"delete_events={m.delete_events}"
    )
    print(
        "  Deleted denominators   : "
        f"distinct_ids={m.distinct_deleted_ids}, delete_events={m.delete_events}"
    )
    print(f"  --- Delete recovery (per delete operation, n={m.delete_events}) ---")
    print(
        f"    WAL Full Entry         : {m.delete_event_full_chunk} "
        f"({m.delete_event_full_chunk_rate:.2f}%)"
    )
    print(
        f"    Identified Vector      : {m.delete_event_identified_only} "
        f"({m.delete_event_identified_only_rate:.2f}%)"
    )
    print(
        f"    Orphan Vector          : {m.delete_event_orphan_only} "
        f"({m.delete_event_orphan_only_rate:.2f}%)"
    )
    print(f"    SQLite Carved Data     : {m.delete_event_sqlite_carved} ({m.delete_event_sqlite_carved_rate:.2f}%)")
    print(f"    Original text recovered: {m.delete_event_text_recovered} ({m.delete_event_text_recovered_rate:.2f}%)")
    print(
        f"    Not recovered          : {m.delete_event_unrecovered} "
        f"({m.delete_event_unrecovered_rate:.2f}%)"
    )
    print(
        f"    Vector recovery total  : {m.delete_event_vector_recovered} "
        f"({m.delete_event_vector_recovered_rate:.2f}%)"
    )
    _print_remnants("delete_event", m, m.delete_events)
    if m.drop_events:
        n = m.drop_items
        print(
            f"  --- Collection drop (per dropped item, drops={m.drop_events}, n={n}) ---"
        )
        for label, count in (
            ("WAL Full Entry", m.drop_full_chunk),
            ("Identified Vector", m.drop_identified_only),
            ("Orphan Vector", m.drop_orphan_only),
            ("SQLite Carved Data", m.drop_sqlite_carved),
            ("Original text recovered", m.drop_text_recovered),
            ("Not recovered", m.drop_unrecovered),
            ("Vector recovery total", m.drop_vector_recovered),
        ):
            print(f"    {label:<23}: {count} ({_percent(count, n):.2f}%)")
        _print_remnants("drop", m, n)
    print(f"  --- Deleted-id recovery (per distinct id, n={m.distinct_deleted_ids}) ---")
    print(
        f"    id-verified          : {m.deleted_id_verified} "
        f"({m.id_verified_rate_by_id:.2f}%)"
    )
    print(
        f"    content-only (orphan): {m.deleted_content_only} "
        f"({m.content_only_rate_by_id:.2f}%)"
    )
    print(
        f"    combined recovered   : {m.deleted_vector_recovered} "
        f"({m.combined_rate_by_id:.2f}%)"
    )
    print(
        f"    Full Entry           : {m.deleted_full_chunk} "
        f"({m.full_chunk_rate_by_id:.2f}%)"
    )
    print(
        "    (latest-vector hit="
        f"{m.deleted_latest_recovered}, original-vector hit="
        f"{m.deleted_original_recovered})"
    )
    print("  --- Non-live recovered vectors (cross-check) ---")
    print(f"    total non-live       : {m.recovered_nonlive_vectors}")
    print(f"    matched deleted      : {m.recovered_matched_deleted}")
    print(f"    matched superseded   : {m.recovered_matched_superseded}")
    print(f"    unexplained          : {m.recovered_unexplained}")
    if m.superseded_total:
        print(
            "  --- Update-superseded recovery (per update operation, "
            f"n={m.superseded_total}) ---"
        )
        print(
            f"    Full Entry             : {m.superseded_full_chunk} "
            f"({m.superseded_full_chunk_rate:.2f}%)"
        )
        print(
            f"    Identified Vector      : {m.superseded_identified_only} "
            f"({m.superseded_identified_only_rate:.2f}%)"
        )
        print(
            f"    Orphan Vector          : {m.superseded_orphan_only} "
            f"({m.superseded_orphan_only_rate:.2f}%)"
        )
        print(
            f"    PERMANENTLY LOST       : {m.superseded_lost} "
            f"({m.superseded_lost_rate:.2f}%)"
        )
        print(
            f"    Vector recovery total  : {m.superseded_recovered} "
            f"({m.superseded_recovered_rate:.2f}%)"
        )
        if m.superseded_recovered_via_live_readd:
            print(
                "    (of which recovered only via an identical live re-add: "
                f"{m.superseded_recovered_via_live_readd})"
            )
    if m.sqlite_unallocated_scanned:
        print("  --- SQLite unallocated space scanned (chroma.sqlite3) ---")
        print(
            f"    free pages={m.sqlite_free_pages}, freeblock bytes={m.sqlite_freeblock_bytes}, "
            f"in-page gap bytes={m.sqlite_gap_bytes}, "
            f"non-zero bytes={m.sqlite_unallocated_nonzero_bytes}"
        )
        if m.sqlite_wal_file_present:
            print("    [WARN] chroma.sqlite3-wal exists and was not scanned.")
    print(f"{'='*66}\n")


_SQL_TYPES = {"int": "INTEGER", "float": "REAL", "str": "TEXT"}


def save_verified_metrics(metrics: VerifiedRecoveryMetrics, output_path: Path) -> None:
    """Persist verified evaluation metrics to a SQLite database.

    The table has one column per ``VerifiedRecoveryMetrics`` field, in field order.
    """
    columns = [
        f"{f.name} {_SQL_TYPES[str(f.type)]}"
        + (" PRIMARY KEY" if f.name == "run_label" else "")
        for f in fields(VerifiedRecoveryMetrics)
    ]
    names = [f.name for f in fields(VerifiedRecoveryMetrics)]
    conn = sqlite3.connect(str(output_path))
    conn.execute("DROP TABLE IF EXISTS evaluation")
    conn.execute(f"CREATE TABLE evaluation ({', '.join(columns)})")
    conn.execute(
        f"INSERT OR REPLACE INTO evaluation ({', '.join(names)}) "
        f"VALUES ({', '.join(':' + n for n in names)})",
        asdict(metrics),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Legacy presence path (default embeddings / pre-ground-truth logs)
# ---------------------------------------------------------------------------


@dataclass
class RecoveryMetrics:
    """Presence-based recovery evaluation (no vector verification)."""

    run_label: str
    operation_summary: str
    total_operations: int
    total_adds: int
    total_deletes: int
    final_alive_count: int
    total_recovered: int
    deleted_full_chunk_recovered: int
    deleted_identified_vector_only_recovered: int
    deleted_orphan_vector_only_recovered: int
    deleted_vector_recovered_total: int
    deleted_by_wal: int
    deleted_by_hnsw: int
    deleted_by_hnsw_orphan: int
    deleted_recovered_identified: int
    deleted_full_chunk_recovery_rate: float
    deleted_identified_vector_only_recovery_rate: float
    deleted_orphan_vector_only_recovery_rate: float
    deleted_vector_recovery_rate: float
    deleted_recovery_rate: float


def _query_operations(ops_db_path: Path) -> dict[str, Any]:
    """Extract ground-truth stats from the generator's operations log."""
    conn = sqlite3.connect(str(ops_db_path))
    try:
        operation_summary = _operation_summary(conn)
        total_operations = conn.execute(
            "SELECT COUNT(*) FROM operations"
        ).fetchone()[0]
        total_adds = conn.execute(
            "SELECT COUNT(*) FROM operations WHERE operation='add' AND result='1'"
        ).fetchone()[0]
        total_deletes = conn.execute(
            "SELECT COUNT(*) FROM operations WHERE operation='delete' AND result='1'"
        ).fetchone()[0]
        rows = conn.execute(
            """
            SELECT target_collection, collection_count
            FROM operations o
            WHERE iteration = (
                SELECT MAX(iteration) FROM operations
                WHERE target_collection = o.target_collection
            )
            GROUP BY target_collection
            """
        ).fetchall()
        final_alive_count = sum(row[1] for row in rows) if rows else 0
    finally:
        conn.close()
    return {
        "operation_summary": operation_summary,
        "total_operations": total_operations,
        "total_adds": total_adds,
        "total_deletes": total_deletes,
        "final_alive_count": final_alive_count,
    }


def _query_recovery(emb_db_path: Path) -> dict[str, Any]:
    """Extract recovery stats from the parser's embeddings output."""
    conn = sqlite3.connect(str(emb_db_path))
    try:
        total_recovered = conn.execute(
            "SELECT COUNT(*) FROM embeddings"
        ).fetchone()[0]
        recovery_rows = conn.execute(
            """
            SELECT
                id,
                MAX(CASE WHEN vector IS NOT NULL THEN 1 ELSE 0 END) AS has_vector,
                MAX(
                    CASE
                        WHEN vector IS NOT NULL
                        AND metadata LIKE '%"chroma:document"%'
                        THEN 1
                        ELSE 0
                    END
                ) AS has_chunk
            FROM embeddings
            WHERE source_chroma_sdk = 0 AND id IS NOT NULL
            GROUP BY id
            """
        ).fetchall()
        full_chunk = sum(1 for _, has_vector, has_chunk in recovery_rows if has_vector and has_chunk)
        vector_only = sum(
            1
            for _, has_vector, has_chunk in recovery_rows
            if has_vector and not has_chunk
        )
        identified = full_chunk + vector_only
        by_wal = conn.execute(
            "SELECT COUNT(DISTINCT id) FROM embeddings "
            "WHERE source_chroma_sdk = 0 AND source_wal = 1 "
            "AND vector IS NOT NULL AND id IS NOT NULL"
        ).fetchone()[0]
        by_hnsw = conn.execute(
            "SELECT COUNT(DISTINCT id) FROM embeddings "
            "WHERE source_chroma_sdk = 0 AND source_hnsw = 1 "
            "AND vector IS NOT NULL AND id IS NOT NULL"
        ).fetchone()[0]
        # HNSW orphans: vector recovered but no ID mapping
        orphan = conn.execute(
            "SELECT COUNT(*) FROM embeddings "
            "WHERE source_chroma_sdk = 0 AND source_hnsw = 1 AND id IS NULL"
        ).fetchone()[0]
        orphan_vector_only = conn.execute(
            "SELECT COUNT(*) FROM embeddings "
            "WHERE source_chroma_sdk = 0 AND source_hnsw = 1 "
            "AND id IS NULL AND vector IS NOT NULL"
        ).fetchone()[0]
    finally:
        conn.close()
    return {
        "total_recovered": total_recovered,
        "deleted_full_chunk_recovered": full_chunk,
        "deleted_identified_vector_only_recovered": vector_only,
        "deleted_orphan_vector_only_recovered": orphan_vector_only,
        "deleted_vector_recovered_total": full_chunk + vector_only + orphan_vector_only,
        "deleted_recovered_identified": identified,
        "deleted_by_wal": by_wal,
        "deleted_by_hnsw": by_hnsw,
        "deleted_by_hnsw_orphan": orphan,
    }


def evaluate_run_legacy(
    ops_db: Path,
    emb_db: Path,
    run_label: str,
) -> RecoveryMetrics:
    """Presence-based recovery evaluation (unverified)."""
    ops = _query_operations(ops_db)
    rec = _query_recovery(emb_db)

    total_deletes = ops["total_deletes"]
    identified = rec["deleted_recovered_identified"]

    rate = _percent(identified, total_deletes)
    full_chunk_rate = _percent(rec["deleted_full_chunk_recovered"], total_deletes)
    identified_vector_only_rate = _percent(
        rec["deleted_identified_vector_only_recovered"], total_deletes
    )
    orphan_vector_only_rate = _percent(
        rec["deleted_orphan_vector_only_recovered"], total_deletes
    )
    vector_recovery_rate = _percent(rec["deleted_vector_recovered_total"], total_deletes)

    return RecoveryMetrics(
        run_label=run_label,
        operation_summary=json.dumps(
            ops["operation_summary"], ensure_ascii=False, sort_keys=True
        ),
        total_operations=ops["total_operations"],
        total_adds=ops["total_adds"],
        total_deletes=total_deletes,
        final_alive_count=ops["final_alive_count"],
        total_recovered=rec["total_recovered"],
        deleted_full_chunk_recovered=rec["deleted_full_chunk_recovered"],
        deleted_identified_vector_only_recovered=rec[
            "deleted_identified_vector_only_recovered"
        ],
        deleted_orphan_vector_only_recovered=rec[
            "deleted_orphan_vector_only_recovered"
        ],
        deleted_vector_recovered_total=rec["deleted_vector_recovered_total"],
        deleted_by_wal=rec["deleted_by_wal"],
        deleted_by_hnsw=rec["deleted_by_hnsw"],
        deleted_by_hnsw_orphan=rec["deleted_by_hnsw_orphan"],
        deleted_recovered_identified=identified,
        deleted_full_chunk_recovery_rate=full_chunk_rate,
        deleted_identified_vector_only_recovery_rate=identified_vector_only_rate,
        deleted_orphan_vector_only_recovery_rate=orphan_vector_only_rate,
        deleted_vector_recovery_rate=vector_recovery_rate,
        deleted_recovery_rate=rate,
    )


def print_metrics(m: RecoveryMetrics) -> None:
    """Print a human-readable summary of legacy recovery metrics."""
    orphan_suffix = f"  (+{m.deleted_by_hnsw_orphan} orphan)" if m.deleted_by_hnsw_orphan else ""
    print(f"\n{'='*60}")
    print(f"  Recovery Evaluation: {m.run_label}")
    print(f"{'='*60}")
    print("  Operation summary:")
    for operation, stats in json.loads(m.operation_summary).items():
        print(
            f"    {operation:<14}: "
            f"{stats['successes']} success / {stats['attempts']} attempts"
        )
    print(f"  Total operations : {m.total_operations}")
    print(f"  Final alive count: {m.final_alive_count}")
    print(f"  Deletes          : {m.total_deletes}")
    print(f"  WAL recovered    : {m.deleted_by_wal}")
    print(f"  HNSW recovered   : {m.deleted_by_hnsw}{orphan_suffix}")
    print("  Recovery classes:")
    print(
        "    Full Entry                : "
        f"{m.deleted_full_chunk_recovered} / {m.total_deletes} "
        f"({m.deleted_full_chunk_recovery_rate:.2f}%)"
    )
    print(
        "    Identified Vector         : "
        f"{m.deleted_identified_vector_only_recovered} / {m.total_deletes} "
        f"({m.deleted_identified_vector_only_recovery_rate:.2f}%)"
    )
    print(
        "    Orphan Vector             : "
        f"{m.deleted_orphan_vector_only_recovered} / {m.total_deletes} "
        f"({m.deleted_orphan_vector_only_recovery_rate:.2f}%)"
    )
    print(
        "  Identified recovery total   : "
        f"{m.deleted_recovered_identified} / {m.total_deletes} "
        f"({m.deleted_recovery_rate:.2f}%)"
    )
    print(
        "  Vector recovery total       : "
        f"{m.deleted_vector_recovered_total} / {m.total_deletes} "
        f"({m.deleted_vector_recovery_rate:.2f}%)"
    )
    print(f"{'='*60}\n")


def save_metrics(metrics: RecoveryMetrics, output_path: Path) -> None:
    """Persist legacy evaluation metrics to a SQLite database."""
    conn = sqlite3.connect(str(output_path))
    conn.execute("DROP TABLE IF EXISTS evaluation")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS evaluation (
            run_label TEXT PRIMARY KEY,
            operation_summary TEXT,
            total_operations INTEGER,
            total_adds INTEGER,
            total_deletes INTEGER,
            final_alive_count INTEGER,
            total_recovered INTEGER,
            deleted_full_chunk_recovered INTEGER,
            deleted_identified_vector_only_recovered INTEGER,
            deleted_orphan_vector_only_recovered INTEGER,
            deleted_vector_recovered_total INTEGER,
            deleted_by_wal INTEGER,
            deleted_by_hnsw INTEGER,
            deleted_by_hnsw_orphan INTEGER,
            deleted_recovered_identified INTEGER,
            deleted_full_chunk_recovery_rate REAL,
            deleted_identified_vector_only_recovery_rate REAL,
            deleted_orphan_vector_only_recovery_rate REAL,
            deleted_vector_recovery_rate REAL,
            deleted_recovery_rate REAL
        )
        """
    )
    conn.execute(
        """
        INSERT OR REPLACE INTO evaluation VALUES (
            :run_label, :operation_summary, :total_operations,
            :total_adds, :total_deletes, :final_alive_count,
            :total_recovered, :deleted_full_chunk_recovered,
            :deleted_identified_vector_only_recovered,
            :deleted_orphan_vector_only_recovered,
            :deleted_vector_recovered_total,
            :deleted_by_wal, :deleted_by_hnsw, :deleted_by_hnsw_orphan,
            :deleted_recovered_identified, :deleted_full_chunk_recovery_rate,
            :deleted_identified_vector_only_recovery_rate,
            :deleted_orphan_vector_only_recovery_rate,
            :deleted_vector_recovery_rate,
            :deleted_recovery_rate
        )
        """,
        asdict(metrics),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _evaluate_residue_lifetime(
    ops_db: Path, emb_db: Path, report_dir: Path, eval_output: Path, dimension: int
) -> None:
    """Replay the log for residue lifetimes and check them against observed forms."""
    thresholds = residue_lifetime.thresholds_by_name(report_dir / "sysdb.sqlite3")
    # Dropped collections are gone from sysdb; fall back to the threshold the run logged.
    default_threshold = (
        residue_lifetime.logged_sync_threshold(ops_db) or residue_lifetime.DEFAULT_SYNC_THRESHOLD
    )
    events, to_next, drops = residue_lifetime.replay_lifetimes(ops_db, thresholds, default_threshold)
    truth = _build_ground_truth(ops_db, dimension)
    recovered, *_ = _build_recovered_vector_map(emb_db)
    observed = [
        (doc_id, _grade_against(doc_id, vec_hash, recovered))
        for doc_id, vec_hash in truth.delete_event_records
    ]
    if len(observed) == len(events) and all(
        doc_id == event.doc_id for (doc_id, _), event in zip(observed, events)
    ):
        for (_, grade), event in zip(observed, events):
            event.observed_form = grade
    else:
        print("[WARN] Delete events do not line up with the ground truth; skipping the model check.")
    observed_drop = {
        doc_id: _grade_against(doc_id, vec_hash, recovered)
        for doc_id, vec_hash in truth.drop_event_records
    }
    for drop in drops:
        drop.observed_form = observed_drop.get(drop.doc_id)
    summary = residue_lifetime.summarize(events, thresholds, to_next, drops, default_threshold)
    residue_lifetime.print_summary(summary)
    residue_lifetime.save(events, summary, eval_output, drops)


def main(args: Namespace | None = None) -> None:
    """Evaluate recovery rate for one run."""
    if args is None:
        args = get_default_args()

    chroma_run_dir = args.chroma_run_dir.resolve()
    report_base = args.report_base.resolve()
    run_label = chroma_run_dir.name
    report_dir = report_base / run_label

    ops_db = chroma_run_dir / "operations.sqlite3"
    emb_db = report_dir / "embeddings.sqlite3"

    print(f"[INFO] Chroma run directory : {chroma_run_dir}")
    print(f"[INFO] Report directory     : {report_dir}")

    if not ops_db.exists():
        print(f"[ERROR] operations.sqlite3 not found: {ops_db}")
        return
    if not emb_db.exists():
        print(f"[ERROR] embeddings.sqlite3 not found: {emb_db}")
        return

    embedding = get_embedding_settings()
    verifiable = embedding.mode.strip().lower() == "hash" and _has_ground_truth_columns(
        ops_db
    )

    eval_output = report_dir / "evaluation.sqlite3"

    if verifiable:
        metrics = evaluate_run_verified(
            ops_db,
            emb_db,
            run_label,
            embedding.mode,
            embedding.dimension,
            chroma_sqlite=chroma_run_dir / "chroma_db" / "chroma.sqlite3",
        )
        print_verified_metrics(metrics)
        save_verified_metrics(metrics, eval_output)
        _evaluate_residue_lifetime(ops_db, emb_db, report_dir, eval_output, embedding.dimension)
        print(f"[INFO] Verified evaluation saved: {eval_output}")
        return

    if embedding.mode.strip().lower() == "hash":
        print(
            "[WARN] operations.sqlite3 lacks doc_id/document columns; "
            "falling back to unverified recovery counts. "
            "Regenerate this run to enable byte-verified evaluation."
        )
    else:
        print(
            f"[INFO] Embedding mode '{embedding.mode}' is not hash; "
            "using unverified presence-based recovery counts."
        )

    with sqlite3.connect(f"file:{ops_db}?mode=ro", uri=True) as conn:
        if "drop_collection" in _operation_summary(conn):
            print(
                "[WARN] This run drops a collection; collection-drop grading and the "
                "SQLite remnant check need the verified path (hash embeddings)."
            )
    metrics = evaluate_run_legacy(ops_db, emb_db, run_label)
    print_metrics(metrics)
    save_metrics(metrics, eval_output)
    print(f"[INFO] Evaluation saved: {eval_output}")
