"""
Residue lifetime model (knowledge-based).

A deleted item stays a Full Entry or Identified Vector until its collection's
vector segment next persists; from then on it is an Orphan Vector. Chroma
persists once the log records applied since the last persist reach the
collection's ``sync_threshold``. Replaying the operation log with that rule
gives, for every delete event:

* ``records_to_clear``: records written to the collection after the delete
  until the persist that turned it into an Orphan Vector (None while pending);
* ``records_waited``: for events still pending at the snapshot, records written
  to the collection since the delete;
* ``model_form``: the form the model predicts at the snapshot (``full``,
  ``identified``, ``orphan``, or ``dropped`` when the collection was dropped
  before any persist reached the delete).

For every item alive in a collection when it is dropped, the replay also predicts
whether the dropped segment directory still holds it: ``identified`` when its
vector record had been persisted before the drop, ``not_persisted`` otherwise.

Comparing ``model_form`` with the form the evaluator observed checks the model
and the tool against each other. Lifetimes are counted in log records: multiply
by the collection's write rate to get time. With no further writes, pending
residue never clears.

The replay assumes the generator's pattern of one log record per write call in
a single process, so a collection persists exactly when its record count since
the last persist reaches its threshold.
"""

from __future__ import annotations

import bisect
import json
import sqlite3
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from chroma_hearts.parser.persistence_status import (
    DEFAULT_SYNC_THRESHOLD,
    collection_sync_thresholds,
)

_WRITE_OPERATIONS = {"add", "update", "upsert", "delete"}


@dataclass
class LifetimeEvent:
    """Lifetime of one delete event, in log records of its collection."""

    doc_id: str
    collection: str
    vector_record: int | None
    delete_record: int
    persist_record: int | None
    records_to_clear: int | None
    records_waited: int | None
    model_form: str
    observed_form: str | None = None


@dataclass
class DropPrediction:
    """Whether a dropped item's vector was persisted before the drop."""

    doc_id: str
    collection: str
    vector_record: int
    last_persist_record: int
    model_form: str
    observed_form: str | None = None


@dataclass
class LifetimeSummary:
    """Run-level summary of :class:`LifetimeEvent` rows."""

    sync_thresholds: str
    delete_events: int
    cleared_events: int
    records_to_clear_median: int
    records_to_clear_p90: int
    records_to_clear_max: int
    pending_events: int
    pending_records_waited_max: int
    records_to_next_persist: str
    dropped_events: int
    model_compared: int
    model_agreement: int
    drop_items: int
    drop_predicted_identified: int
    drop_model_compared: int
    drop_model_agreement: int


def thresholds_by_name(sysdb_sqlite: Path) -> dict[str, int]:
    """Collection name -> sync_threshold, from the parser's copied sysdb."""
    if not sysdb_sqlite.exists():
        return {}
    with sqlite3.connect(f"file:{sysdb_sqlite}?mode=ro", uri=True) as conn:
        return {name: t for name, t in collection_sync_thresholds(conn).values()}


def logged_sync_threshold(ops_db: Path) -> int | None:
    """The sync_threshold the generator recorded in the operation log, if any."""
    with sqlite3.connect(f"file:{ops_db}?mode=ro", uri=True) as conn:
        has_table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'run_config'"
        ).fetchone()
        if not has_table:
            return None
        row = conn.execute("SELECT value FROM run_config WHERE key = 'sync_threshold'").fetchone()
    return int(row[0]) if row else None


def replay_lifetimes(
    ops_db: Path, thresholds: dict[str, int], default_threshold: int = DEFAULT_SYNC_THRESHOLD
) -> tuple[list[LifetimeEvent], dict[str, int], list[DropPrediction]]:
    """Replay the operation log; return the delete events, the records each
    collection still needs before its next persist, and drop predictions."""
    conn = sqlite3.connect(f"file:{ops_db}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            """
            SELECT iteration, target_collection, operation, result, doc_id
            FROM operations
            ORDER BY iteration ASC
            """
        ).fetchall()
    finally:
        conn.close()

    records: dict[str, int] = {}
    since_persist: dict[str, int] = {}
    persists: dict[str, list[int]] = {}
    last_vector_record: dict[tuple[str, str], int] = {}
    dropped: set[str] = set()
    events: list[LifetimeEvent] = []
    alive: dict[tuple[str, str], bool] = {}
    drops: list[DropPrediction] = []

    for _iteration, collection, operation, result, doc_id in rows:
        collection = str(collection)
        if operation == "drop_collection":
            dropped.add(collection)
            last_persist = (persists.get(collection) or [0])[-1]
            for (c, doc), is_alive in alive.items():
                if c == collection and is_alive:
                    vector_record = last_vector_record[(c, doc)]
                    drops.append(
                        DropPrediction(
                            doc_id=doc,
                            collection=collection,
                            vector_record=vector_record,
                            last_persist_record=last_persist,
                            model_form="identified" if vector_record <= last_persist else "not_persisted",
                        )
                    )
                    alive[(c, doc)] = False
            continue
        if str(result) == "0" or doc_id is None or operation not in _WRITE_OPERATIONS:
            continue
        record = records[collection] = records.get(collection, 0) + 1
        key = (collection, str(doc_id))
        if operation == "delete":
            events.append(
                LifetimeEvent(
                    doc_id=str(doc_id),
                    collection=collection,
                    vector_record=last_vector_record.get(key),
                    delete_record=record,
                    persist_record=None,
                    records_to_clear=None,
                    records_waited=None,
                    model_form="pending",
                )
            )
        else:
            last_vector_record[key] = record
        alive[key] = operation != "delete"
        since_persist[collection] = since_persist.get(collection, 0) + 1
        if since_persist[collection] >= thresholds.get(collection, default_threshold):
            persists.setdefault(collection, []).append(record)
            since_persist[collection] = 0

    for event in events:
        points = persists.get(event.collection, [])
        i = bisect.bisect_left(points, event.delete_record)
        if i < len(points):
            event.persist_record = points[i]
            event.records_to_clear = points[i] - event.delete_record
            event.model_form = "orphan"
        elif event.collection in dropped:
            event.model_form = "dropped"
        else:
            event.records_waited = records[event.collection] - event.delete_record
            # The add record survives pruning unless a later persist passed it
            # (pruning removes seq_id < the persisted seq, so equal survives).
            add_pruned = bool(points) and event.vector_record is not None and points[-1] > event.vector_record
            event.model_form = "identified" if add_pruned else "full"

    to_next = {
        c: thresholds.get(c, default_threshold) - since_persist.get(c, 0)
        for c in {e.collection for e in events if e.model_form in ("full", "identified")}
    }
    return events, to_next, drops


def _nearest_rank(values: list[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))]


def summarize(
    events: list[LifetimeEvent],
    thresholds: dict[str, int],
    to_next: dict[str, int],
    drops: list[DropPrediction] | None = None,
    default_threshold: int = DEFAULT_SYNC_THRESHOLD,
) -> LifetimeSummary:
    cleared = [e.records_to_clear for e in events if e.records_to_clear is not None]
    pending = [e for e in events if e.model_form in ("full", "identified")]
    compared = [e for e in events if e.observed_form in ("full", "identified", "orphan")
                and e.model_form in ("full", "identified", "orphan")]
    used = {e.collection for e in events}
    drops = drops or []
    drop_compared = [d for d in drops if d.observed_form is not None]
    drop_match = {"identified": "identified", "not_persisted": "none"}
    return LifetimeSummary(
        sync_thresholds=json.dumps(
            {c: thresholds.get(c, default_threshold) for c in sorted(used)}, sort_keys=True
        ),
        delete_events=len(events),
        cleared_events=len(cleared),
        records_to_clear_median=_nearest_rank(cleared, 0.5),
        records_to_clear_p90=_nearest_rank(cleared, 0.9),
        records_to_clear_max=max(cleared, default=0),
        pending_events=len(pending),
        pending_records_waited_max=max((e.records_waited or 0 for e in pending), default=0),
        records_to_next_persist=json.dumps(to_next, sort_keys=True),
        dropped_events=sum(1 for e in events if e.model_form == "dropped"),
        model_compared=len(compared),
        model_agreement=sum(1 for e in compared if e.model_form == e.observed_form),
        drop_items=len(drops),
        drop_predicted_identified=sum(1 for d in drops if d.model_form == "identified"),
        drop_model_compared=len(drop_compared),
        drop_model_agreement=sum(
            1 for d in drop_compared if drop_match[d.model_form] == d.observed_form
        ),
    )


def print_summary(s: LifetimeSummary) -> None:
    print("  --- Residue lifetime (log records; x write rate = time) ---")
    print(f"    sync_threshold         : {s.sync_thresholds}")
    print(
        f"    cleared before snapshot: {s.cleared_events} delete events, records to clear "
        f"median={s.records_to_clear_median} p90={s.records_to_clear_p90} max={s.records_to_clear_max}"
    )
    print(
        f"    still Full/Identified  : {s.pending_events} events, waited up to "
        f"{s.pending_records_waited_max} records; records to next persist {s.records_to_next_persist}"
    )
    if s.dropped_events:
        print(f"    in dropped collections : {s.dropped_events} events (no persist reached them)")
    print(
        f"    model vs observed form : {s.model_agreement}/{s.model_compared} agree"
    )
    if s.drop_items:
        print(
            f"    dropped items          : {s.drop_items}, predicted to survive as Identified "
            f"{s.drop_predicted_identified}; model vs observed {s.drop_model_agreement}/"
            f"{s.drop_model_compared} agree"
        )


_SQL_TYPES = {"int": "INTEGER", "int | None": "INTEGER", "str": "TEXT", "str | None": "TEXT"}


def save(
    events: list[LifetimeEvent],
    summary: LifetimeSummary,
    output_path: Path,
    drops: list[DropPrediction] | None = None,
) -> None:
    """Write ``residue_lifetime`` (per delete event), ``residue_drop_prediction``
    (per dropped item) and ``residue_lifetime_summary``."""
    with sqlite3.connect(output_path) as conn:
        for table, cls, rows in (
            ("residue_lifetime", LifetimeEvent, [asdict(e) for e in events]),
            ("residue_drop_prediction", DropPrediction, [asdict(d) for d in drops or []]),
            ("residue_lifetime_summary", LifetimeSummary, [asdict(summary)]),
        ):
            names = [f.name for f in fields(cls)]
            columns = ", ".join(f"{f.name} {_SQL_TYPES[str(f.type)]}" for f in fields(cls))
            conn.execute(f"DROP TABLE IF EXISTS {table}")
            conn.execute(f"CREATE TABLE {table} ({columns})")
            conn.executemany(
                f"INSERT INTO {table} ({', '.join(names)}) VALUES ({', '.join(':' + n for n in names)})",
                rows,
            )
