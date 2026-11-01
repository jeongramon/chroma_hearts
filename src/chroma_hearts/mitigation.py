"""
Operator-side mitigations for deletion residue in a persisted Chroma store.

These procedures MODIFY the store. They are meant for operators, and for the
mitigation experiments run on copies. Never run them on evidence: acquire and
parse an untouched copy first. Each procedure needs exclusive access, with no
other client writing to the store.

* :func:`flush_pending` persists a collection's pending log tail through the
  public configuration API. This clears Full Entry and Identified Vector
  residue (they become Orphan Vectors). The collection's ``sync_threshold`` is
  lowered to the pending record count, one read runs the backfill (which
  persists once the threshold is reached and then prunes the log), and the
  original threshold is restored. No data is written. The backfill only
  re-applies the tail when the index is loaded from disk, so this runs in a
  fresh process and needs every other process that has the store open (the
  application) to be stopped. A process that already holds the index in memory
  has applied the tail and does not persist it again.
* :func:`rebuild_collection` copies the live items into a fresh collection,
  verifies the copy, drops the old collection and takes over its name. This
  leaves Orphan Vectors behind only in the old segment directory. Ids,
  documents and metadata must match exactly. Vectors must match within float32
  rounding: in cosine space the index re-normalizes each inserted vector, which
  can change the last bit even of an already normalized vector.
* :func:`remove_unreferenced_segments` deletes vector segment directories that no
  collection references, such as those a collection drop leaves on disk.
* :func:`vacuum_sqlite` runs SQLite ``VACUUM`` so freed pages no longer hold
  deleted rows.

The flush relies on behaviour of the Rust execution path (``sync_threshold`` is
read when the index is loaded, and the read-time backfill applies the persist
rule). It is verified on chromadb 1.4.0 and 1.5.9.
"""

from __future__ import annotations

import shutil
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from chroma_hearts.generator import embedding as _embedding  # noqa: F401  (registers the hash EF)
from chroma_hearts.parser.persistence_status import collection_sync_thresholds

_GET_BATCH = 1000


def _client(store: Path) -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=str(store), settings=Settings(anonymized_telemetry=False))


def _sqlite_ro(store: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{Path(store) / 'chroma.sqlite3'}?mode=ro", uri=True)


def _collection_state(store: Path, name: str) -> tuple[str, int, int]:
    """(collection id, sync_threshold, records pending beyond the persisted vector seq)."""
    with _sqlite_ro(store) as conn:
        thresholds = collection_sync_thresholds(conn)
        collection_id = next(cid for cid, (n, _) in thresholds.items() if n == name)
        vector_segment = conn.execute(
            "SELECT id FROM segments WHERE collection = ? AND scope = 'VECTOR'", (collection_id,)
        ).fetchone()[0]
        row = conn.execute("SELECT seq_id FROM max_seq_id WHERE segment_id = ?", (vector_segment,)).fetchone()
        persisted = int(row[0]) if row and row[0] is not None else 0
        pending = conn.execute(
            "SELECT COUNT(*) FROM embeddings_queue WHERE topic LIKE ? AND seq_id > ?",
            (f"%/{collection_id}", persisted),
        ).fetchone()[0]
    return collection_id, thresholds[collection_id][1], pending


@dataclass
class FlushResult:
    collection: str
    sync_threshold: int
    pending_before: int
    pending_after: int
    seconds: float


def _flush_in_this_process(store: str, collection_name: str, low: int, original: int) -> None:
    collection = _client(Path(store)).get_collection(collection_name)
    collection.modify(configuration={"hnsw": {"sync_threshold": low}})
    try:
        collection.count()  # first read: backfill loads the index, persists the tail, prunes
    finally:
        collection.modify(configuration={"hnsw": {"sync_threshold": original}})


def flush_pending(store: Path, collection_name: str) -> FlushResult:
    """Persist the pending log tail of one collection without writing any data.

    Runs in a freshly spawned process so the index is loaded from disk. No other
    process may have the store open.
    """
    import multiprocessing

    store = Path(store)
    _, threshold, pending = _collection_state(store, collection_name)
    started = time.perf_counter()
    if pending:
        worker = multiprocessing.get_context("spawn").Process(
            target=_flush_in_this_process,
            args=(str(store), collection_name, max(pending, 1), threshold),
        )
        worker.start()
        worker.join()
        if worker.exitcode != 0:
            raise RuntimeError(f"flush of '{collection_name}' failed (exit code {worker.exitcode})")
    _, _, pending_after = _collection_state(store, collection_name)
    return FlushResult(collection_name, threshold, pending, pending_after, time.perf_counter() - started)


def live_snapshot(collection: Any) -> dict[str, tuple[bytes, str | None, Any]]:
    """id -> (float32 vector bytes, document, metadata) for every live item."""
    import numpy as np

    snapshot: dict[str, tuple[bytes, str | None, Any]] = {}
    total = collection.count()
    for offset in range(0, total, _GET_BATCH):
        batch = collection.get(
            include=["embeddings", "documents", "metadatas"], limit=_GET_BATCH, offset=offset
        )
        for item_id, vector, document, metadata in zip(
            batch["ids"], batch["embeddings"], batch["documents"], batch["metadatas"]
        ):
            snapshot[item_id] = (np.asarray(vector, dtype=np.float32).tobytes(), document, metadata)
    return snapshot


@dataclass
class RebuildResult:
    collection: str
    live_items: int
    bytes_rewritten: int
    store_bytes_before: int
    store_bytes_peak: int
    verified: bool
    seconds: float
    vectors_bitwise_identical: int = 0
    vectors_max_abs_diff: float = 0.0


VECTOR_TOLERANCE = 1e-6


def compare_snapshots(source: dict, copy: dict) -> tuple[bool, int, float]:
    """(matches, vectors identical bit for bit, largest absolute vector difference).

    Ids, documents and metadata must be equal; vectors within ``VECTOR_TOLERANCE``.
    """
    import numpy as np

    if source.keys() != copy.keys():
        return False, 0, float("inf")
    identical, max_diff, ok = 0, 0.0, True
    for item_id, (vector, document, metadata) in source.items():
        vector2, document2, metadata2 = copy[item_id]
        if document != document2 or metadata != metadata2:
            ok = False
        if vector == vector2:
            identical += 1
            continue
        diff = float(np.max(np.abs(np.frombuffer(vector, np.float32) - np.frombuffer(vector2, np.float32))))
        max_diff = max(max_diff, diff)
        ok = ok and diff <= VECTOR_TOLERANCE
    return ok, identical, max_diff


def _dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def rebuild_collection(store: Path, collection_name: str) -> RebuildResult:
    """Copy live items into a fresh collection, verify, drop the old one, take its name.

    The old collection is dropped only after the copy is verified, so a failure
    before that point leaves the original untouched. The old segment directory
    stays on disk until :func:`remove_unreferenced_segments` runs.
    """
    store = Path(store)
    started = time.perf_counter()
    size_before = _dir_bytes(store)
    client = _client(store)
    old = client.get_collection(collection_name)
    source = live_snapshot(old)
    _, threshold, _ = _collection_state(store, collection_name)

    temp_name = f"{collection_name}_rebuild"
    try:
        new = client.create_collection(temp_name, configuration=old.configuration, metadata=old.metadata)
    except Exception:
        new = client.create_collection(
            temp_name,
            configuration={"hnsw": {"sync_threshold": threshold}},
            metadata=old.metadata,
            embedding_function=old._embedding_function,  # noqa: SLF001
        )
    items = list(source.items())
    rewritten = 0
    import numpy as np

    for start in range(0, len(items), _GET_BATCH):
        chunk = items[start:start + _GET_BATCH]
        new.add(
            ids=[i for i, _ in chunk],
            embeddings=[np.frombuffer(v, dtype=np.float32) for _, (v, _, _) in chunk],
            documents=[d for _, (_, d, _) in chunk],
            metadatas=[m for _, (_, _, m) in chunk],
        )
        rewritten += sum(len(v) + len((d or "").encode()) + len(str(m).encode()) for _, (v, d, m) in chunk)
    size_peak = _dir_bytes(store)

    verified, bitwise, max_diff = compare_snapshots(source, live_snapshot(new))
    if not verified:
        client.delete_collection(temp_name)
        raise RuntimeError(f"Rebuilt copy of '{collection_name}' does not match; original kept.")
    client.delete_collection(collection_name)
    new.modify(name=collection_name)
    return RebuildResult(
        collection_name, len(items), rewritten, size_before, size_peak, verified,
        time.perf_counter() - started, bitwise, max_diff,
    )


class DiskPeak:
    """Sample a directory's size while a procedure runs; ``extra`` is the peak above the start.

    Use as a context manager around the procedure. Sampling runs in a background
    thread every ``interval`` seconds.
    """

    def __init__(self, path: Path, interval: float = 0.02) -> None:
        import threading

        self.path, self.interval = Path(path), interval
        self.start = self.peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.peak = max(self.peak, _dir_bytes(self.path))
            except FileNotFoundError:
                pass  # a file vanished mid-walk; sample again
            self._stop.wait(self.interval)

    def __enter__(self) -> "DiskPeak":
        self.start = self.peak = _dir_bytes(self.path)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, _dir_bytes(self.path))

    @property
    def extra(self) -> int:
        return self.peak - self.start


def remove_unreferenced_segments(store: Path) -> list[str]:
    """Delete vector segment directories that no collection references.

    Run with no client open on the store.
    """
    store = Path(store)
    with _sqlite_ro(store) as conn:
        referenced = {str(r[0]) for r in conn.execute("SELECT id FROM segments")}
    removed = []
    for directory in store.iterdir():
        if directory.is_dir() and (directory / "header.bin").exists() and directory.name not in referenced:
            shutil.rmtree(directory)
            removed.append(directory.name)
    return removed


def vacuum_sqlite(store: Path) -> int:
    """Run SQLite VACUUM on chroma.sqlite3; return bytes reclaimed. No client may be open."""
    path = Path(store) / "chroma.sqlite3"
    before = path.stat().st_size
    conn = sqlite3.connect(path)
    try:
        conn.execute("VACUUM")
    finally:
        conn.close()
    return before - path.stat().st_size
