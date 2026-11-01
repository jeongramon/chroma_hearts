"""Throwaway working copies of a persisted Chroma directory.

Every stage of the recovery pipeline reads evidence — ``chroma.sqlite3`` and the
HNSW segment binaries — that must not be altered by the act of reading it.
Opening a SQLite database read-write is enough to change it: SQLite may roll a
hot journal back, checkpoint and truncate a ``-wal`` file, or create ``-wal`` and
``-shm`` siblings next to the original. For this tool the write-ahead log *is* a
primary recovery source, so a checkpoint on open can destroy exactly the deleted
records being recovered.

The rule is therefore: never point SQLite or a ``PersistentClient`` at the
original directory. Copy it, work on the copy, discard the copy.
"""

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
import shutil
import tempfile
import uuid


@contextmanager
def working_copy(source_chroma_db_path: Path, prefix: str = "chroma_hearts") -> Iterator[Path]:
    """Copy a persisted Chroma directory to temp space and yield the copy.

    The copy is removed when the context exits, including on error. The source
    directory is only ever read, never opened for writing.

    Args:
        source_chroma_db_path: Persisted Chroma directory (holding
            ``chroma.sqlite3`` and the per-segment HNSW UUID directories).
        prefix: Label used in the temporary directory name, to make it obvious
            which stage left a copy behind if cleanup is ever interrupted.

    Yields:
        Path to the working copy, laid out identically to the source.
    """
    source = Path(source_chroma_db_path).resolve()
    if not source.exists():
        raise FileNotFoundError(f"Source path does not exist: {source}")
    if not source.is_dir():
        raise NotADirectoryError(f"Source path is not a directory: {source}")

    copy_root = Path(tempfile.gettempdir()) / f"{prefix}_{uuid.uuid4().hex}"
    copy_db = copy_root / "chroma_db"
    try:
        copy_root.mkdir(parents=True, exist_ok=False)
        shutil.copytree(source, copy_db, dirs_exist_ok=False)
        yield copy_db
    finally:
        if copy_root.exists():
            shutil.rmtree(copy_root, ignore_errors=True)
