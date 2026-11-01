"""
HNSW extractor (pipeline step 3):
Parse HNSW binary index files and merge vectors into embeddings.sqlite3.
"""

from dataclasses import dataclass
import pickle
import json
import math
import sqlite3
import struct
from pathlib import Path
from typing import Any, Dict

from .source_copy import working_copy
from .recovery_store import ensure_schema, add_evidence


@dataclass
class HnswHeader:
    """Parsed values from header.bin."""

    persistence_version: int
    offset_level0: int
    max_elements: int
    cur_element_count: int
    size_data_per_element: int
    label_offset: int
    offset_data: int
    maxlevel: int
    enterpoint_node: int
    max_m: int
    max_m0: int
    mult: float
    ef_construction: int

    @classmethod
    def from_bytes(cls, raw: bytes) -> "HnswHeader":
        """Parse fixed offsets from header.bin."""
        return cls(
            persistence_version=struct.unpack_from("<I", raw, 0x0)[0],
            offset_level0=struct.unpack_from("<Q", raw, 0x4)[0],
            max_elements=struct.unpack_from("<Q", raw, 0x0C)[0],
            cur_element_count=struct.unpack_from("<Q", raw, 0x14)[0],
            size_data_per_element=struct.unpack_from("<Q", raw, 0x1C)[0],
            label_offset=struct.unpack_from("<Q", raw, 0x24)[0],
            offset_data=struct.unpack_from("<Q", raw, 0x2C)[0],
            maxlevel=struct.unpack_from("<I", raw, 0x34)[0],
            enterpoint_node=struct.unpack_from("<I", raw, 0x38)[0],
            max_m=struct.unpack_from("<Q", raw, 0x3C)[0],
            max_m0=struct.unpack_from("<Q", raw, 0x44)[0],
            mult=struct.unpack_from("<d", raw, 0x4C)[0],
            ef_construction=struct.unpack_from("<Q", raw, 0x54)[0],
        )


@dataclass
class Level0Element:
    """Parsed level0 element block from data_level0.bin."""

    hnsw_label: int
    vector_blob: bytes
    level0_neighbor_count: int
    delete_mark: int
    neighbor_ids: list[int]


def _read_header(header_path: Path) -> HnswHeader:
    """Load and parse one header.bin."""
    return HnswHeader.from_bytes(header_path.read_bytes())


def _parse_level0_elements(data_path: Path, header: HnswHeader) -> list[Level0Element]:
    """Parse all active blocks from data_level0.bin using header offsets."""
    raw = data_path.read_bytes()
    elements: list[Level0Element] = []

    neighbor_slot_size = int(4 * header.max_m0)
    vector_size = int(header.label_offset - header.offset_data)
    label_size = 8

    for idx in range(int(header.cur_element_count)):
        block_start = int(header.offset_level0 + idx * header.size_data_per_element)
        block_end = block_start + int(header.size_data_per_element)
        if block_end > len(raw):
            break

        level0_neighbor_count = struct.unpack_from("<H", raw, block_start + 0x0)[0]
        delete_mark = struct.unpack_from("<B", raw, block_start + 0x2)[0]

        neighbors_start = block_start + 0x4
        neighbor_ids = list(
            struct.unpack_from(f"<{int(header.max_m0)}i", raw, neighbors_start)
        )

        vector_start = block_start + int(header.offset_data)
        vector_end = vector_start + vector_size
        vector_blob = raw[vector_start:vector_end]

        label_start = block_start + int(header.label_offset)
        hnsw_label = struct.unpack_from("<Q", raw, label_start)[0]

        elements.append(
            Level0Element(
                hnsw_label=hnsw_label,
                vector_blob=vector_blob,
                level0_neighbor_count=level0_neighbor_count,
                delete_mark=delete_mark,
                neighbor_ids=neighbor_ids[: min(level0_neighbor_count, len(neighbor_ids))],
            )
        )

    return elements


def _load_pickle_label_to_id(pickle_path: Path) -> dict[int, str]:
    """Load label->id mapping from index_metadata.pickle.

    Missing pickle means the label->id mapping is unrecoverable for this
    segment, not that the segment has no data: ``header.bin``/``data_level0.bin``
    are parsed regardless, and every label then falls back to an orphan
    (``id=NULL``) row via ``_upsert_hnsw_embedding``.
    """
    if not pickle_path.exists():
        return {}
    meta = pickle.loads(pickle_path.read_bytes())
    return {int(k): str(v) for k, v in meta.get("label_to_id", {}).items()}


def _load_segment_collection_maps(
    sysdb_sqlite_path: Path,
) -> tuple[dict[str, str], dict[str, str]]:
    """Load vector segment -> collection and collection -> name mappings."""
    if not sysdb_sqlite_path.exists():
        raise FileNotFoundError(f"sysdb sqlite not found: {sysdb_sqlite_path}")

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


def load_segment_spaces(sysdb_path: Path) -> dict[str, str]:
    """Read declared metrics; do not infer normalization from length.bin values."""
    with sqlite3.connect(f'file:{sysdb_path}?mode=ro', uri=True) as conn:
        conn.row_factory = sqlite3.Row
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        by_collection = {}
        for row in conn.execute('SELECT * FROM collections'):
            row = dict(row)
            config = json.loads(row.get('config_json_str') or '{}')
            hnsw = config.get('hnsw_configuration', config.get('hnsw', {})) or {}
            space = hnsw.get('space')
            schema = json.loads(row.get('schema_str') or '{}')
            for definition in [schema.get('defaults', {}), schema.get('keys', {}).get('#embedding', {})]:
                space = definition.get('float_list', {}).get('vector_index', {}).get('config', {}).get('space', space)
            if space is not None:
                by_collection[row['id']] = space
        result = {r['id']: by_collection[r['collection']] for r in conn.execute('SELECT id,collection FROM segments')
                  if r['collection'] in by_collection}
        if 'segment_metadata' in tables:
            for row in conn.execute("SELECT segment_id,str_value FROM segment_metadata WHERE key='hnsw:space'"):
                if row['str_value']:
                    result[row['segment_id']] = row['str_value']
        return result


def restore_cosine_vector(raw: bytes, norm: float) -> bytes:
    """Undo stored normalization with float32 multiplication, as get_items does."""
    if not math.isfinite(norm) or norm < 0 or len(raw) % 4:
        raise ValueError('Invalid HNSW original norm or vector length')
    values = struct.unpack(f'<{len(raw)//4}f', raw)
    return struct.pack(f'<{len(values)}f', *(v * norm for v in values))


def _upsert_hnsw_embedding(
    dst_conn: sqlite3.Connection,
    item_id: str | None,
    hnsw_label: int,
    vector_segment_id: str,
    collection_id: str | None,
    collection_name: str | None,
    vector_blob: bytes,
    delete_mark: int,
    vector_details: dict | None = None,
) -> bool:
    """
    Merge one HNSW element into the embeddings table without destroying data.

    The HNSW binary holds the vector as of the last flush, which may be an older
    version than one already recovered from the WAL. So an existing row is only
    reused when it does not already carry a *different* vector for this id:

    - a row whose ``vector`` is NULL (an id/label mapping placeholder) is filled
      in place;
    - a row whose ``vector`` already equals this HNSW blob is the same version,
      so its HNSW fields (label, segment, delete_mark) are back-filled in place;
    - otherwise the id already has a different vector version recovered — insert a
      new row for this HNSW version instead of overwriting the other one.

    Returns True if an existing row was updated, False if a new row was inserted.
    """
    ensure_schema(dst_conn)
    target_rowid: int | None = None
    if item_id is not None:
        candidates = dst_conn.execute(
            """
            SELECT rowid, vector
            FROM embeddings
            WHERE id = ?
                    AND source_sqlite_carved = 0
            AND (collection_id = ? OR collection_id IS NULL)
            ORDER BY
                collection_id IS NULL ASC,
                source_chroma_sdk DESC,
                source_wal DESC
            """,
            (item_id, collection_id),
        ).fetchall()
        null_rowid: int | None = None
        for rowid, existing_vector in candidates:
            if existing_vector is not None and bytes(existing_vector) == vector_blob:
                target_rowid = rowid  # same version already present
                break
            if existing_vector is None and null_rowid is None:
                null_rowid = rowid  # placeholder we can fill
        if target_rowid is None:
            target_rowid = null_rowid

    if target_rowid is not None:
        dst_conn.execute(
            """
            UPDATE embeddings
            SET
                vector = ?,
                hnsw_label = ?,
                vector_segment_id = COALESCE(vector_segment_id, ?),
                collection_id = COALESCE(collection_id, ?),
                collection_name = COALESCE(collection_name, ?),
                source_hnsw = 1,
                delete_mark = ?
            WHERE rowid = ?
            """,
            (
                sqlite3.Binary(vector_blob),
                hnsw_label,
                vector_segment_id,
                collection_id,
                collection_name,
                bool(delete_mark & 0x01),
                target_rowid,
            ),
        )
        add_evidence(dst_conn, target_rowid, 'hnsw',
                     source_file=f'{vector_segment_id}/data_level0.bin', item_id=item_id,
                     collection_id=collection_id, vector=vector_blob,
                     details={'hnsw_label': hnsw_label, 'delete_mark': bool(delete_mark & 1),
                              'vector_segment_id': vector_segment_id, **(vector_details or {})})
        return True

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
            None,
            None,
            "Add",
            None,
            item_id,
            None,
            hnsw_label,
            None,
            vector_segment_id,
            collection_id,
            collection_name,
            sqlite3.Binary(vector_blob),
            None,
            None,
            False,
            False,
            True,
            bool(delete_mark & 0x01),
        ),
    )
    rowid = dst_conn.execute('SELECT last_insert_rowid()').fetchone()[0]
    add_evidence(dst_conn, rowid, 'hnsw', source_file=f'{vector_segment_id}/data_level0.bin',
                 item_id=item_id, collection_id=collection_id, vector=vector_blob,
                 details={'hnsw_label': hnsw_label, 'delete_mark': bool(delete_mark & 1),
                          'vector_segment_id': vector_segment_id, **(vector_details or {})})
    return False


def extract_hnsw(source_chroma_db_path: Path, output_dir: Path) -> Dict[str, Any]:
    """Parse HNSW binaries into the report, working on a throwaway copy.

    The binaries are only read, but the source is still copied so that no stage
    of the pipeline ever touches the original evidence directory. Callers that
    already hold a working copy should call :func:`extract_hnsw_from_copy`
    instead to avoid copying twice.
    """
    with working_copy(source_chroma_db_path, prefix="chroma_hearts_hnsw") as copy_db:
        return extract_hnsw_from_copy(copy_db, output_dir)


def extract_hnsw_from_copy(
    source_chroma_db_path: Path, output_dir: Path,
    segment_spaces: dict[str, str] | None = None,
) -> Dict[str, Any]:
    """Parse HNSW files from UUID folders and merge vectors into embeddings.sqlite3.

    ``source_chroma_db_path`` must already be a working copy. Use
    :func:`extract_hnsw` to copy first.
    """
    source_root = source_chroma_db_path.resolve()
    sysdb_sqlite = output_dir.resolve() / "sysdb.sqlite3"
    target_sqlite = output_dir.resolve() / "embeddings.sqlite3"
    if not target_sqlite.exists():
        raise FileNotFoundError(f"Target embeddings sqlite not found: {target_sqlite}")

    segment_to_collection, collection_to_name = _load_segment_collection_maps(
        sysdb_sqlite
    )

    spaces = dict(segment_spaces or {})
    spaces.update(load_segment_spaces(sysdb_sqlite))

    uuid_dirs = sorted(
        p
        for p in source_root.iterdir()
        if p.is_dir()
        and (p / "header.bin").exists()
        and (p / "data_level0.bin").exists()
    )

    with sqlite3.connect(str(target_sqlite)) as dst_conn:
        parsed_elements = 0
        mapped_elements = 0
        updated_rows = 0
        inserted_rows = 0
        missing_label_count = 0
        # Segment directories no collection references any more, e.g. the ones a
        # collection drop leaves on disk. Their rows keep vector_segment_id and
        # carry collection_id NULL.
        unreferenced_segment_dirs: list[str] = []

        for uuid_dir in uuid_dirs:
            vector_segment_id = uuid_dir.name
            collection_id = segment_to_collection.get(vector_segment_id)
            if collection_id is None:
                unreferenced_segment_dirs.append(vector_segment_id)
            collection_name = (
                collection_to_name.get(collection_id)
                if collection_id is not None
                else None
            )
            label_to_id = _load_pickle_label_to_id(uuid_dir / "index_metadata.pickle")
            header = _read_header(uuid_dir / "header.bin")
            elements = _parse_level0_elements(uuid_dir / "data_level0.bin", header)
            parsed_elements += len(elements)

            lengths = None
            if spaces.get(vector_segment_id) == 'cosine':
                length_path = uuid_dir / 'length.bin'
                lengths = length_path.read_bytes()
                if len(lengths) < 4 * len(elements):
                    raise ValueError(f'Truncated cosine length.bin: {length_path}')
            for element_index, element in enumerate(elements):
                vector_blob = element.vector_blob
                vector_details = None
                if lengths is not None:
                    norm = struct.unpack_from('<f', lengths, 4 * element_index)[0]
                    vector_blob = restore_cosine_vector(vector_blob, norm)
                    vector_details = {'space': 'cosine', 'original_norm': norm,
                                      'element_index': element_index,
                                      'raw_vector_hex': element.vector_blob.hex(),
                                      'norm_source_file': f'{vector_segment_id}/length.bin',
                                      'norm_byte_offset': 4 * element_index}
                item_id = label_to_id.get(int(element.hnsw_label))
                if _upsert_hnsw_embedding(
                    dst_conn,
                    item_id,
                    int(element.hnsw_label),
                    vector_segment_id,
                    collection_id,
                    collection_name,
                    vector_blob,
                    element.delete_mark,
                    vector_details,
                ):
                    updated_rows += 1
                else:
                    inserted_rows += 1
                if item_id is None:
                    missing_label_count += 1
                else:
                    mapped_elements += 1

        dst_conn.commit()

    return {
        "step": "hnsw_extract",
        "source_chroma_db_path": str(source_root),
        "target_embeddings_sqlite": str(target_sqlite),
        "uuid_folders": len(uuid_dirs),
        "parsed_elements": parsed_elements,
        "mapped_elements": mapped_elements,
        "missing_label_count": missing_label_count,
        "unreferenced_segment_dirs": unreferenced_segment_dirs,
        "updated_rows": updated_rows,
        "inserted_rows": inserted_rows,
        "status": "ok",
    }
