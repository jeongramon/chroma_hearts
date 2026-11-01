"""
Main scaffold for persisted parser pipeline.
"""

from argparse import Namespace
import json
from pathlib import Path
import shutil

from chroma_hearts.config import get_parser_defaults as get_default_args
from chroma_hearts.config import parse_parser_args as parse_args
from .hnsw_extractor import extract_hnsw_from_copy
from .persistence_status import report_persistence_status
from .sdk_extractor import extract_via_sdk
from .source_copy import working_copy
from .wal_merger import merge_wal_from_copy
from .sqlite_extractor import extract_sqlite_carved


def main(args: Namespace | None = None) -> None:
    """Pipeline entry point."""
    if args is None:
        args = parse_args()
    source_chroma_db_path = args.chroma_db_path.resolve()
    output_dir = args.output_dir.resolve()

    print(f"[INFO] Chroma DB run directory: {args.chroma_run_dir}")
    print(f"[INFO] Report base directory: {args.report_base}")
    print(f"[INFO] Run label: {args.run_label}")
    print(f"[INFO] Chroma DB path: {source_chroma_db_path}")
    print(f"[INFO] Report output directory: {output_dir}")

    if output_dir.exists():
        if output_dir.is_dir():
            print(f"[INFO] Existing output directory found. Removing: {output_dir}")
            shutil.rmtree(output_dir)
        else:
            print(f"[INFO] Existing output file found. Removing: {output_dir}")
            output_dir.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Keep a snapshot for WAL/HNSW recovery. SDK extraction makes its own copy
    # of this snapshot so SDK writes cannot alter evidence used by later stages.
    with working_copy(source_chroma_db_path, prefix="chroma_hearts_run") as copy_db:
        print(f"[INFO] Working on a throwaway copy: {copy_db}")

        step1_result = extract_via_sdk(copy_db, output_dir=output_dir)
        print(f"[INFO] Step 1 saved embeddings rows: {step1_result}")

        carving = extract_sqlite_carved(copy_db, output_dir)
        print("[INFO] SQLite intact-record carving:", carving)

        step2_result = merge_wal_from_copy(copy_db, output_dir)
        print("[INFO] Step 2 result:", step2_result)

        step3_result = extract_hnsw_from_copy(copy_db, output_dir)
        print("[INFO] Step 3 result:", step3_result)
        for segment_dir in step3_result["unreferenced_segment_dirs"]:
            print(
                f"[INFO] Segment directory {segment_dir} is not referenced by any "
                "collection (e.g. left behind by a collection drop)"
            )

        step4_result = report_persistence_status(copy_db, output_dir)
        print(
            "[INFO] Step 4 result: persistence status of "
            f"{step4_result['collections']} collection(s) -> {step4_result['target_sqlite']}"
        )
        add_only = 0
        for row in step4_result["rows"]:
            _, name, _, threshold, vector_max, _, _, pending, by_op, to_next, oldest, _ = row
            if not pending:
                continue
            if set(json.loads(by_op)) <= {"add"}:
                add_only += 1  # only live data waiting; no deletion residue to clear
                continue
            print(
                f"[INFO]   {name}: {pending} record(s) {by_op} wait beyond persisted seq "
                f"{vector_max} since {oldest}; {to_next} more record(s) until the next "
                f"persist (sync_threshold={threshold})"
            )
        if add_only:
            print(f"[INFO]   {add_only} more collection(s) have only pending adds (see persistence.sqlite3)")


if __name__ == "__main__":
    main()
