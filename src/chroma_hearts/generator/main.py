"""
Generate and operate a local Chroma DB from documents.
"""

from argparse import Namespace
from pathlib import Path
import random
import shutil

from chromadb.api.models.Collection import Collection

from chroma_hearts.config import get_generator_defaults as get_default_args
from chroma_hearts.config import get_embedding_settings
from chroma_hearts.config import parse_generator_args as parse_args
from chroma_hearts.config import resolve_generator_run_dir
from .chroma_operations import (
    COLLECTION_NAME,
    add_documents,
    create_collection,
    create_random_collection_with_client,
    delete_random_document,
    get_client,
    set_sync_threshold,
    search_documents,
    update_random_document,
    upsert_random_document,
)
from chroma_hearts.parser.persistence_status import DEFAULT_SYNC_THRESHOLD
from .operation_log import OperationLog

LOG_EVERY = 100

OPERATION_POPULATION = ["add", "search", "delete", "update", "upsert", "new_collection"]


def build_operation_weights(
    add_weight: int,
    search_weight: int,
    delete_weight: int,
    update_weight: int,
    upsert_weight: int,
    new_collection_weight: int,
) -> list[int]:
    """Build the weighted operation pool."""
    return [
        max(0, add_weight),
        max(0, search_weight),
        max(0, delete_weight),
        max(0, update_weight),
        max(0, upsert_weight),
        max(0, new_collection_weight),
    ]


def run_random_operation(
    collection: Collection,
    input_dir: Path,
    client,
    operation_weights: list[int],
) -> tuple[str, int | str, str | None, str | None]:
    """Run one random operation from the operation pool.

    Returns ``(operation, result, doc_id, document)`` where ``result`` keeps the
    original success semantics (count / new collection name) and ``doc_id`` /
    ``document`` capture the affected id and embedded text for recovery
    verification (both ``None`` for search / new_collection).
    """
    operation = random.choices(
        population=OPERATION_POPULATION,
        weights=operation_weights,
        k=1,
    )[0]

    if operation == "add":
        added, doc_id, document = add_documents(collection, input_dir)
        return operation, added, doc_id, document
    elif operation == "search":
        executed = search_documents(collection)
        return operation, executed, None, None
    elif operation == "delete":
        deleted, doc_id, document = delete_random_document(collection)
        return operation, deleted, doc_id, document
    elif operation == "update":
        updated, doc_id, document = update_random_document(collection)
        return operation, updated, doc_id, document
    elif operation == "upsert":
        upserted, doc_id, document = upsert_random_document(collection)
        return operation, upserted, doc_id, document
    else:
        new_collection = create_random_collection_with_client(client)
        return operation, new_collection.name, None, None


def drop_primary_collection(
    client,
    collection_map: dict[str, Collection],
    iteration: int,
    operation_log: OperationLog,
) -> list[tuple[str, int | str]]:
    """Drop the primary collection and keep the workload running.

    The drop is logged as ``drop_collection`` with the number of items the
    collection held, so the evaluator can treat every one of them as deleted at
    this iteration. If no collection is left, a fresh one is created (and logged
    as ``new_collection``) for the remaining iterations.

    Returns ``(operation, result)`` pairs for the operation summary.
    """
    dropped = collection_map.pop(COLLECTION_NAME)
    dropped_count = dropped.count()
    client.delete_collection(COLLECTION_NAME)
    operation_log.record(iteration, COLLECTION_NAME, "drop_collection", dropped_count, 0)
    performed: list[tuple[str, int | str]] = [("drop_collection", dropped_count)]
    print(f"[INFO] {iteration}: dropped collection '{COLLECTION_NAME}' ({dropped_count} items)")

    if not collection_map:
        replacement = create_random_collection_with_client(client)
        collection_map[replacement.name] = replacement
        operation_log.record(iteration, replacement.name, "new_collection", replacement.name, 0)
        performed.append(("new_collection", replacement.name))
        print(f"[INFO] {iteration}: continuing in new collection '{replacement.name}'")
    return performed


def main(args: Namespace | None = None) -> None:
    """Entry point for generating and operating Chroma DB."""
    if args is None:
        args = parse_args()
    input_dir = args.input_dir.resolve()
    base_output_dir = args.output_dir.resolve()
    iterations = max(1, args.iterations)
    add_weight = max(0, args.add_weight)
    search_weight = max(0, args.search_weight)
    delete_weight = max(0, args.delete_weight)
    update_weight = max(0, args.update_weight)
    upsert_weight = max(0, args.upsert_weight)
    new_collection_weight = max(0, args.new_collection_weight)
    drop_at = max(0, getattr(args, "drop_at", 0))
    seed = getattr(args, "seed", -1)
    sync_threshold = max(0, getattr(args, "sync_threshold", 0))
    if seed >= 0:
        random.seed(seed)
    set_sync_threshold(sync_threshold)
    run_output_dir = resolve_generator_run_dir(
        base_output_dir,
        add_weight,
        search_weight,
        delete_weight,
        update_weight,
        upsert_weight,
        new_collection_weight,
        drop_at,
        sync_threshold,
    )
    operation_weights = build_operation_weights(
        add_weight,
        search_weight,
        delete_weight,
        update_weight,
        upsert_weight,
        new_collection_weight,
    )

    print(f"[INFO] Input directory: {input_dir}")
    print(f"[INFO] Output base directory: {base_output_dir}")
    print(f"[INFO] Run output directory: {run_output_dir}")
    print(f"[INFO] Iterations: {iterations}")
    embedding_settings = get_embedding_settings()
    print(
        "[INFO] Embedding: "
        f"mode={embedding_settings.mode}, dimension={embedding_settings.dimension}"
    )
    print(
        "[INFO] Operation weights: "
        f"add={add_weight}, search={search_weight}, delete={delete_weight}, "
        f"update={update_weight}, upsert={upsert_weight}, new_collection={new_collection_weight}"
    )
    print(
        f"[INFO] Seed: {seed if seed >= 0 else 'unseeded'}, "
        f"sync_threshold: {sync_threshold or 'Chroma default'}"
    )
    if drop_at:
        print(f"[INFO] Collection drop: primary collection at iteration {drop_at}")
        if drop_at > iterations:
            print(f"[WARN] drop_at={drop_at} exceeds iterations={iterations}; no drop will happen.")

    if run_output_dir.is_file():
        print(f"[INFO] Existing output file found. Removing: {run_output_dir}")
        run_output_dir.unlink()
    elif run_output_dir.exists():
        chroma_dir = run_output_dir / "chroma_db"
        operations_db_path = run_output_dir / "operations.sqlite3"
        if chroma_dir.exists():
            print(f"[INFO] Existing chroma_db found. Removing: {chroma_dir}")
            shutil.rmtree(chroma_dir)
        if operations_db_path.exists():
            print(f"[INFO] Existing operations log found. Removing: {operations_db_path}")
            operations_db_path.unlink()
    run_output_dir.mkdir(parents=True, exist_ok=True)

    client = get_client(run_output_dir)
    collection = create_collection(run_output_dir)
    collection_map: dict[str, Collection] = {collection.name: collection}
    print("[INFO] Collection created.")

    operations_db_path = run_output_dir / "operations.sqlite3"
    operation_summary: dict[str, dict[str, int]] = {
        operation: {"attempts": 0, "successes": 0}
        for operation in OPERATION_POPULATION
    }
    with OperationLog(operations_db_path) as operation_log:
        # The evaluator's lifetime model needs the threshold even for collections
        # that a drop removes from the store.
        operation_log.record_config("sync_threshold", sync_threshold or DEFAULT_SYNC_THRESHOLD)
        for iteration in range(1, iterations + 1):
            if iteration == drop_at:
                for operation, result in drop_primary_collection(
                    client, collection_map, iteration, operation_log
                ):
                    stats = operation_summary.setdefault(
                        operation, {"attempts": 0, "successes": 0}
                    )
                    stats["attempts"] += 1
                    if str(result) != "0":
                        stats["successes"] += 1
                continue

            target_name = random.choice(list(collection_map.keys()))
            target_collection = collection_map[target_name]
            operation, result, doc_id, document = run_random_operation(
                target_collection,
                input_dir,
                client,
                operation_weights,
            )

            if operation == "new_collection":
                new_name = str(result)
                if new_name not in collection_map:
                    collection_map[new_name] = client.get_collection(new_name)
                target_name = new_name
                target_collection = collection_map[target_name]

            count = target_collection.count()
            operation_log.record(
                iteration,
                target_name,
                operation,
                result,
                count,
                doc_id=doc_id,
                document=document,
            )
            operation_summary.setdefault(operation, {"attempts": 0, "successes": 0})
            operation_summary[operation]["attempts"] += 1
            if str(result) != "0":
                operation_summary[operation]["successes"] += 1

            if iteration == 1 or iteration % LOG_EVERY == 0 or iteration == iterations:
                print(
                    f"[INFO] {iteration}/{iterations} "
                    f"Target={target_name}, Operation={operation}, "
                    f"result={result}, collection_count={count}"
                )

    final_count = sum(c.count() for c in collection_map.values())
    print(f"\n[INFO] Done. Final item count across collections: {final_count}")
    print("[INFO] Operation summary:")
    for operation, stats in sorted(operation_summary.items()):
        print(
            f"  - {operation:<14}: "
            f"{stats['successes']} success / {stats['attempts']} attempts"
        )
    print(f"[INFO] Operation log saved: {operations_db_path}")


if __name__ == "__main__":
    main()
