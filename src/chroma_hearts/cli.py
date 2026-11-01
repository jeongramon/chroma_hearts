"""
Chroma Hearts interactive entry point.
"""

from argparse import Namespace
import sys

from chroma_hearts.config import (
    build_parser_run_namespace,
    get_evaluator_defaults,
    get_generator_defaults,
    get_parser_defaults,
    get_pdf_collector_defaults,
)
from chroma_hearts.evaluator.main import main as run_evaluator
from chroma_hearts.generator.main import main as run_generator
from chroma_hearts.parser.main import main as run_parser
from chroma_hearts.pdf_collector.main import main as run_collector
from chroma_hearts.prompts import prompt_int, prompt_menu, prompt_path


def run_pdf_collector() -> None:
    """Run sample PDF collection with interactive prompts."""
    defaults = get_pdf_collector_defaults()

    print("\n[Collect Sample PDFs]")
    print("Press Enter to use the default value for each prompt.\n")

    args = Namespace(
        count=prompt_int("Number of PDFs to download", defaults.count),
        output_dir=prompt_path("Output directory", defaults.output_dir),
    )
    run_collector(args)


def run_sample_generator() -> None:
    """Run sample Chroma DB generation with interactive prompts."""
    defaults = get_generator_defaults()

    print("\n[Sample Chroma DB Generate]")
    print("Press Enter to use the default value for each prompt.\n")
    print("Operation pool: add, search, delete, update, upsert, new_collection\n")

    args = Namespace(
        input_dir=prompt_path("Input documents directory", defaults.input_dir),
        output_dir=prompt_path("Output base directory", defaults.output_dir),
        iterations=prompt_int("Iterations", defaults.iterations),
        add_weight=prompt_int("Add operation weight", defaults.add_weight),
        search_weight=prompt_int("Search operation weight", defaults.search_weight),
        delete_weight=prompt_int("Delete operation weight", defaults.delete_weight),
        update_weight=prompt_int("Update operation weight", defaults.update_weight),
        upsert_weight=prompt_int("Upsert operation weight", defaults.upsert_weight),
        new_collection_weight=prompt_int("New collection operation weight", defaults.new_collection_weight),
        drop_at=prompt_int("Drop primary collection at iteration (0 = never)", defaults.drop_at),
        seed=prompt_int("Random seed (-1 = unseeded)", defaults.seed),
        sync_threshold=prompt_int("hnsw sync_threshold (0 = Chroma default)", defaults.sync_threshold),
    )
    run_generator(args)


def run_recover_chroma_db() -> None:
    """Run persisted Chroma DB recovery with interactive prompts."""
    defaults = get_parser_defaults()

    print("\n[Recover Chroma DB]")
    print("Press Enter to use the default value for each prompt.\n")

    chroma_run_dir = prompt_path("Chroma DB run directory", defaults.chroma_run_dir)
    report_base = prompt_path("Report base directory", defaults.report_base)
    args = build_parser_run_namespace(
        chroma_run_dir=chroma_run_dir,
        report_base=report_base,
    )
    run_parser(args)


def run_evaluate_recovery() -> None:
    """Run recovery rate evaluation with interactive prompts."""
    defaults = get_evaluator_defaults()

    print("\n[Evaluate Recovery Rate]")
    print("Press Enter to use the default value for each prompt.\n")

    args = Namespace(
        chroma_run_dir=prompt_path("Chroma DB run directory", defaults.chroma_run_dir),
        report_base=prompt_path("Report base directory", defaults.report_base),
    )
    run_evaluator(args)


def main() -> None:
    """Interactive CLI entry point."""
    choice = prompt_menu()

    if choice == "4":
        print("Exiting.")
        return
    if choice == "0":
        run_pdf_collector()
        return
    if choice == "1":
        run_sample_generator()
        return
    if choice == "2":
        run_recover_chroma_db()
        return
    if choice == "3":
        run_evaluate_recovery()
        return


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted.")
        sys.exit(130)
