"""Load project defaults from config.toml and build CLI argument parsers."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from functools import lru_cache
from pathlib import Path
import tomllib
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.toml"


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    """Load and cache config.toml from the project root."""
    with CONFIG_PATH.open("rb") as config_file:
        return tomllib.load(config_file)


def resolve_path(path_value: str | Path) -> Path:
    """Resolve a project-relative path against the project root."""
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def parse_args(description: str, defaults: Namespace, arg_specs: list[dict[str, Any]]) -> Namespace:
    """Build an ArgumentParser from shared defaults and per-command specs."""
    parser = ArgumentParser(description=description)
    for spec in arg_specs:
        flags = spec["flags"]
        dest = spec["dest"]
        parser.add_argument(
            *flags,
            dest=dest,
            type=spec.get("type", str),
            default=getattr(defaults, dest),
            help=spec["help"],
        )
    return parser.parse_args()


def get_pdf_collector_defaults() -> Namespace:
    """Return default argument values for PDF collection."""
    config = load_config()["pdf_collector"]
    return Namespace(
        count=config["count"],
        output_dir=resolve_path(config["output_dir"]),
    )


def parse_pdf_collector_args() -> Namespace:
    """Parse command-line arguments for PDF collection."""
    defaults = get_pdf_collector_defaults()
    return parse_args(
        description="Collect sample PDFs from arXiv",
        defaults=defaults,
        arg_specs=[
            {
                "flags": ("-n", "--count"),
                "dest": "count",
                "type": int,
                "help": f"Number of PDFs to download (default: {defaults.count})",
            },
            {
                "flags": ("-o", "--output-dir"),
                "dest": "output_dir",
                "type": Path,
                "help": f"Directory to save PDFs (default: {config_path('pdf_collector', 'output_dir')})",
            },
        ],
    )


def resolve_weighted_run_dir(
    base_output_dir: Path,
    add_weight: int,
    search_weight: int,
    delete_weight: int,
    update_weight: int,
    upsert_weight: int,
    new_collection_weight: int,
    drop_at: int = 0,
    sync_threshold: int = 0,
) -> Path:
    """Resolve per-run output directory under a base path using weight labels.

    A run that drops its collection gets a ``_drop{iteration}`` suffix, and a run
    with an explicit ``sync_threshold`` a ``_sync{value}`` suffix, so neither shares
    a directory with the default run of the same weights.
    """
    label = f"{add_weight}_{search_weight}_{delete_weight}_{update_weight}_{upsert_weight}_{new_collection_weight}"
    if drop_at > 0:
        label += f"_drop{drop_at}"
    if sync_threshold > 0:
        label += f"_sync{sync_threshold}"
    return base_output_dir.resolve() / label


def resolve_generator_run_dir(
    base_output_dir: Path,
    add_weight: int,
    search_weight: int,
    delete_weight: int,
    update_weight: int,
    upsert_weight: int,
    new_collection_weight: int,
    drop_at: int = 0,
    sync_threshold: int = 0,
) -> Path:
    """Resolve per-run output directory under the configured generator base path."""
    return resolve_weighted_run_dir(
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


def resolve_parser_output_dir(chroma_run_dir: Path, report_base: Path) -> Path:
    """Resolve report output directory from the Chroma run folder name."""
    return report_base.resolve() / chroma_run_dir.resolve().name


def resolve_generator_chroma_db_dir(
    base_output_dir: Path,
    add_weight: int,
    search_weight: int,
    delete_weight: int,
    update_weight: int,
    upsert_weight: int,
    new_collection_weight: int,
    drop_at: int = 0,
    sync_threshold: int = 0,
) -> Path:
    """Resolve persisted Chroma DB directory for one generator run."""
    return resolve_generator_run_dir(
        base_output_dir,
        add_weight,
        search_weight,
        delete_weight,
        update_weight,
        upsert_weight,
        new_collection_weight,
        drop_at,
        sync_threshold,
    ) / "chroma_db"


def get_generator_defaults() -> Namespace:
    """Return default argument values for sample ChromaDB generation."""
    config = load_config()["generator"]
    return Namespace(
        input_dir=resolve_path(config["input_dir"]),
        output_dir=resolve_path(config["output_dir"]),
        iterations=config["iterations"],
        add_weight=config["add_weight"],
        search_weight=config["search_weight"],
        delete_weight=config["delete_weight"],
        update_weight=config["update_weight"],
        upsert_weight=config["upsert_weight"],
        new_collection_weight=config["new_collection_weight"],
        drop_at=int(config.get("drop_at", 0)),
        seed=int(config.get("seed", -1)),
        sync_threshold=int(config.get("sync_threshold", 0)),
    )


def parse_generator_args() -> Namespace:
    """Parse command-line arguments for sample ChromaDB generation."""
    defaults = get_generator_defaults()
    return parse_args(
        description="Sample ChromaDB generator and random operator",
        defaults=defaults,
        arg_specs=[
            {
                "flags": ("-i", "--input-dir"),
                "dest": "input_dir",
                "type": Path,
                "help": f"Input documents directory (default: {config_path('generator', 'input_dir')})",
            },
            {
                "flags": ("-o", "--output-dir"),
                "dest": "output_dir",
                "type": Path,
                "help": (
                    "Base output directory; run data is stored under "
                    "{add}_{search}_{delete}_{update}_{upsert}_{new_collection}/ "
                    f"(default: {config_path('generator', 'output_dir')})"
                ),
            },
            {
                "flags": ("-n", "--iterations"),
                "dest": "iterations",
                "type": int,
                "help": f"Number of random operations to run (default: {defaults.iterations})",
            },
            {
                "flags": ("--add-weight",),
                "dest": "add_weight",
                "type": int,
                "help": f"Relative weight for add operations (default: {defaults.add_weight})",
            },
            {
                "flags": ("--search-weight",),
                "dest": "search_weight",
                "type": int,
                "help": f"Relative weight for search operations (default: {defaults.search_weight})",
            },
            {
                "flags": ("--delete-weight",),
                "dest": "delete_weight",
                "type": int,
                "help": f"Relative weight for delete operations (default: {defaults.delete_weight})",
            },
            {
                "flags": ("--update-weight",),
                "dest": "update_weight",
                "type": int,
                "help": f"Relative weight for update operations (default: {defaults.update_weight})",
            },
            {
                "flags": ("--upsert-weight",),
                "dest": "upsert_weight",
                "type": int,
                "help": f"Relative weight for upsert operations (default: {defaults.upsert_weight})",
            },
            {
                "flags": ("--new-collection-weight",),
                "dest": "new_collection_weight",
                "type": int,
                "help": f"Relative weight for new_collection operations (default: {defaults.new_collection_weight})",
            },
            {
                "flags": ("--drop-at",),
                "dest": "drop_at",
                "type": int,
                "help": (
                    "Drop the primary collection at this iteration; the remaining "
                    "iterations continue in the other (or a fresh) collection. "
                    f"0 disables the drop (default: {defaults.drop_at})"
                ),
            },
            {
                "flags": ("--seed",),
                "dest": "seed",
                "type": int,
                "help": f"Random seed for the operation sequence; -1 leaves it unseeded (default: {defaults.seed})",
            },
            {
                "flags": ("--sync-threshold",),
                "dest": "sync_threshold",
                "type": int,
                "help": (
                    "hnsw sync_threshold for the collections the generator creates; "
                    f"0 keeps Chroma's default (default: {defaults.sync_threshold})"
                ),
            },
        ],
    )


def build_parser_run_namespace(
    chroma_run_dir: Path,
    report_base: Path,
) -> Namespace:
    """Build parser args from a Chroma run directory and report base path."""
    chroma_run_dir = chroma_run_dir.resolve()
    report_base = report_base.resolve()
    run_label = chroma_run_dir.name
    return Namespace(
        chroma_run_dir=chroma_run_dir,
        report_base=report_base,
        run_label=run_label,
        chroma_db_path=chroma_run_dir / "chroma_db",
        output_dir=resolve_parser_output_dir(chroma_run_dir, report_base),
    )


def get_parser_defaults() -> Namespace:
    """Return default argument values for persisted parser."""
    config = load_config()
    parser_config = config["parser"]
    generator_config = config["generator"]
    chroma_run_dir = resolve_generator_run_dir(
        resolve_path(generator_config["output_dir"]),
        generator_config["add_weight"],
        generator_config["search_weight"],
        generator_config["delete_weight"],
        generator_config["update_weight"],
        generator_config["upsert_weight"],
        generator_config["new_collection_weight"],
        int(generator_config.get("drop_at", 0)),
        int(generator_config.get("sync_threshold", 0)),
    )
    report_base = resolve_path(parser_config["output_dir"])
    return build_parser_run_namespace(
        chroma_run_dir=chroma_run_dir,
        report_base=report_base,
    )


def parse_parser_args() -> Namespace:
    """Parse command-line arguments for persisted parser."""
    defaults = get_parser_defaults()
    args = parse_args(
        description="Persisted Chroma DB parser scaffold",
        defaults=defaults,
        arg_specs=[
            {
                "flags": ("-i", "--chroma-run-dir"),
                "dest": "chroma_run_dir",
                "type": Path,
                "help": (
                    "Chroma DB run directory containing chroma_db/ "
                    f"(default: {defaults.chroma_run_dir})"
                ),
            },
            {
                "flags": ("-o", "--report-base"),
                "dest": "report_base",
                "type": Path,
                "help": (
                    "Base report directory; output is stored under <run_label>/ "
                    f"(default: {config_path('parser', 'output_dir')})"
                ),
            },
        ],
    )
    return build_parser_run_namespace(
        chroma_run_dir=args.chroma_run_dir,
        report_base=args.report_base,
    )


def get_evaluator_defaults() -> Namespace:
    """Return default argument values for recovery evaluation."""
    config = load_config()
    parser_config = config["parser"]
    generator_config = config["generator"]
    chroma_run_dir = resolve_generator_run_dir(
        resolve_path(generator_config["output_dir"]),
        generator_config["add_weight"],
        generator_config["search_weight"],
        generator_config["delete_weight"],
        generator_config["update_weight"],
        generator_config["upsert_weight"],
        generator_config["new_collection_weight"],
        int(generator_config.get("drop_at", 0)),
        int(generator_config.get("sync_threshold", 0)),
    )
    report_base = resolve_path(parser_config["output_dir"])
    return Namespace(
        chroma_run_dir=chroma_run_dir,
        report_base=report_base,
    )


def parse_evaluator_args() -> Namespace:
    """Parse command-line arguments for recovery evaluation."""
    defaults = get_evaluator_defaults()
    args = parse_args(
        description="Recovery-rate evaluator",
        defaults=defaults,
        arg_specs=[
            {
                "flags": ("-i", "--chroma-run-dir"),
                "dest": "chroma_run_dir",
                "type": Path,
                "help": (
                    "Chroma DB run directory containing operations.sqlite3 "
                    f"(default: {defaults.chroma_run_dir})"
                ),
            },
            {
                "flags": ("-o", "--report-base"),
                "dest": "report_base",
                "type": Path,
                "help": (
                    "Base report directory holding <run_label>/embeddings.sqlite3 "
                    f"(default: {defaults.report_base})"
                ),
            },
        ],
    )
    return Namespace(
        chroma_run_dir=args.chroma_run_dir.resolve(),
        report_base=args.report_base.resolve(),
    )


def config_path(section: str, key: str) -> str:
    """Return a config path value as written in config.toml."""
    return str(load_config()[section][key])


def get_embedding_settings() -> Namespace:
    """Return embedding mode settings from config.toml."""
    embedding_config = load_config()["generator"].get("embedding", {})
    if not isinstance(embedding_config, dict):
        raise ValueError("[generator.embedding] must be a table in config.toml.")
    mode = str(embedding_config.get("mode", "default"))
    dimension = int(embedding_config.get("dimension", 384))
    return Namespace(mode=mode, dimension=dimension)
