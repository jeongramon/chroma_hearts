"""Shared paths, provenance and isolated-process helpers for paper experiments."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

EXPERIMENT_STORES = REPO_ROOT / "sample" / "experiments"
RESULTS_DIR = REPO_ROOT / "sample" / "experiment_results"


def subprocess_env() -> dict[str, str]:
    """Environment that makes ``chroma_hearts`` importable from src/."""
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(SRC_DIR) + (os.pathsep + existing if existing else "")
    return env


def run(cmd: list[str], log: Path | None = None) -> None:
    """Run a subprocess from the repo root, raising on failure (output to ``log``)."""
    if log is None:
        subprocess.run(cmd, cwd=str(REPO_ROOT), env=subprocess_env(), check=True)
        return
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as handle:
        handle.write("[RUN] " + " ".join(cmd) + "\n")
        handle.flush()
        subprocess.run(cmd, cwd=str(REPO_ROOT), env=subprocess_env(), check=True,
                       stdout=handle, stderr=subprocess.STDOUT)


def chromadb_version() -> str:
    import chromadb

    return chromadb.__version__


def corpus_provenance(input_dir: Path) -> dict:
    """Record the actual PDFs and the common extraction output, without creating data."""
    from chroma_hearts.generator.document_processing import load_documents_from_dir

    input_dir = input_dir.resolve()
    pdfs = sorted(input_dir.glob("*.pdf"))
    if not pdfs:
        raise ValueError(f"No PDF inputs: {input_dir}")
    ids, documents, metadata = load_documents_from_dir(input_dir)
    if {m["source"] for m in metadata} != {p.name for p in pdfs}:
        raise ValueError("Some PDFs produced no text chunks")
    pdf_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in pdfs}
    chunk_hashes = {i: hashlib.sha256(d.encode()).hexdigest() for i, d in zip(ids, documents)}
    return {"input_dir": str(input_dir), "pdf_count": len(pdfs),
            "pdf_sha256": pdf_hashes,
            "corpus_sha256": hashlib.sha256(json.dumps(pdf_hashes, sort_keys=True).encode()).hexdigest(),
            "chunk_size": 2000, "overlap": 300, "corpus_chunks": len(documents),
            "unique_chunks": len(set(documents)), "chunk_sha256": chunk_hashes}


def prepare_pdf_experiment(args: Any, prefix: str, version: str) -> Path:
    """Keep tagged runs separate and save provenance next to their CSV results."""
    if not args.tag or Path(args.tag).name != args.tag or args.tag in (".", ".."):
        raise ValueError("tag must be a single directory name")
    source = corpus_provenance(args.input_dir)
    root = EXPERIMENT_STORES / f"{prefix}_{version}_{args.tag}"
    provenance_path = RESULTS_DIR / f"{prefix}_provenance_{version}_{args.tag}.json"
    reuse = getattr(args, "reuse_runs", False)
    if reuse:
        previous = json.loads(provenance_path.read_text())
        assert previous["source"] == source, "Input corpus changed"
    else:
        if provenance_path.exists():
            raise FileExistsError(provenance_path)
        root.mkdir(parents=True, exist_ok=False)
        code_paths = sorted((REPO_ROOT / "scripts").glob("*.py")) + sorted(SRC_DIR.rglob("*.py"))
        record = {"source": source, "arguments": {k: str(v) if isinstance(v, Path) else v
                                                for k, v in vars(args).items()},
                  "environment": {"python": sys.version, "platform": platform.platform(),
                                  **{p: importlib.metadata.version(p) for p in
                                     ["chromadb", "chroma-hnswlib", "PyMuPDF", "numpy"]}},
                  "code_sha256": {str(p.relative_to(REPO_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in code_paths}}
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        provenance_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(f"[DATA] {source['pdf_count']} PDFs, {source['corpus_chunks']} chunks; {root}", flush=True)
    return root


def spawn_call(func: Callable[..., Any], *args: Any) -> Any:
    """Call ``func`` in a freshly spawned process, so no Chroma state is shared."""
    with multiprocessing.get_context("spawn").Pool(1) as pool:
        return pool.apply(func, args)


def write_csv(rows: list[dict], path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] wrote {path} ({len(rows)} rows)")

def fingerprint(store: Path) -> dict:
    return {str(p.relative_to(store)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(store.rglob('*')) if p.is_file()}

def _vector(text: str) -> bytes:
    import array

    from chroma_hearts.generator.embedding import hash_text_to_vector

    return array.array("f", hash_text_to_vector(text, 384)).tobytes()

def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
