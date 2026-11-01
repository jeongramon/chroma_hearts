# Chroma Hearts

Chroma Hearts audits deletion residue in local, persisted Chroma stores. It extracts
original text, vectors, and ID mappings from Chroma's logical WAL, raw HNSW files,
and intact SQLite records in unused space. SDK queries on a separate copy identify
the currently active items; they are not the source of deleted data.

The experiments evaluate recovery on paired item-deletion / Collection Drop
workloads, audit an Open WebUI knowledge base, and evaluate staged cleanup.
See [reproduction instructions](scripts/README.md), [sample data](sample/README.md),
and [the recorded results](sample/experiment_results/README.md).

## Installation

Python 3.11 or newer (reference experiment environment: Python 3.12.3).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
# Needed for the HNSW library comparison:
pip install -r requirements-experiments.txt
```

The controlled workload uses `chromadb==1.5.9`. The bundled Open WebUI snapshots
were produced by Open WebUI 0.7.2 with Chroma 1.4.0. These are the tested formats,
not a claim of compatibility with every Chroma version.

## Audit a store

```bash
python -m chroma_hearts.parser -i /path/to/run -o /path/to/reports
```

The input layout is `/path/to/run/chroma_db/chroma.sqlite3`, alongside HNSW segment
directories. Reports go to `/path/to/reports/run/`. Use a new output directory for
each audit: the parser replaces an existing report with the same run label.
The [Open WebUI driver](scripts/README.md#open-webui-case-study) accepts
its bundled `vector_db/` layout directly.

The parser copies the source before SQLite analysis. SDK access uses another copy
so that backfill or pruning cannot change the static artifacts under analysis.
Stop the application before collecting a snapshot, keep reports outside the input,
and allow temporary disk space for two copies.

| Output | Contents |
|---|---|
| `embeddings.sqlite3` | Merged recovered values and a separate `recovery_evidence` table retaining each source |
| `sysdb.sqlite3` | Collection and segment metadata |
| `sqlite_carving.json` | Carving status, record count, and source hash |
| `persistence.sqlite3` | Persisted sequence, pending WAL records, and `sync_threshold` per collection |

In `embeddings`, `source_chroma_sdk`, `source_wal`, `source_hnsw`, and
`source_sqlite_carved` identify the extraction sources. `vector` contains float32
bytes; original text is stored as `chroma:document` in `metadata`. Unknown IDs stay
`NULL`. In `recovery_evidence`, `source_file`, `source_table`, `region_type`,
`page_no`, `record_offset`, and `byte_ranges` retain the available provenance.

An item absent from the SDK snapshot is a residue candidate. Confirming its deletion
requires a deletion record or a before-snapshot; absence alone does not establish
when or why it was removed.

## Residue categories and metrics

For a known deleted target, the evaluator selects one representative category in
this priority order:

| Priority | Category | Recovery path |
|---|---|---|
| 1 | WAL Full Entry | Current `embeddings_queue` record with ID, vector, and original text |
| 2 | Identified Vector | Persisted HNSW vector with an ID mapping |
| 3 | Orphan Vector | Persisted HNSW vector without an ID mapping |
| 4 | SQLite Carved Data | Additional recovery from intact records in SQLite unused regions |

SQLite Carved Data can contain text, vectors, or both. The priority is a reporting
rule, not the order in which copies disappear. All source evidence is retained;
vector and text totals include every matching path, including carved text for a
target represented as an HNSW vector. An unreferenced directory is not synonymous
with an Orphan Vector: its ID mapping may still exist.

- **Controlled validation recall:** recovered targets / targets whose complete vector or
  complete original text independently remains in the snapshot. The known-target
  byte scanner and SQLite decoding construct this reference after blind extraction.
- **Open WebUI and mitigation residue rates:** audit-confirmed targets / all deleted targets.
  These rates do not use the byte scanner as their denominator.
- **Final mitigation verification:** independent known-target full-value scans of
  C3 and D2 supplement the Chroma Hearts audit.

The parser does not take the expected deleted values as input. SQLite carving uses
record layout checks and preserved overflow chains; it does not reconstruct missing
record headers. File-system unallocated space and process memory are outside scope.

## Run the experiments

There are three main experiment commands, with separate validation commands:

```bash
python scripts/prepare_pdfs.py --download
python scripts/run_workload_audit.py --input-dir sample/documents_arxiv50
python scripts/run_openwebui_audit.py
python scripts/run_mitigation_audit.py
```

See [scripts/README.md](scripts/README.md) for the complete sequence, validation
commands, input hashes, and the distinction between recorded results and new runs.
Each public script supports `--help`. Private `_*.py` files are shared helpers.

The interactive menu remains available through `python main.py`. The general
`pdf_collector`, `generator`, and `evaluator` modules are utilities; the experiment
protocol is implemented by the dedicated scripts above.

## Mitigation

`chroma_hearts.mitigation` provides offline maintenance operations. They modify the
store, require exclusive access, and should not be applied to original evidence.
The experiment driver applies them only to derived copies.

| Sequence | Action |
|---|---|
| C0 → C1 | Persist pending deletion changes and prune the corresponding logical WAL |
| C1 → C2 | Rebuild from active items, verify the copy, and remove obsolete index directories |
| C2 → C3 | Rebuild the SQLite file with `VACUUM` |
| D0 → D1 | After Collection Drop, remove unreferenced segment directories |
| D1 → D2 | Rebuild the SQLite file with `VACUUM` |

Persistence alone can leave HNSW soft-deleted vectors and SQLite remnants.
The evaluation checks active IDs, documents, metadata, vectors, settings, and
functional searches. It does not establish invariant ANN rankings, online
availability, or erasure of the entire storage medium.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests cover evidence isolation, HNSW normalization, recovery-path grading,
SQLite record handling, independent full-value scans, and the workload/API baseline.
