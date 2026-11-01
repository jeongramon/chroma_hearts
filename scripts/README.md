# Running the experiments

Run commands from the repository root after installing the packages in the main
[README](../README.md). No old experiment output is required for a new run.
The dated JSON files in `sample/experiment_results/` contain recorded experiment
results. New runs use the default `paper` tag and do not overwrite those files.

## Inputs

```bash
# Verify PDFs already in sample/documents_arxiv50:
python scripts/prepare_pdfs.py
# Or copy matching PDFs from a local collection (filenames may differ):
python scripts/prepare_pdfs.py --source-dir /path/to/pdfs
# Or explicitly download the 50 versioned arXiv PDFs:
python scripts/prepare_pdfs.py --download
```

[sample/arxiv50_manifest.json](../sample/arxiv50_manifest.json) records every PDF
and chunk hash, versioned download URLs, and the original extraction environment.
The script rejects different bytes and verifies 1,161 chunks at 2,000 characters
with 300-character overlap. Downloads depend on arXiv availability. If a PDF is
served with different bytes, obtain the matching archived file; do not silently
replace the reference hash. The reference environment used PyMuPDF 2.0 and NumPy
2.5.4; the experiment requirements pin these versions for repeatable extraction.

## Controlled validation

```bash
python scripts/run_workload_audit.py --input-dir sample/documents_arxiv50
python scripts/verify_recovery_output_values.py
```

The first command creates five 1,500-choice workloads using seeds 1–5 and
add/delete/search probabilities 60%/30%/10%. Each chunk is inserted at most once.
A paired run uses the same operation plan and ends with Collection Drop. Embeddings
are deterministic 384-dimensional float32 hashes; L2 and `sync_threshold=1000`
are the Chroma defaults. The writer process exits before auditing.

For each snapshot the script:

1. Runs the production parser without deleted values and runs `get_items()` on a
   separate copy, without ID mappings or WAL access.
2. Constructs the independent full-vector/full-text reference using known targets,
   raw searches, native SQLite values, and a separate record probe whose payloads
   are decoded again by SQLite. Partial matches do not enter the denominator.
3. Grades production output and saves per-target evidence, reference counts, and
   pooled recall. Text recall for the vector-only API is not applicable.

`verify_recovery_output_values.py` checks every merged/source output against the
complete insertion history, including vector/text pairing and known ID links.
It is a value-validity check, not a general deletion-attribution precision estimate.
The report retains every mismatch and the command exits unsuccessfully if any are
found.

Stores, plans, and reports are written beneath
`sample/experiments/workload_audit_1.5.9_paper/`; the summary is
`sample/experiment_results/workload_audit_1.5.9_paper.json`.
Use a new `--tag` for another run. `--seeds 1` is useful for checking the workflow
but does not reproduce the recorded five-run aggregate.

## Open WebUI case study

```bash
python scripts/run_openwebui_audit.py
```

The three [bundled snapshots](../sample/README.md) are sufficient. The driver builds
the before-reference on a disposable copy and audits S1/S2 without target values.
It uses Open WebUI's cosine setting for remaining HNSW directories, including the
one left by Collection Drop, and restores scales with `length.bin`.

Grading is restricted to the knowledge-base collection. File-specific collections
are outside the deletion scope. Whole-vector matches accept either the before-SDK
or the same item's before-WAL bytes; document matches are exact. Historical SQLite
PK/text attribution is used only during grading. The byte scanner is not used.

Results: `sample/experiment_results/openwebui_paper.json`; details:
`sample/experiments/openwebui_paper/`. The driver requires a fresh output directory.
Use `--input-dir`, `--output-dir`, `--result`, and `--collection-name` for an explicit
alternative input or destination. This command re-audits the captured application
states; it does not recreate the application's original upload/deletion session.

## Mitigation

```bash
python scripts/run_mitigation_audit.py
python scripts/verify_mitigation_full_bytes.py
python scripts/verify_mitigation_search.py
```

Run [controlled validation](#controlled-validation) first. The mitigation driver
uses the same ten snapshots and keeps 35 derived states: C0–C3 and D0–D2 for each
seed. It grades only production-parser
output. The two verification commands independently check final full-value absence
(with positive controls), and active data/settings plus fresh-process top-10 search.
They do not modify the original snapshots. Timing fields retained in raw JSON are
diagnostics; these checks do not establish scale/cost or ANN-quality performance.

Default outputs are `mitigation_paper.json`, `mitigation_full_bytes_paper.json`,
and `mitigation_search_paper.json` under `sample/experiment_results/`.
For other tags, pass the corresponding `--workload`, `--mitigation`, and `--output`
paths; `run_mitigation_audit.py` also accepts `--tag` and `--seeds`.
A separate historical recovery summary can be supplied with `--recovery` to the
mitigation driver and output-value checker.

## Tables and recorded results

```bash
# Export the recorded results; requires no SDK access or generated stores:
python scripts/summarize_results.py
# Export a new run separately:
python scripts/summarize_results.py \
  --recovery sample/experiment_results/workload_audit_1.5.9_paper.json \
  --openwebui sample/experiment_results/openwebui_paper.json \
  --mitigation sample/experiment_results/mitigation_paper.json \
  --output-dir sample/experiments/result_tables
```

The default export produces the three CSV tables in
[sample/experiment_results/tables](../sample/experiment_results/tables/).
Recorded JSON values are not rewritten to fit new runs. PDF hashes and operation
plans are repeatable; exact physical residue can also depend on the runtime,
metadata/path lengths, allocation, and persistence behavior. Compare new results
with their own independently verified reference, not by forcing recorded counts.

## Maintained entry points

| Script | Purpose |
|---|---|
| `prepare_pdfs.py` | Obtain/verify the exact PDF and chunk corpus |
| `run_workload_audit.py` | Generate paired workloads and evaluate recovery |
| `run_openwebui_audit.py` | Audit and grade the captured application states |
| `run_mitigation_audit.py` | Apply and audit staged mitigation on copies |
| `verify_recovery_output_values.py` | Validate recovered values and ID connections |
| `verify_mitigation_full_bytes.py` | Independently search final states for known full values |
| `verify_mitigation_search.py` | Check active data, settings, and functional searches |
| `summarize_results.py` | Export the three experiment result tables |

`_*.py` modules contain shared process/provenance helpers, before-snapshot
extraction, independent scanners, and SQLite reference decoding. Superseded
synthetic-PDF, update/upsert, ratio/threshold, scale/cost, semantic-recall, and
intermediate-report drivers were removed from the current tree; their source and
observations remain in Git history (the pre-cleanup revision is `d49fa5f`).
