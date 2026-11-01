# Recorded experiment results

These files contain recorded recovery, Open WebUI, and mitigation results.
Dates, original paths, and code hashes inside historical JSON are
preserved as provenance. Embedded absolute paths describe the original machine;
use [the current scripts](../../scripts/README.md) to create new, local outputs.

## Result tables

The CSVs in [tables/](tables/) are generated directly from the recorded JSON:

| Experiment | Readable table | Original result |
|---|---|---|
| Recovery comparison | [recovery_comparison.csv](tables/recovery_comparison.csv) | [Recovery paths](recovery_paths_carving_integrated_20261025.json) |
| Open WebUI | [openwebui_residue.csv](tables/openwebui_residue.csv) | [Case audit](openwebui_case_verified_20261025.json) |
| Staged mitigation | [mitigation_residue.csv](tables/mitigation_residue.csv) | [Mitigation audit](mitigation_audit_arxiv50_20261025.json) |

### Recovery recall

| Deletion | Data | Independent reference targets | Chroma Hearts | HNSW get_items |
|---|---|---:|---:|---:|
| Item Deletion | Vector | 2,211 | 100.00% (2,211) | 21.30% (471) |
| Item Deletion | Original text | 604 | 99.01% (598) | — |
| Collection Drop | Vector | 2,214 | 99.95% (2,213) | 58.54% (1,296) |
| Collection Drop | Original text | 2,172 | 100.00% (2,172) | — |

Each rate pools five runs. The denominator is targets whose complete data remains,
confirmed by the known-target byte scanner and independent SQLite decoding. The
API does not return text. Missing full values are six texts and one vector whose
bytes remain but whose record structure the production carver cannot identify.

### Open WebUI and mitigation: audit-confirmed residue

W/I/O/C mean WAL Full Entry, Identified Vector, Orphan Vector, SQLite Carved Data.
One representative category is selected in that order. Vector and text rates
include every recovery path. Their denominator is **all deleted targets**, unlike
the [recovery recall](#recovery-recall) above.

| State | Targets | W % | I % | O % | C % | Vector % | Text % |
|---|---:|---:|---:|---:|---:|---:|---:|
| Open WebUI S1 | 1,241 | 0.00 | 55.76 | 44.24 | 0.00 | 100.00 | 70.59 |
| Open WebUI S2 | 2,434 | 0.00 | 81.06 | 0.00 | 18.41 | 97.33 | 94.21 |
| C0 | 2,211 | 6.69 | 21.17 | 72.14 | 0.00 | 100.00 | 27.05 |
| C1 | 2,211 | 0.00 | 0.00 | 100.00 | 0.00 | 100.00 | 27.00 |
| C2 | 2,211 | 0.00 | 0.00 | 0.00 | 5.34 | 0.00 | 5.34 |
| C3 | 2,211 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| D0 | 2,312 | 0.00 | 56.06 | 0.00 | 43.25 | 95.72 | 93.94 |
| D1 | 2,312 | 0.00 | 0.00 | 0.00 | 93.94 | 39.71 | 93.94 |
| D2 | 2,312 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

C1 persists deletion changes and prunes WAL; C2 rebuilds from active items and
removes obsolete indexes; C3 additionally runs SQLite VACUUM. D1 removes the
unreferenced directories after Drop; D2 additionally runs VACUUM. Each step
includes its predecessors. S1's C=0 does not mean carving found nothing: its
876 recovered texts belong to targets already represented by HNSW paths.

## Supporting validation

| Claim | Evidence |
|---|---|
| Every output value and known ID link checked against insertion history; no errors | [Output validation](recovery_output_values_arxiv50_20261026.json) |
| Final C3/D2 full-vector and full-text scans find no known deleted targets; positive controls pass | [Independent byte-scanner verification](mitigation_full_bytes_arxiv50_20261026.json) |
| Active values/settings preserved; 18,092 fresh-process queries return only active items | [Search and preservation verification](mitigation_search_arxiv50_20261026.json) |
| Five operation plans and paired source-snapshot hashes | [Original workload record](workload_audit_1.5.9_arxiv50_repeat_20261025.json) |
| PDF/chunk hashes, original dependencies and source-code hashes | [Workload provenance](workload_audit_provenance_1.5.9_arxiv50_repeat_20261025.json) |

The original workload record predates integrated SQLite carving: its old aggregate
does not include carved data. Use `recovery_paths_carving_integrated_20261025.json`
for final recall; retain the workload record only for source provenance. Historical
code hashes and removed intermediate artifacts can be resolved in Git history at
`d49fa5f`. Large per-target references and generated stores are local working data,
not a downloadable release archive.

Value-validation output counts include overlapping evidence and are not physical
copy counts. Search checks establish functional operation, not unchanged ANN
rankings. Final no-hit results cover the remaining files and full target values;
they do not establish that all fragments or filesystem-level copies were erased.
Diagnostic time/space fields in the historical mitigation JSON are retained but
are not a cost benchmark.

Regenerate the three CSV tables without opening any evidence:

```bash
python scripts/summarize_results.py
```

Earlier synthetic-PDF results, update/upsert and ratio/threshold sweeps, scale/cost
benchmarks, semantic-recall experiments, and intermediate reports have been removed
from this directory. Their historical versions remain in Git; current reproduction
starts from the arXiv manifest and bundled Open WebUI snapshots.
