"""Export the manuscript's tables from recorded results without opening evidence."""
import argparse
import csv
import json
from pathlib import Path
from _experiment_common import RESULTS_DIR


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--recovery', type=Path, default=RESULTS_DIR/'recovery_paths_carving_integrated_20261025.json')
    ap.add_argument('--openwebui', type=Path, default=RESULTS_DIR/'openwebui_case_verified_20261025.json')
    ap.add_argument('--mitigation', type=Path, default=RESULTS_DIR/'mitigation_audit_arxiv50_20261025.json')
    ap.add_argument('--output-dir', type=Path, default=RESULTS_DIR/'tables')
    args = ap.parse_args()
    load = lambda p: json.loads(p.read_text())
    recovery = load(args.recovery)
    rows = []
    for row in recovery['aggregate']:
        n = row['targets']
        api = row['get_items'] if row['metric'] == 'vector' else None
        rows.append(dict(deletion=row['deletion'], metric=row['metric'], reference_targets=n,
                         chroma_hearts=row['chroma_hearts'], get_items=api,
                         chroma_hearts_recall_pct=round(100*row['chroma_hearts']/n, 2) if n else None,
                         get_items_recall_pct=round(100*api/n, 2) if api is not None and n else None))
    tables = {'recovery_comparison': rows}
    fields = ('wal_full_entry','identified_vector','orphan_vector','sqlite_carved_data',
              'vector_recovered','text_recovered')
    for name, source, key, label in [('openwebui_residue',args.openwebui,'runs','deletion'),
                                     ('mitigation_residue',args.mitigation,'aggregate','stage')]:
        rows = []
        for row in load(source)[key]:
            n = row['targets']
            assert sum(row[k] for k in (*fields[:4], 'none')) == n
            result = {label: row[label], 'targets': n}
            for field in fields:
                result[field] = row[field]
                result[field+'_pct'] = round(100*row[field]/n, 2) if n else None
            rows.append(result)
        tables[name] = rows
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        with (args.output_dir/(name+'.csv')).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
        print(name, json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
