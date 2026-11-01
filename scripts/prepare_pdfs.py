"""Prepare and verify the paper's 50 arXiv PDFs and 1,161 extracted chunks."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request
from _experiment_common import REPO_ROOT, corpus_provenance


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-dir', type=Path, help='Copy matching PDF bytes from this directory (recursive)')
    ap.add_argument('--download', action='store_true', help='Download missing versioned PDFs from arXiv')
    ap.add_argument('--output-dir', type=Path, default=REPO_ROOT/'sample/documents_arxiv50')
    args = ap.parse_args()
    manifest = json.loads((REPO_ROOT/'sample/arxiv50_manifest.json').read_text())
    expected = manifest['pdf_sha256']
    destination = args.output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    copies = {}
    if args.source_dir:
        for path in args.source_dir.rglob('*.pdf'):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in expected.values():
                copies[digest] = path
    for name, digest in expected.items():
        path = destination/name
        if path.exists():
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f'Existing PDF has unexpected bytes: {path}')
            continue
        if digest in copies:
            data = copies[digest].read_bytes()
        elif args.download:
            request = urllib.request.Request(manifest['download_urls'][name],
                                             headers={'User-Agent': 'Chroma-Hearts-reproduction/0.1'})
            with urllib.request.urlopen(request, timeout=120) as response:
                data = response.read()
            time.sleep(3)
        else:
            raise FileNotFoundError(f'{path}: provide --source-dir or --download')
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f'{name}: PDF differs from the published input; not saved')
        path.write_bytes(data)
        print('[PDF]', name, flush=True)
    actual = corpus_provenance(destination)
    for key in ('pdf_sha256', 'corpus_chunks', 'chunk_sha256'):
        if actual[key] != manifest[key]:
            raise ValueError(f'{key} differs; use the recorded extraction versions and no extra PDFs')
    print(f"Verified {actual['pdf_count']} PDFs and {actual['corpus_chunks']} chunks: {destination}")


if __name__ == '__main__':
    main()
