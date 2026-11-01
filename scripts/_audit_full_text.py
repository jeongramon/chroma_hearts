"""Whole-chunk reference search, separate from recovery output and partial probes."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

from chroma_hearts.parser.source_copy import working_copy
from chroma_hearts.evaluator.sqlite_unallocated import KnownTexts, _text_needles


def whole_text_encodings(text: str) -> dict[str, bytes]:
    """Exact UTF-8 bytes and complete JSON string literals; no normalization."""
    if not text:
        raise ValueError("Empty text cannot define a full-chunk match")
    forms = {"utf8": text.encode("utf-8"),
             "json_utf8": json.dumps(text, ensure_ascii=False).encode("utf-8"),
             "json_ascii": json.dumps(text, ensure_ascii=True).encode("utf-8")}
    seen, unique = set(), {}
    for name, value in forms.items():
        if value not in seen:
            unique[name] = value
            seen.add(value)
    return unique


def find_full_text(files: dict[str, bytes], text: str) -> list[dict]:
    hits = []
    for encoding, needle in whole_text_encodings(text).items():
        for name, data in files.items():
            start = 0
            while (offset := data.find(needle, start)) >= 0:
                hits.append({"file": name, "offset": offset, "length": len(needle), "encoding": encoding})
                start = offset + 1
    return hits


def native_documents(store: Path) -> list[dict]:
    """Read native complete values on a disposable copy, never the parser's database."""
    documents = []
    with working_copy(store, prefix="whole_text_reference") as cp:
        with sqlite3.connect(f"file:{cp / 'chroma.sqlite3'}?mode=ro", uri=True) as conn:
            assert conn.execute("PRAGMA encoding").fetchone()[0] == "UTF-8"
            for seq, item_id, metadata in conn.execute(
                    "SELECT seq_id,id,metadata FROM embeddings_queue WHERE metadata IS NOT NULL"):
                document = json.loads(metadata).get("chroma:document")
                if isinstance(document, str):
                    documents.append({"table": "embeddings_queue", "row": seq, "id": item_id, "text": document})
            for row, item_id, document in conn.execute(
                    "SELECT m.id,e.embedding_id,m.string_value FROM embedding_metadata m "
                    "JOIN embeddings e ON e.id=m.id WHERE m.key='chroma:document' AND m.string_value IS NOT NULL"):
                documents.append({"table": "embedding_metadata", "row": row, "id": item_id, "text": document})
    return documents


def scan_full_texts(store: Path, manifest: dict) -> dict:
    files = {str(p.relative_to(store)): p.read_bytes() for p in sorted(store.rglob("*")) if p.is_file()}
    live = manifest["live"]
    active_texts = [item["text"] for item in live.values()]
    native = defaultdict(list)
    for row in native_documents(store):
        if row["id"] not in live or live[row["id"]]["text"] != row["text"]:
            native[row["text"]].append({k: v for k, v in row.items() if k != "text"})
    all_items = manifest.get("all_deleted", manifest["deleted"]) + list(live.values())
    known = KnownTexts(item["text"] for item in all_items)
    events = []
    for index, target in enumerate(manifest["deleted"]):
        text = target["text"]
        hits = find_full_text(files, text)
        overlaps_active = any(text in active for active in active_texts)
        logical = native[text]
        confirmed = bool(logical) or (bool(hits) and not overlaps_active)
        partial = []
        if not confirmed:
            for name, data in files.items():
                for needle in _text_needles(text, known):
                    offset = data.find(needle)
                    if offset >= 0:
                        partial.append({"file": name, "offset": offset, "length": len(needle)})
                        break
        events.append({"event": index, "id": target["id"],
                       "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "text_characters": len(text),
                       "raw_full_hits": hits, "native_full_values": logical,
                       "full_text_residual_confirmed": confirmed,
                       "overlaps_active_text": overlaps_active,
                       "ambiguous_active_overlap": bool(hits) and overlaps_active and not logical,
                       "partial_only": bool(partial) and not confirmed,
                       "partial_hits": partial})
    fields = ["full_text_residual_confirmed", "overlaps_active_text", "ambiguous_active_overlap", "partial_only"]
    return {"events": events, "counts": {k: sum(e[k] for e in events) for k in fields},
            "scope": "Exact whole UTF-8 chunks and complete JSON string literals per file, plus complete native "
                     "SQLite WAL/metadata document values on a copy. No cross-file joins, normalization, or raw "
                     "fragment reconstruction. Raw-only matches contained in active texts are excluded conservatively. "
                     "Distinctive partial windows are diagnostic only, not full-text positives."}


def verify_full_text_hits(store: Path, manifest: dict, reference: dict) -> int:
    """Re-read each raw offset and native value to validate the recorded reference."""
    files = {str(p.relative_to(store)): p.read_bytes() for p in store.rglob("*") if p.is_file()}
    native = {(r["table"], r["row"], r["id"]): r["text"] for r in native_documents(store)}
    checked = 0
    for target, event in zip(manifest["deleted"], reference["events"]):
        forms = whole_text_encodings(target["text"])
        for hit in event["raw_full_hits"]:
            needle = forms[hit["encoding"]]
            assert files[hit["file"]][hit["offset"]:hit["offset"] + hit["length"]] == needle
            if hit["encoding"].startswith("json"):
                assert json.loads(needle) == target["text"]
            else:
                assert needle.decode("utf-8") == target["text"]
            checked += 1
        for row in event["native_full_values"]:
            assert native[(row["table"], row["row"], row["id"])] == target["text"]
            checked += 1
    return checked
