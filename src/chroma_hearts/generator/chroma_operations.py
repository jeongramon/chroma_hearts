"""
Chroma DB operations for add, search, and delete flows.
"""

from pathlib import Path
import random
import uuid
from typing import Sequence

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings

from chroma_hearts.config import get_embedding_settings
from .document_processing import collect_pdf_files, extract_text_from_pdf, split_text_into_chunks
from .embedding import build_embedding_function

COLLECTION_NAME = "sample_documents"
DEFAULT_QUERY_COUNT = 3
_ADD_QUEUE = []
_ADD_CURSOR = 0
_ADD_SOURCE_DIR = None

QUESTION_POOL = [
    "What is the author's main argument in this document?",
    "Which passage describes the local economy in the 19th century?",
    "Where does the text mention changes in the education system?",
    "Is there a section about religious activity or church records?",
    "Find a paragraph discussing war or social conflict.",
    "Is there a passage about the life of a key historical figure?",
    "Show content that explains geography or place names.",
    "Does the text mention migration or population movement?",
    "Where are administrative boundary changes discussed?",
    "Find references to legal reforms, ordinances, or institutions.",
    "Is there any discussion of agriculture or crop production?",
    "Where does the document describe trade or commerce?",
    "Find content about transportation development like railroads or roads.",
    "Is there information related to family history or genealogy?",
    "Find a section listing events by year.",
    "Where are statistics such as population, tax, or production mentioned?",
    "Is there any record of disasters, epidemics, or famine?",
    "Find content about founding local schools or colleges.",
    "Where does the text discuss women's roles or social status?",
    "Is there a passage describing labor conditions or occupations?",
    "Find sentences explaining grammar or pronunciation rules.",
    "Is there a section with conversational examples or dialogues?",
    "Find dictionary-style definitions of words.",
    "Where are synonyms, antonyms, or related expressions presented?",
    "Show a passage that includes sentence translation examples.",
    "Is there an explanation of idioms or proverbs?",
    "Find parts that classify words by part of speech.",
    "Where does the text explain verb conjugation or tense?",
    "Is pronunciation notation or orthography guidance included?",
    "Find exercises intended for language learners.",
    "Is there an alphabetical index of names or places?",
    "Find where the origin or etymology of a word is explained.",
    "Show content that looks like a historical timeline.",
    "Is there a quotation from a letter, speech, or official document?",
    "Find examples describing cultural customs or practices.",
    "Is there content about rituals such as marriage or funerals?",
    "Where is industrial development explained with cause and effect?",
    "Find references to colonial policy or administrative rule.",
    "Is there a section about mapping, surveying, or border setting?",
    "Find records of court judgments or dispute resolution.",
    "Where does the text discuss prices, currency, or taxation?",
    "Find a passage that reads like a preface or conclusion.",
    "Show a part of the document that is dense with quotations.",
    "Find a section where proper nouns appear repeatedly.",
    "Where is a key concept explicitly defined?",
    "Is there any question-and-answer or FAQ-style text?",
    "Find descriptions of natural environment such as rivers or climate.",
    "Where does the text discuss urbanization or settlement formation?",
    "Is there content resembling references, sources, or footnotes?",
    "Find a passage explaining social class or status hierarchy.",
]


_SYNC_THRESHOLD = 0


def set_sync_threshold(value: int) -> None:
    """Use this hnsw sync_threshold for collections created from now on (0 = default)."""
    global _SYNC_THRESHOLD
    _SYNC_THRESHOLD = max(0, int(value))


def _hnsw_configuration() -> dict:
    """Collection configuration kwargs carrying the chosen sync_threshold, if any."""
    if _SYNC_THRESHOLD:
        return {"configuration": {"hnsw": {"sync_threshold": _SYNC_THRESHOLD}}}
    return {}


def _get_embedding_function():
    """Build the configured embedding function for new collections."""
    settings = get_embedding_settings()
    return build_embedding_function(settings.mode, settings.dimension)


def _get_client(output_dir: Path) -> chromadb.PersistentClient:
    """Create a persistent Chroma client from output dir."""
    resolved_output = output_dir.resolve()
    resolved_output.mkdir(parents=True, exist_ok=True)
    chroma_path = resolved_output / "chroma_db"
    return chromadb.PersistentClient(
        path=str(chroma_path),
        settings=Settings(anonymized_telemetry=False),
    )


def get_client(output_dir: Path) -> chromadb.PersistentClient:
    """Public client factory for reuse across operations."""
    return _get_client(output_dir)


def create_collection(output_dir: Path) -> Collection:
    """Create a clean collection in a persistent local Chroma path."""
    client = _get_client(output_dir)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    return client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Sample documents vector store"},
        embedding_function=_get_embedding_function(),
        **_hnsw_configuration(),
    )


def create_random_collection(output_dir: Path) -> str:
    """Create an extra collection with a random name."""
    client = _get_client(output_dir)
    name = f"sample_extra_{uuid.uuid4().hex[:8]}"
    client.get_or_create_collection(
        name=name,
        metadata={"description": "Extra sample collection"},
        embedding_function=_get_embedding_function(),
    )
    return name


def create_random_collection_with_client(client: chromadb.PersistentClient) -> Collection:
    """Create and return an extra collection using an existing client."""
    name = f"sample_extra_{uuid.uuid4().hex[:8]}"
    return client.get_or_create_collection(
        name=name,
        metadata={"description": "Extra sample collection"},
        embedding_function=_get_embedding_function(),
        **_hnsw_configuration(),
    )


def get_collection(output_dir: Path, name: str) -> Collection:
    """Load one collection by name."""
    client = _get_client(output_dir)
    return client.get_collection(name=name)


def _prepare_add_queue(input_dir: Path) -> None:
    """Prepare or refresh the in-memory add queue for sequential chunk inserts."""
    global _ADD_QUEUE, _ADD_CURSOR, _ADD_SOURCE_DIR
    resolved_input = input_dir.resolve()
    if _ADD_SOURCE_DIR == resolved_input and _ADD_QUEUE:
        return

    queue = []
    for pdf_path in collect_pdf_files(resolved_input):
        text = extract_text_from_pdf(pdf_path)
        chunks = split_text_into_chunks(text)
        if not chunks:
            continue

        for idx, chunk in enumerate(chunks):
            queue.append(
                (
                    f"{pdf_path.stem}_chunk_{idx:05d}",
                    chunk,
                    {
                        "source": pdf_path.name,
                        "path": str(pdf_path),
                        "chunk_index": idx,
                        "chunk_size": len(chunk),
                    },
                )
            )

    _ADD_QUEUE = queue
    _ADD_CURSOR = 0
    _ADD_SOURCE_DIR = resolved_input


def add_documents(
    collection: Collection, input_dir: Path
) -> tuple[int, str | None, str | None]:
    """Add one chunk per call, recycling the queue when chunks are deleted.

    Returns ``(count, doc_id, document)`` so the operation log can record the
    exact id and text embedded for later recovery verification.
    """
    global _ADD_CURSOR
    _prepare_add_queue(input_dir)

    if not _ADD_QUEUE:
        print("[ADD] No chunked documents available to add.")
        return 0, None, None

    total = len(_ADD_QUEUE)
    for _ in range(total):
        chunk_id, chunk_text, chunk_metadata = _ADD_QUEUE[_ADD_CURSOR]
        _ADD_CURSOR = (_ADD_CURSOR + 1) % total

        existing = collection.get(ids=[chunk_id], include=[])
        if existing.get("ids"):
            continue

        collection.add(
            ids=[chunk_id],
            documents=[chunk_text],
            metadatas=[chunk_metadata],
        )
        return 1, chunk_id, chunk_text

    return 0, None, None


def search_documents(collection: Collection, n_queries: int = DEFAULT_QUERY_COUNT) -> int:
    """Run semantic search using random queries from a fixed query pool."""
    total_docs = collection.count()
    if total_docs == 0:
        print("[SEARCH] Skipped: collection is empty.")
        return 0

    sample_size = min(n_queries, len(QUESTION_POOL))
    sampled_queries = random.sample(QUESTION_POOL, sample_size)
    for idx, query in enumerate(sampled_queries, start=1):
        #print(f"[SEARCH] Query {idx}/{sample_size}: {query}")
        results = collection.query(
            query_texts=[query],
            n_results=min(3, total_docs),
            include=["documents", "metadatas", "distances"],
        )
        for rank, (doc_id, metadata, distance) in enumerate(
            zip(results["ids"][0], results["metadatas"][0], results["distances"][0]),
            start=1,
        ):
            source = metadata.get("source") if metadata else "unknown"
            #print(f"  {rank}. id={doc_id}, source={source}, distance={distance:.4f}")
    return sample_size


def delete_random_document(collection: Collection) -> tuple[int, str | None, str | None]:
    """Delete one random document from the collection if available.

    Returns ``(count, doc_id, None)``; the deleted text is not fetched here
    since the evaluator reconstructs it by replaying the operation log.
    """
    total_docs = collection.count()
    if total_docs == 0:
        print("[DELETE] Skipped: collection is empty.")
        return 0, None, None

    existing = collection.get(limit=1, offset=random.randrange(total_docs), include=[])
    target_ids: Sequence[str] = existing.get("ids", [])
    if not target_ids:
        print("[DELETE] Skipped: no deletable ids found.")
        return 0, None, None

    collection.delete(ids=list(target_ids))
    #print(f"[DELETE] Removed ids: {list(target_ids)}")
    return len(target_ids), target_ids[0], None


def update_random_document(collection: Collection) -> tuple[int, str | None, str | None]:
    """Update one random existing document.

    Returns ``(count, doc_id, updated_document)`` with the exact text embedded.
    """
    total_docs = collection.count()
    if total_docs == 0:
        print("[UPDATE] Skipped: collection is empty.")
        return 0, None, None

    existing = collection.get(limit=1, offset=random.randrange(total_docs), include=["documents", "metadatas"])
    target_ids: Sequence[str] = existing.get("ids", [])
    if not target_ids:
        print("[UPDATE] Skipped: no updatable ids found.")
        return 0, None, None

    doc = (existing.get("documents") or [""])[0] or ""
    meta = (existing.get("metadatas") or [None])[0] or {}
    updated_doc = f"{doc}\n[updated]"
    meta["last_operation"] = "update"
    collection.update(ids=list(target_ids), documents=[updated_doc], metadatas=[meta])
    return 1, target_ids[0], updated_doc


def upsert_random_document(collection: Collection) -> tuple[int, str | None, str | None]:
    """Upsert one document (existing id if possible, otherwise new id).

    Returns ``(count, doc_id, document)`` with the exact text embedded.
    """
    total_docs = collection.count()
    if total_docs > 0:
        existing = collection.get(limit=1, offset=random.randrange(total_docs), include=["documents", "metadatas"])
        target_ids: Sequence[str] = existing.get("ids", [])
        if target_ids:
            doc = (existing.get("documents") or [""])[0] or ""
            meta = (existing.get("metadatas") or [None])[0] or {}
            meta["last_operation"] = "upsert_existing"
            upserted_doc = f"{doc}\n[upserted]"
            collection.upsert(
                ids=list(target_ids),
                documents=[upserted_doc],
                metadatas=[meta],
            )
            return 1, target_ids[0], upserted_doc

    new_id = f"upsert_{uuid.uuid4().hex[:12]}"
    new_doc = "Generated by upsert operation."
    collection.upsert(
        ids=[new_id],
        documents=[new_doc],
        metadatas=[{"source": "upsert", "last_operation": "upsert_new"}],
    )
    return 1, new_id, new_doc
