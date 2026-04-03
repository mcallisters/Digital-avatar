"""
src/vector_store.py
─────────────────────────────────────────────────────────────────────────────
ChromaDB-backed semantic search over publication chunks produced by chunker.py.

Responsibilities:
  - Load all __chunks.json files from data/json_outputs/
  - Embed chunk content via OpenAI text-embedding-3-small
  - Store + persist in ChromaDB (local, no server needed)
  - Semantic search at query time with optional filters
  - Formatted context builder for LLM prompts

Usage
─────
  # At app startup:
  from src.vector_store import init_vector_store
  init_vector_store()

  # In publications agent:
  from src.vector_store import get_publications_context
  context = get_publications_context(user_query)

  # To add one new paper without rebuilding:
  from src.vector_store import index_chunks_file
  index_chunks_file("data/json_outputs/new_paper__chunks.json")
"""

import json
import os
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions


# ── Config ────────────────────────────────────────────────────────────────────

CHROMA_PERSIST_DIR     = Path(__file__).parent.parent / "data" / "chroma_db"
JSON_OUTPUTS_DIR       = Path(__file__).parent.parent / "data" / "json_outputs"
COLLECTION_NAME        = "publications"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

# Chunk types to embed and index
INDEXED_CHUNK_TYPES = {
    "overview",
    "introduction",
    "methods",
    "results",
    "discussion",
    "conclusion",
    "acknowledgements",
}

# Chunk types excluded from search results (indexed but not surfaced)
SEARCH_EXCLUDE_TYPES = {"acknowledgements"}


# ── Client setup ──────────────────────────────────────────────────────────────

def _get_embedding_fn():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in environment")
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name=OPENAI_EMBEDDING_MODEL,
    )


def _get_client() -> chromadb.PersistentClient:
    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))


def _get_or_create_collection(client, embedding_fn) -> chromadb.Collection:
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        return client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_fn,
        )
    return client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )


# ── Chunk loading ─────────────────────────────────────────────────────────────

def load_chunks_from_dir(json_dir: str | Path | None = None) -> list[dict]:
    """Load all __chunks.json files and return a flat list of all chunks."""
    json_dir    = Path(json_dir) if json_dir else JSON_OUTPUTS_DIR
    chunk_files = sorted(json_dir.glob("*__chunks.json"))

    if not chunk_files:
        raise FileNotFoundError(
            f"No __chunks.json files found in {json_dir}.\n"
            "Run the PDF pipeline first: python test_pdf_reader.py"
        )

    all_chunks = []
    for path in chunk_files:
        with open(path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        all_chunks.extend(chunks)
        print(f"  📄 {path.name}: {len(chunks)} chunks")

    print(f"  ✅ {len(all_chunks)} total chunks from {len(chunk_files)} paper(s)")
    return all_chunks


# Max words per embedding chunk — text-embedding-3-small limit is 8192 tokens
# Scientific text (gene names, drug names, abbreviations) tokenizes at ~1.87
# tokens/word, so safe ceiling = 8192 / 1.87 = 4380 words. Use 4000 for margin.
MAX_EMBEDDING_WORDS = 4000


def _split_content(content: str, max_words: int = MAX_EMBEDDING_WORDS) -> list[str]:
    """
    Split a long content string into overlapping sub-chunks for embedding.
    Splits on paragraph boundaries where possible.
    Returns a list of strings, each under max_words.
    """
    words = content.split()
    if len(words) <= max_words:
        return [content]

    sub_chunks = []
    overlap    = 100   # word overlap between chunks for context continuity
    start      = 0

    while start < len(words):
        end        = min(start + max_words, len(words))
        chunk_text = " ".join(words[start:end])

        # Try to split on a paragraph boundary near the end
        last_para = chunk_text.rfind("\n\n")
        if last_para > len(chunk_text) * 0.5:   # only if break is in latter half
            chunk_text = chunk_text[:last_para].strip()

        sub_chunks.append(chunk_text)
        if end >= len(words):
            break
        start = end - overlap   # overlap for continuity

    return sub_chunks


def _to_chroma_format(
    chunks: list[dict],
    existing_ids: set[str] | None = None,
) -> tuple[list, list, list]:
    """
    Convert chunk dicts to ChromaDB (ids, documents, metadatas) format.
    Filters by INDEXED_CHUNK_TYPES, skips empty content and duplicates.
    Automatically splits chunks that exceed MAX_EMBEDDING_WORDS.
    """
    ids, documents, metadatas = [], [], []
    skipped = {"type": 0, "empty": 0, "dup": 0}
    split_count = 0

    for chunk in chunks:
        chunk_type = chunk.get("chunk_type", "")
        content    = chunk.get("content", "").strip()
        chunk_id   = chunk.get("chunk_id", "")

        if chunk_type not in INDEXED_CHUNK_TYPES:
            skipped["type"] += 1
            continue
        if not content or chunk.get("word_count", 0) < 20:
            skipped["empty"] += 1
            continue
        if existing_ids and chunk_id in existing_ids:
            skipped["dup"] += 1
            continue

        base_meta = {
            "paper_id":    chunk.get("paper_id",    ""),
            "chunk_type":  chunk_type,
            "title":       chunk.get("title",       ""),
            "authors":     chunk.get("authors",     ""),
            "year":        str(chunk.get("year",    "")),
            "doi":         chunk.get("doi",         ""),
            "source_file": chunk.get("source_file", ""),
            "word_count":  str(chunk.get("word_count", 0)),
        }

        # Split oversized chunks before embedding
        sub_chunks = _split_content(content)
        if len(sub_chunks) > 1:
            split_count += len(sub_chunks)

        for i, sub_content in enumerate(sub_chunks):
            sub_id = f"{chunk_id}__p{i}" if len(sub_chunks) > 1 else chunk_id
            ids.append(sub_id)
            documents.append(sub_content)
            metadatas.append({**base_meta, "sub_chunk": str(i)})

    for reason, count in skipped.items():
        if count:
            print(f"  ⏭️  Skipped {count} ({reason})")
    if split_count:
        print(f"  ✂️  Split {split_count} oversized chunks (>{MAX_EMBEDDING_WORDS} words)")

    return ids, documents, metadatas


# ── Index builders ────────────────────────────────────────────────────────────

def build_index(
    json_dir: str | Path | None = None,
    force_rebuild: bool = False,
) -> chromadb.Collection:
    """
    Load all __chunks.json files from json_dir and embed into ChromaDB.
    Skips chunks already indexed (incremental by default).

    Args:
        json_dir      : directory with __chunks.json files (default: data/json_outputs/)
        force_rebuild : wipe and recreate collection from scratch

    Returns:
        The ChromaDB collection.
    """
    print(f"\n{'─'*50}")
    print("BUILDING VECTOR INDEX")
    print(f"{'─'*50}")

    client       = _get_client()
    embedding_fn = _get_embedding_fn()

    if force_rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"  🗑️  Deleted existing collection")
        except Exception:
            pass

    collection = _get_or_create_collection(client, embedding_fn)

    # Get existing IDs for incremental update
    existing_ids: set[str] = set()
    if not force_rebuild:
        try:
            result = collection.get(include=[])
            existing_ids = set(result["ids"])
            if existing_ids:
                print(f"  ℹ️  {len(existing_ids)} chunks already indexed")
        except Exception:
            pass

    print(f"\n  Loading from {json_dir or JSON_OUTPUTS_DIR}")
    chunks = load_chunks_from_dir(json_dir)

    ids, documents, metadatas = _to_chroma_format(chunks, existing_ids)

    if not ids:
        print(f"  ✅ Nothing new to index")
        return collection

    # Upsert in batches of 100
    for i in range(0, len(ids), 100):
        collection.upsert(
            ids       = ids[i:i+100],
            documents = documents[i:i+100],
            metadatas = metadatas[i:i+100],
        )

    print(f"\n  ✅ Indexed {len(ids)} new chunks into '{COLLECTION_NAME}'")
    print(f"  💾 Persisted to {CHROMA_PERSIST_DIR}")
    return collection


def index_chunks_file(chunks_file: str | Path) -> int:
    """
    Incrementally index a single __chunks.json file.
    Safe to call repeatedly — skips already-indexed chunks.

    Returns number of new chunks added.
    """
    chunks_file = Path(chunks_file)
    if not chunks_file.exists():
        raise FileNotFoundError(f"Not found: {chunks_file}")

    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    client       = _get_client()
    embedding_fn = _get_embedding_fn()
    collection   = _get_or_create_collection(client, embedding_fn)

    try:
        existing_ids = set(collection.get(include=[])["ids"])
    except Exception:
        existing_ids = set()

    ids, documents, metadatas = _to_chroma_format(chunks, existing_ids)

    if not ids:
        print(f"  ✅ {chunks_file.name}: already fully indexed")
        return 0

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    print(f"  ✅ {chunks_file.name}: added {len(ids)} chunks")
    return len(ids)


# ── Search ────────────────────────────────────────────────────────────────────

def search_publications(
    query: str,
    n_results: int = 5,
    chunk_types: list[str] | None = None,
    paper_id: str | None = None,
) -> list[dict]:
    """
    Semantic search over indexed publication chunks.

    Args:
        query       : natural language question
        n_results   : chunks to return (default 5)
        chunk_types : filter by type e.g. ["methods", "results"]
        paper_id    : filter to a specific paper

    Returns:
        List of dicts: content, paper_id, chunk_type, title, authors,
                       year, doi, distance
    """
    client       = _get_client()
    embedding_fn = _get_embedding_fn()

    try:
        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_fn,
        )
    except Exception:
        raise RuntimeError(
            "Vector store not initialized. Call init_vector_store() at startup."
        )

    # Build where-filter
    filters = []

    effective_types = chunk_types or [
        t for t in INDEXED_CHUNK_TYPES if t not in SEARCH_EXCLUDE_TYPES
    ]
    if len(effective_types) == 1:
        filters.append({"chunk_type": {"$eq": effective_types[0]}})
    elif effective_types:
        filters.append({"chunk_type": {"$in": list(effective_types)}})

    if paper_id:
        filters.append({"paper_id": {"$eq": paper_id}})

    where = {"$and": filters} if len(filters) > 1 else (filters[0] if filters else None)

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    return [
        {
            "content":    doc,
            "paper_id":   meta.get("paper_id",   ""),
            "chunk_type": meta.get("chunk_type", ""),
            "title":      meta.get("title",      ""),
            "authors":    meta.get("authors",    ""),
            "year":       meta.get("year",       ""),
            "doi":        meta.get("doi",        ""),
            "distance":   round(dist, 4),
        }
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


def get_collection_stats() -> dict:
    """Return counts of indexed papers and chunk types."""
    client = _get_client()
    try:
        col   = client.get_collection(name=COLLECTION_NAME)
        items = col.get(include=["metadatas"])
        metas = items["metadatas"]
        type_counts: dict[str, int] = {}
        for m in metas:
            ct = m["chunk_type"]
            type_counts[ct] = type_counts.get(ct, 0) + 1
        papers = {m["paper_id"] for m in metas}
        return {
            "total_chunks": len(items["ids"]),
            "paper_count":  len(papers),
            "papers":       sorted(papers),
            "chunk_types":  type_counts,
        }
    except Exception:
        return {"total_chunks": 0, "paper_count": 0, "papers": [], "chunk_types": {}}


# ── Context builder for LLM prompts ──────────────────────────────────────────

def get_publications_context(
    query: str,
    n_results: int = 5,
    chunk_types: list[str] | None = None,
    paper_id: str | None = None,
) -> str:
    """
    Retrieve relevant chunks and format them for injection into an LLM prompt.

    Returns a formatted string with numbered excerpts and metadata headers.
    """
    hits = search_publications(query, n_results=n_results,
                               chunk_types=chunk_types, paper_id=paper_id)

    if not hits:
        return "No relevant publications found for this query."

    lines = ["Relevant publication excerpts:\n"]
    for i, hit in enumerate(hits, 1):
        lines.append(
            f"[{i}] \"{hit['title']}\" ({hit['year']}) — "
            f"{hit['chunk_type'].upper()} | DOI: {hit['doi']}"
        )
        lines.append(f"    {hit['content'][:1200]}")
        lines.append("")

    return "\n".join(lines)



def get_all_publications() -> list[dict]:
    """
    Return one metadata record per unique paper from ChromaDB.
    Used for listing all publications without semantic search.
    Results are sorted by year descending.
    """
    client = _get_client()
    try:
        col   = client.get_collection(name=COLLECTION_NAME)
        items = col.get(include=["metadatas"])
        metas = items["metadatas"]

        # Deduplicate by paper_id — keep one record per paper
        seen = {}
        for m in metas:
            pid = m.get("paper_id", "")
            if pid and pid not in seen:
                seen[pid] = {
                    "paper_id": pid,
                    "title":    m.get("title",   ""),
                    "authors":  m.get("authors", ""),
                    "year":     m.get("year",    ""),
                    "doi":      m.get("doi",     ""),
                }

        # Sort by year descending
        return sorted(
            seen.values(),
            key=lambda x: x["year"],
            reverse=True,
        )
    except Exception as e:
        print(f"[get_all_publications] Error: {e}")
        return []

# ── Startup helper ────────────────────────────────────────────────────────────

def init_vector_store(force_rebuild: bool = False):
    """
    Ensure the vector index is ready. Call once at application startup.

    FastAPI example:
        from contextlib import asynccontextmanager
        from src.vector_store import init_vector_store

        @asynccontextmanager
        async def lifespan(app):
            init_vector_store()
            yield

        app = FastAPI(lifespan=lifespan)
    """
    build_index(force_rebuild=force_rebuild)
