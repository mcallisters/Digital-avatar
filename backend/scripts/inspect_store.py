"""
scripts/inspect_store.py
─────────────────────────────────────────────────────────────────────────────
Human-readable inspection of the ChromaDB vector store.

Usage
─────
  # Summary overview
  python scripts/inspect_store.py

  # Show full content of every chunk
  python scripts/inspect_store.py --full

  # Filter by chunk type
  python scripts/inspect_store.py --type methods

  # Filter by paper
  python scripts/inspect_store.py --paper "combinatorial_therapy"

  # Test a semantic search query
  python scripts/inspect_store.py --query "cobimetinib regorafenib synergy"

  # Export everything to a readable text file
  python scripts/inspect_store.py --export
"""

import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vector_store import (
    _get_client,
    _get_embedding_fn,
    COLLECTION_NAME,
    get_collection_stats,
    search_publications,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_collection():
    client       = _get_client()
    embedding_fn = _get_embedding_fn()
    try:
        return client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_fn,
        )
    except Exception:
        print(f"❌ Collection '{COLLECTION_NAME}' not found.")
        print("   Run: python scripts/ingest_papers.py")
        sys.exit(1)


def print_separator(char="─", width=60):
    print(char * width)


def truncate(text: str, max_chars: int = 120) -> str:
    text = text.replace("\n", " ").strip()
    return text[:max_chars] + "…" if len(text) > max_chars else text


# ── Views ─────────────────────────────────────────────────────────────────────

def show_summary():
    """Print a high-level overview of the store."""
    stats = get_collection_stats()

    print_separator("═")
    print("CHROMADB STORE SUMMARY")
    print_separator("═")
    print(f"  Location       : data/chroma_db/")
    print(f"  Collection     : {COLLECTION_NAME}")
    print(f"  Papers indexed : {stats['paper_count']}")
    print(f"  Total chunks   : {stats['total_chunks']}")

    if stats["chunk_types"]:
        print(f"\n  CHUNK TYPES:")
        for ct, count in sorted(stats["chunk_types"].items()):
            bar = "█" * count
            print(f"    {ct:<18} {count:>3}  {bar}")

    if stats["papers"]:
        print(f"\n  PAPERS:")
        for p in stats["papers"]:
            print(f"    {p}")

    print()


def show_chunks(
    chunk_type: str | None = None,
    paper_filter: str | None = None,
    full_content: bool = False,
):
    """List all chunks with optional filters."""
    collection = get_collection()
    results    = collection.get(include=["documents", "metadatas"])

    items = list(zip(results["ids"], results["documents"], results["metadatas"]))

    # Apply filters
    if chunk_type:
        items = [i for i in items if i[2].get("chunk_type") == chunk_type]
    if paper_filter:
        items = [i for i in items if paper_filter.lower() in i[2].get("paper_id", "").lower()]

    if not items:
        print(f"  No chunks found matching filters.")
        return

    print_separator("═")
    print(f"CHUNKS ({len(items)} found)")
    if chunk_type:    print(f"  Filter: chunk_type = {chunk_type}")
    if paper_filter:  print(f"  Filter: paper      = {paper_filter}")
    print_separator("═")

    # Group by paper for readability
    by_paper: dict[str, list] = {}
    for id_, doc, meta in items:
        pid = meta.get("paper_id", "unknown")
        by_paper.setdefault(pid, []).append((id_, doc, meta))

    for paper_id, paper_items in sorted(by_paper.items()):
        first_meta = paper_items[0][2]
        print(f"\n  📄 {first_meta.get('title', paper_id)[:65]}")
        print(f"     {paper_id}  |  {first_meta.get('year','')}  |  DOI: {first_meta.get('doi','—')}")
        print_separator()

        for id_, doc, meta in sorted(paper_items, key=lambda x: x[2].get("chunk_type","")):
            ct       = meta.get("chunk_type", "—")
            words    = meta.get("word_count", "—")
            sub      = meta.get("sub_chunk", "0")
            sub_tag  = f" [part {int(sub)+1}]" if sub != "0" else ""
            print(f"\n  [{ct}{sub_tag}]  {words} words")
            print(f"  id: {id_}")

            if full_content:
                print(f"\n  {doc}\n")
            else:
                print(f"  {truncate(doc, 160)}")

    print()


def show_search(query: str, n_results: int = 5):
    """Run a semantic search and display results."""
    print_separator("═")
    print(f"SEMANTIC SEARCH")
    print_separator("═")
    print(f"  Query      : {query}")
    print(f"  Top results: {n_results}")
    print_separator()

    hits = search_publications(query, n_results=n_results)

    if not hits:
        print("  No results found.")
        return

    for i, hit in enumerate(hits, 1):
        print(f"\n  [{i}] distance = {hit['distance']:.4f}  "
              f"({'very relevant' if hit['distance'] < 0.3 else 'relevant' if hit['distance'] < 0.5 else 'loosely related'})")
        print(f"      type    : {hit['chunk_type']}")
        print(f"      paper   : {hit['title'][:60]}")
        print(f"      doi     : {hit['doi']}")
        print(f"      content : {truncate(hit['content'], 200)}")

    print()


def export_to_file(output_path: str = "data/chroma_db_export.txt"):
    """Export all chunks to a readable text file."""
    collection = get_collection()
    results    = collection.get(include=["documents", "metadatas"])
    items      = list(zip(results["ids"], results["documents"], results["metadatas"]))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"ChromaDB Export — {COLLECTION_NAME}\n")
        f.write(f"Total chunks: {len(items)}\n")
        f.write("=" * 80 + "\n\n")

        # Group by paper
        by_paper: dict[str, list] = {}
        for id_, doc, meta in items:
            pid = meta.get("paper_id", "unknown")
            by_paper.setdefault(pid, []).append((id_, doc, meta))

        for paper_id, paper_items in sorted(by_paper.items()):
            first_meta = paper_items[0][2]
            f.write(f"PAPER: {first_meta.get('title', paper_id)}\n")
            f.write(f"ID   : {paper_id}\n")
            f.write(f"Year : {first_meta.get('year','')}\n")
            f.write(f"DOI  : {first_meta.get('doi','')}\n")
            f.write(f"Authors: {first_meta.get('authors','')}\n")
            f.write("-" * 80 + "\n\n")

            for id_, doc, meta in sorted(paper_items, key=lambda x: x[2].get("chunk_type","")):
                ct    = meta.get("chunk_type", "—")
                words = meta.get("word_count", "—")
                sub   = meta.get("sub_chunk", "0")
                sub_tag = f" [part {int(sub)+1}]" if sub != "0" else ""
                f.write(f"[{ct.upper()}{sub_tag}]  {words} words\n")
                f.write(f"chunk_id: {id_}\n\n")
                f.write(doc)
                f.write("\n\n" + "─" * 60 + "\n\n")

    print(f"  ✅ Exported {len(items)} chunks to {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Inspect ChromaDB vector store contents"
    )
    parser.add_argument("--full",   action="store_true",
                        help="Show full chunk content (default: truncated preview)")
    parser.add_argument("--type",   metavar="CHUNK_TYPE",
                        help="Filter by chunk type e.g. methods, results, overview")
    parser.add_argument("--paper",  metavar="PAPER_ID",
                        help="Filter by paper ID (partial match)")
    parser.add_argument("--query",  metavar="QUERY",
                        help="Run a semantic search query")
    parser.add_argument("--n",      type=int, default=5,
                        help="Number of search results (default: 5)")
    parser.add_argument("--export", action="store_true",
                        help="Export all chunks to data/chroma_db_export.txt")
    args = parser.parse_args()

    if args.query:
        show_search(args.query, n_results=args.n)
    elif args.export:
        export_to_file()
    elif args.type or args.paper or args.full:
        show_chunks(
            chunk_type    = args.type,
            paper_filter  = args.paper,
            full_content  = args.full,
        )
    else:
        show_summary()
        show_chunks()


if __name__ == "__main__":
    main()
