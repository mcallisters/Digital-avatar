"""
scripts/ingest_papers.py
─────────────────────────────────────────────────────────────────────────────
Batch ingest script — runs the full pipeline on every PDF in data/pdfs/
and loads all resulting chunks into ChromaDB.

Usage
─────
  # Ingest all new PDFs (skips already-processed ones):
  python scripts/ingest_papers.py

  # Force re-process a specific paper:
  python scripts/ingest_papers.py --paper "my_paper.pdf"

  # Rebuild the entire vector index from scratch:
  python scripts/ingest_papers.py --rebuild-index

  # Dry run — show what would be processed without doing it:
  python scripts/ingest_papers.py --dry-run

Pipeline per PDF
────────────────
  1. pdf_reader.py     — extract text + figures
  2. section_parser.py — detect and normalize sections
  3. chunker.py        — assemble JSON chunks
  4. vector_store.py   — embed and index into ChromaDB
"""

import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pdf_reader import extract_pdf, save_extracted_text, save_extraction_manifest
from src.section_parser import parse_sections
from src.chunker import chunk_paper
from src.vector_store import index_chunks_file, build_index, get_collection_stats

# ── Directories ───────────────────────────────────────────────────────────────

PDF_DIR      = Path("data/pdfs")
EXTRACT_DIR  = Path("data/extracted_texts")
FIGURES_DIR  = Path("data/figures")
JSON_DIR     = Path("data/json_outputs")


def get_paper_id_from_manifest(pdf_path: Path) -> str | None:
    """Check if a PDF has already been processed by looking for its manifest."""
    # We don't know the paper_id yet without processing — scan manifests
    for manifest_path in EXTRACT_DIR.glob("*__manifest.json"):
        import json
        with open(manifest_path) as f:
            manifest = json.load(f)
        if manifest.get("source_file") == pdf_path.name:
            return manifest.get("paper_id")
    return None


def get_chunks_path(paper_id: str) -> Path:
    return JSON_DIR / f"{paper_id}__chunks.json"


def process_pdf(
    pdf_path: Path,
    openai_client: OpenAI,
    force: bool = False,
    verbose: bool = True,
) -> str | None:
    """
    Run the full extraction → parse → chunk pipeline on one PDF.
    Returns the paper_id, or None if processing failed.

    Skips if chunks JSON already exists (unless force=True).
    """
    def log(msg): print(msg) if verbose else None

    log(f"\n{'─'*60}")
    log(f"  📄 {pdf_path.name}")
    log(f"{'─'*60}")

    # ── Check if already processed ────────────────────────────────────────────
    existing_id = get_paper_id_from_manifest(pdf_path)
    if existing_id and get_chunks_path(existing_id).exists() and not force:
        log(f"  ✅ Already processed → {existing_id}")
        log(f"     (use --force to reprocess)")
        return existing_id

    try:
        # ── Step 1: Extract ───────────────────────────────────────────────────
        log(f"  🔍 Extracting text and figures...")
        result = extract_pdf(
            str(pdf_path),
            figures_dir   = str(FIGURES_DIR),
            openai_client = openai_client,
        )
        log(f"     {result.page_count} pages | {result.word_count} words | "
            f"{len(result.figures)} figures | quality={result.quality_score:.2f}")

        save_extracted_text(result, str(EXTRACT_DIR))
        save_extraction_manifest(result, str(EXTRACT_DIR))

        # ── Step 2: Parse sections ────────────────────────────────────────────
        log(f"  🔍 Parsing sections...")
        parse_result = parse_sections(
            full_text = result.full_text,
            paper_id  = result.paper_id,
        )
        canonical_keys = parse_result.canonical_keys if hasattr(parse_result, 'canonical_keys') else list(parse_result.sections.keys())
        log(f"     sections: {canonical_keys}")

        # ── Step 3: Chunk ─────────────────────────────────────────────────────
        log(f"  🔍 Chunking...")
        chunks = chunk_paper(
            extraction         = result,
            parse              = parse_result,
            output_dir         = str(JSON_DIR),
            include_references = False,
            include_figures    = True,
        )
        log(f"     {len(chunks)} chunks → {get_chunks_path(result.paper_id).name}")

        return result.paper_id

    except Exception as e:
        print(f"  ❌ Failed: {pdf_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def ingest_to_vector_store(paper_ids: list[str], verbose: bool = True):
    """Index all chunk files for the given paper IDs into ChromaDB."""
    def log(msg): print(msg) if verbose else None

    log(f"\n{'─'*60}")
    log("INDEXING INTO CHROMADB")
    log(f"{'─'*60}")

    total_new = 0
    for paper_id in paper_ids:
        chunks_path = get_chunks_path(paper_id)
        if not chunks_path.exists():
            log(f"  ⚠️  Chunks file not found: {chunks_path}")
            continue
        n = index_chunks_file(chunks_path)
        total_new += n

    log(f"\n  ✅ Done — {total_new} new chunks added to vector store")


def print_stats():
    """Print current collection stats."""
    stats = get_collection_stats()
    print(f"\n{'─'*60}")
    print("VECTOR STORE STATUS")
    print(f"{'─'*60}")
    print(f"  papers indexed : {stats['paper_count']}")
    print(f"  total chunks   : {stats['total_chunks']}")
    if stats['chunk_types']:
        print(f"  chunk types    :")
        for ct, count in sorted(stats['chunk_types'].items()):
            print(f"    {ct:<18} {count}")
    if stats['papers']:
        print(f"  papers:")
        for p in stats['papers']:
            print(f"    {p}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ingest PDFs through the full pipeline into ChromaDB"
    )
    parser.add_argument(
        "--paper",
        help="Process a specific PDF filename only (e.g. 'my_paper.pdf')",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process PDFs even if already extracted",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Wipe and rebuild the ChromaDB index from all existing chunk files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without doing anything",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print current vector store stats and exit",
    )
    args = parser.parse_args()

    # Stats only
    if args.stats:
        print_stats()
        return

    # Rebuild index from existing chunks only (no PDF processing)
    if args.rebuild_index:
        print("Rebuilding vector index from all existing chunk files...")
        build_index(force_rebuild=True)
        print_stats()
        return

    # Collect PDFs to process
    if args.paper:
        pdf_files = [PDF_DIR / args.paper]
        if not pdf_files[0].exists():
            print(f"❌ PDF not found: {pdf_files[0]}")
            sys.exit(1)
    else:
        pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"❌ No PDFs found in {PDF_DIR}/")
        sys.exit(1)

    print(f"\n{'═'*60}")
    print(f"  INGEST PIPELINE")
    print(f"{'═'*60}")
    print(f"  PDFs to process : {len(pdf_files)}")
    print(f"  Force reprocess : {args.force}")
    print(f"  Dry run         : {args.dry_run}")

    if args.dry_run:
        print(f"\n  Would process:")
        for f in pdf_files:
            existing_id = get_paper_id_from_manifest(f)
            status = "skip (already processed)" if (
                existing_id and get_chunks_path(existing_id).exists() and not args.force
            ) else "process"
            print(f"    [{status}] {f.name}")
        return

    # Initialize OpenAI client
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Process each PDF
    successful_ids = []
    for pdf_path in pdf_files:
        paper_id = process_pdf(pdf_path, openai_client, force=args.force)
        if paper_id:
            successful_ids.append(paper_id)

    print(f"\n  Processed {len(successful_ids)}/{len(pdf_files)} papers successfully")

    # Index into ChromaDB
    if successful_ids:
        ingest_to_vector_store(successful_ids)

    print_stats()


if __name__ == "__main__":
    main()
