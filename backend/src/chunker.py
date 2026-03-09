"""
src/chunker.py
─────────────────────────────────────────────────────────────────────────────
Assembles the final standardized chunked JSON from:
  - ExtractionResult  (pdf_reader.py)  — text, figures, metadata
  - ParseResult       (section_parser.py) — canonical sections

Output: one JSON file per paper in data/json_outputs/
Each file contains a list of chunk dicts, one per canonical section + figures.

Canonical chunk_type values (always present, empty string if not found):
  overview        abstract / implicit abstract
  introduction
  methods         all methods sub-sections merged
  results         all results sub-sections merged
  discussion
  conclusion      (often merged into discussion in some journals)
  acknowledgements
  references      (optional, skipped by default)
  figures         one chunk per extracted figure

Schema per chunk
────────────────
{
  "paper_id":    str,        # slug__year
  "chunk_type":  str,        # canonical section name
  "chunk_id":    str,        # paper_id__chunk_type__index
  "title":       str,        # paper title (from metadata, not parser heuristic)
  "authors":     str,        # from PDF metadata
  "year":        str,        # from PDF metadata
  "doi":         str,        # from PDF metadata
  "source_file": str,        # original PDF filename
  "content":     str,        # section body text
  "word_count":  int,
  "figure_id":   str | null, # only for figure chunks
  "image_path":  str | null, # only for figure chunks
  "caption":     str | null, # only for figure chunks
  "confidence":  str,        # "regex" | "implicit" | "inherited:X" | "figure"
}
"""

import os
import re
import json
from pathlib import Path
from dataclasses import dataclass, field

# Import our pipeline types
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.pdf_reader import ExtractionResult
from src.section_parser import ParseResult, DetectedSection


# ── Config ────────────────────────────────────────────────────────────────────

# Canonical sections to always include in output (even if empty)
CANONICAL_SECTIONS = [
    "overview",
    "introduction",
    "methods",
    "results",
    "discussion",
    "conclusion",
    "acknowledgements",
]

# Figure chunks use a lower word threshold — captions can be short
MIN_FIGURE_WORDS = 3

# Sections to skip by default (add "references" here to exclude them)
SKIP_SECTIONS = {"other"}

# Minimum words for a chunk to be included (filters empty/stub sections)
MIN_CHUNK_WORDS = 20

# Results section fix: these italic/bold inline result sub-headers in JCI
# end with a period and won't be caught as headers — we detect them as
# sentence-level anchors and treat the content before the next one as results
RESULTS_INLINE_SUBHEADER_RE = re.compile(
    r'^[A-Z][^.]{10,80}\.\s*$',   # Title-case sentence ending in period, 10-80 chars
)


# ── Title resolution ──────────────────────────────────────────────────────────

# Regex to detect non-title strings — journal IDs, page ranges, DOI fragments
_BAD_TITLE_RE = re.compile(
    r'^\s*[A-Z]+-\d+|'        # journal ID like MCT-22-0486
    r'\d+\.\.\d+|'            # page range like 1100..1111
    r'^10\.\d{4}|'            # DOI
    r'^[a-z0-9_-]+$',         # all lowercase slug
    re.IGNORECASE,
)


def _looks_like_title(text: str) -> bool:
    """Return True if text looks like a real paper title."""
    if not text or len(text.split()) < 4:
        return False
    if _BAD_TITLE_RE.search(text):
        return False
    return True


def _resolve_title(extraction: ExtractionResult, parse: ParseResult) -> str:
    """
    Pick the best available title, in priority order:
    1. PDF filename — always preferred since files are named with full titles
       e.g. "2023 Pan-cancer pharmacogenomic analysis of patient-derived..."
    2. PDF metadata title — only if it looks like a real title
    3. Parser-detected title
    4. First sentence of overview/introduction
    5. paper_id as fallback
    """
    # Option 1: PDF filename — unconditionally preferred
    # metadata["source"] is set by pdf_reader to Path(pdf_path).name
    source = extraction.metadata.get("source", "") or Path(extraction.pdf_path).name
    if source:
        name = Path(source).stem                          # strip .pdf
        name = re.sub(r'^\d{4}\s+', '', name).strip() # strip leading year
        if len(name.split()) >= 4:
            return name

    # Option 2: PDF metadata title — only if it doesn't look like an article ID
    meta_title = extraction.metadata.get("title", "").strip()
    meta_clean = re.sub(r'^\d{4}\s+', '', meta_title).strip()
    if _looks_like_title(meta_clean):
        return meta_clean

    # Option 3: parser-detected title
    parser_title = parse.title.strip()
    if _looks_like_title(parser_title) and len(parser_title.split()) <= 25:
        return parser_title

    # Option 4: first sentence of overview or introduction
    for section_key in ("overview", "introduction"):
        section_content = parse.sections.get(section_key, "")
        if section_content:
            first_sentence = section_content.split(".")[0].strip()
            if _looks_like_title(first_sentence) and len(first_sentence.split()) <= 30:
                return first_sentence

    return extraction.paper_id.replace("_", " ").title()


def _resolve_authors(extraction: ExtractionResult, full_text: str) -> str:
    """
    Get author list. PDF metadata is often empty for journals.
    Falls back to extracting the author line from the first page text.
    """
    meta_author = extraction.metadata.get("author", "").strip()
    if meta_author:
        return meta_author

    # Look for author line pattern in first 50 lines:
    # Typically: "First Last,1,2 First Last,3 ... and First Last1,2"
    lines = full_text.splitlines()[:50]
    for line in lines:
        stripped = line.strip()
        # Author lines typically have multiple comma-separated names with
        # superscript numbers, and contain " and " near the end
        if " and " in stripped and stripped.count(",") >= 2:
            # Check it looks like names not institutions
            if not any(word in stripped.lower() for word in
                       ["university", "institute", "department", "center",
                        "hospital", "school", "college", "laboratory"]):
                # Clean up superscript numbers
                cleaned = re.sub(r'\d+,?\d*', '', stripped)
                cleaned = re.sub(r'\s+', ' ', cleaned).strip().strip(",")
                if len(cleaned) > 10:
                    return cleaned

    return ""


# ── Results section repair ────────────────────────────────────────────────────

def _repair_results_section(
    parse: ParseResult,
    extraction: ExtractionResult,
) -> dict[str, str]:
    """
    JCI and similar journals embed results sub-headers as italic sentences
    ending in a period (e.g. "Transcriptomic analysis of ICB-resistant melanoma.")
    These don't get detected as headers, so their content bleeds into methods
    via the inheritance system.

    Strategy:
      - Find the Results header position in the raw text
      - Find the next CANONICAL header after it (Discussion, Methods, etc.)
      - Everything between those two positions = results content
      - Re-parse that block to extract inline sub-headers as sub-chunks

    Returns updated sections dict with corrected results content.
    """
    sections = dict(parse.sections)
    full_text = extraction.full_text

    # Find Results and next major header positions in full text
    results_match = re.search(r'^\s*Results\s*$', full_text, re.MULTILINE)
    if not results_match:
        return sections  # no fix needed

    results_start = results_match.end()

    # Find the next canonical section header after Results
    next_header_patterns = [
        r'^\s*Discussion\s*$',
        r'^\s*Methods?\s*$',
        r'^\s*Conclusion\s*$',
        r'^\s*Acknowledgements?\s*$',
        r'^\s*References?\s*$',
    ]
    results_end = len(full_text)
    for pat in next_header_patterns:
        m = re.search(pat, full_text[results_start:], re.MULTILINE)
        if m:
            candidate_end = results_start + m.start()
            if candidate_end < results_end:
                results_end = candidate_end

    results_body = full_text[results_start:results_end].strip()

    if results_body and len(results_body.split()) > MIN_CHUNK_WORDS:
        sections["results"] = results_body

    return sections


# ── Chunk builder ─────────────────────────────────────────────────────────────

def _make_chunk_id(paper_id: str, chunk_type: str, index: int) -> str:
    return f"{paper_id}__{chunk_type}__{index:03d}"


def build_chunks(
    extraction: ExtractionResult,
    parse: ParseResult,
    skip_sections: set[str] | None = None,
    include_references: bool = False,
    include_figures: bool = True,
) -> list[dict]:
    """
    Build the final list of chunk dicts from extraction + parse results.

    Args:
        extraction        : ExtractionResult from pdf_reader
        parse             : ParseResult from section_parser
        skip_sections     : set of canonical names to exclude
        include_references: whether to include a references chunk
        include_figures   : whether to include figure chunks

    Returns:
        List of chunk dicts ready for JSON serialisation.
    """
    if skip_sections is None:
        skip_sections = SKIP_SECTIONS.copy()
    if not include_references:
        skip_sections.add("references")

    # ── Resolve metadata ──────────────────────────────────────────────────────
    title   = _resolve_title(extraction, parse)
    authors = _resolve_authors(extraction, extraction.full_text)
    year    = extraction.metadata.get("year", "")
    doi     = extraction.metadata.get("doi", "")
    source  = extraction.metadata.get("source", "")
    paper_id = extraction.paper_id

    # ── Repair results section ────────────────────────────────────────────────
    sections = _repair_results_section(parse, extraction)

    # ── Build section chunks ──────────────────────────────────────────────────
    chunks = []
    section_index = 0

    for chunk_type in CANONICAL_SECTIONS:
        if chunk_type in skip_sections:
            continue

        content = sections.get(chunk_type, "").strip()
        word_count = len(content.split()) if content else 0

        # Always emit a chunk for core sections, even if empty (makes
        # downstream processing simpler — just check word_count == 0)
        if word_count < MIN_CHUNK_WORDS and chunk_type not in (
            "overview", "introduction", "methods", "results", "discussion"
        ):
            continue

        # Find confidence for this section from the detected list
        confidence = "unknown"
        for sec in parse.all_detected:
            if sec.canonical == chunk_type and sec.content.strip():
                confidence = sec.confidence
                break

        chunks.append({
            "paper_id":    paper_id,
            "chunk_type":  chunk_type,
            "chunk_id":    _make_chunk_id(paper_id, chunk_type, section_index),
            "title":       title,
            "authors":     authors,
            "year":        year,
            "doi":         doi,
            "source_file": source,
            "content":     content,
            "word_count":  word_count,
            "figure_id":   None,
            "image_path":  None,
            "caption":     None,
            "confidence":  confidence,
        })
        section_index += 1

    # ── Figure chunks ─────────────────────────────────────────────────────────
    if include_figures:
        for fig in extraction.figures:
            caption = fig.caption or ""
            content = caption if caption else f"[Figure {fig.figure_id} — no caption detected]"
            chunks.append({
                "paper_id":    paper_id,
                "chunk_type":  "figure",
                "chunk_id":    _make_chunk_id(paper_id, "figure", section_index),
                "title":       title,
                "authors":     authors,
                "year":        year,
                "doi":         doi,
                "source_file": source,
                "content":     content,
                "word_count":  len(content.split()),
                "figure_id":   fig.figure_id,
                "image_path":  fig.image_path,
                "caption":     caption,
                "confidence":  "figure",
            })
            section_index += 1

    return chunks


# ── Save ──────────────────────────────────────────────────────────────────────

def save_chunks(
    chunks: list[dict],
    paper_id: str,
    output_dir: str = "data/json_outputs",
) -> str:
    """Save chunks to data/json_outputs/<paper_id>__chunks.json"""
    os.makedirs(output_dir, exist_ok=True)
    filename  = f"{paper_id}__chunks.json"
    file_path = os.path.join(output_dir, filename)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"  💾 Chunks saved → {file_path}")
    return file_path


# ── Pretty printer ────────────────────────────────────────────────────────────

def print_chunk_report(chunks: list[dict]):
    """Print a human-readable summary of the chunk output."""
    print(f"\n{'─'*60}")
    print("CHUNK REPORT")
    print(f"{'─'*60}")

    if not chunks:
        print("  ❌ No chunks produced")
        return

    meta = chunks[0]
    print(f"  paper_id  : {meta['paper_id']}")
    print(f"  title     : {meta['title'][:80]}")
    print(f"  authors   : {meta['authors'][:65] or '—'}")
    print(f"  year      : {meta['year']}")
    print(f"  doi       : {meta['doi']}")
    print(f"  chunks    : {len(chunks)}")

    print(f"\n  {'CHUNK_TYPE':<18} {'WORDS':>6}  PREVIEW")
    print(f"  {'─'*18} {'─'*6}  {'─'*40}")
    for chunk in chunks:
        ct      = chunk["chunk_type"]
        words   = chunk["word_count"]
        preview = chunk["content"][:60].replace("\n", " ")
        fig_tag = f" [{chunk['figure_id']}]" if chunk.get("figure_id") else ""
        empty   = "  ⚠️  EMPTY" if words < MIN_CHUNK_WORDS else ""
        print(f"  {ct:<18} {words:>6}  {preview}…{fig_tag}{empty}")


# ── Pipeline entry point ──────────────────────────────────────────────────────

def chunk_paper(
    extraction: ExtractionResult,
    parse: ParseResult,
    output_dir: str = "data/json_outputs",
    include_references: bool = False,
    include_figures: bool = True,
) -> list[dict]:
    """
    Full chunking pipeline for one paper.
    Builds chunks, prints report, saves JSON.

    Returns the list of chunks.
    """
    chunks = build_chunks(
        extraction,
        parse,
        include_references=include_references,
        include_figures=include_figures,
    )
    print_chunk_report(chunks)
    save_chunks(chunks, extraction.paper_id, output_dir)
    return chunks
