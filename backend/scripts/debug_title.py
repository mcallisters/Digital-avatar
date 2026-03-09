"""
scripts/debug_title.py
─────────────────────────────────────────────────────────────────────────────
Diagnose why a paper's title is not resolving correctly.
Shows exactly what values _resolve_title sees at runtime.

Usage
─────
  python scripts/debug_title.py
  python scripts/debug_title.py "data/pdfs/your_paper.pdf"
"""

import sys
import os
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

from src.pdf_reader import extract_pdf

# ── Default to the pan-cancer paper ──────────────────────────────────────────
if len(sys.argv) > 1:
    pdf_path = sys.argv[1]
else:
    pdf_path = (
        "data/pdfs/2023 Pan-Cancer Pharmacogenomic Analysis of Patient-Derived "
        "Tumor Cells Using Clinically Relevant Drug Exposures.pdf"
    )

if not Path(pdf_path).exists():
    print(f"❌ File not found: {pdf_path}")
    sys.exit(1)

print(f"\n{'═'*60}")
print(f"TITLE DEBUG: {Path(pdf_path).name[:60]}")
print(f"{'═'*60}")

# Run extraction (no vision API, but figures_dir must be a real path)
result = extract_pdf(pdf_path, figures_dir="data/figures", openai_client=None)

print(f"\n  ExtractionResult fields:")
print(f"    pdf_path : {result.pdf_path}")
print(f"    paper_id : {result.paper_id}")

print(f"\n  metadata dict:")
for k, v in result.metadata.items():
    print(f"    {k:<12}: {str(v)[:80]}")

# Simulate _resolve_title logic step by step
print(f"\n  _resolve_title simulation:")

# Option 1: filename
source = result.metadata.get("source", "") or Path(result.pdf_path).name
print(f"\n  Option 1 — filename:")
print(f"    source      : {source}")
if source:
    name = Path(source).stem
    name = re.sub(r'^\d{4}\s+', '', name).strip()
    print(f"    after clean : {name}")
    print(f"    word count  : {len(name.split())}")
    if len(name.split()) >= 4:
        print(f"    ✅ WOULD USE: {name}")
    else:
        print(f"    ❌ Too short, falling through")
else:
    print(f"    ❌ source is empty")

# Option 2: metadata title
meta_title = result.metadata.get("title", "").strip()
meta_clean = re.sub(r'^\d{4}\s+', '', meta_title).strip()
print(f"\n  Option 2 — metadata title:")
print(f"    raw         : {meta_title}")
print(f"    cleaned     : {meta_clean}")

print(f"\n{'═'*60}\n")
