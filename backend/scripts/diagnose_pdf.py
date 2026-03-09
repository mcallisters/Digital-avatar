"""
scripts/diagnose_pdf.py
─────────────────────────────────────────────────────────────────────────────
Diagnostic tool — shows exactly what PyMuPDF sees in each page of a PDF.
Useful for debugging figure extraction issues.

Usage
─────
  python scripts/diagnose_pdf.py "data/pdfs/your_paper.pdf"
"""

import sys
from pathlib import Path
import fitz


def diagnose(pdf_path: str):
    doc = fitz.open(pdf_path)

    print(f"\n{'═'*60}")
    print(f"PDF DIAGNOSTIC: {Path(pdf_path).name}")
    print(f"{'═'*60}")
    print(f"  Pages : {len(doc)}")
    print(f"  Title : {doc.metadata.get('title', 'n/a')[:60]}")

    for page_num, page in enumerate(doc):
        page_number = page_num + 1
        images      = page.get_images(full=True)
        words       = len(page.get_text("text").split())
        page_area   = page.rect.width * page.rect.height

        if not images:
            continue

        print(f"\n  {'─'*56}")
        print(f"  PAGE {page_number}  |  {len(images)} image(s)  |  {words} words")
        print(f"  Page size: {page.rect.width:.0f} x {page.rect.height:.0f} pt")

        for i, img_info in enumerate(images):
            xref = img_info[0]
            try:
                info = doc.extract_image(xref)
                w, h = info["width"], info["height"]

                # Get position on page
                try:
                    rects = page.get_image_rects(xref)
                    if rects:
                        r = rects[0]
                        img_area   = r.width * r.height
                        pct_page   = (img_area / page_area) * 100
                        pos = f"pos=({r.x0:.0f},{r.y0:.0f}) size=({r.width:.0f}x{r.height:.0f}) {pct_page:.0f}% of page"
                    else:
                        pos = "position unknown"
                except Exception:
                    pos = "position unknown"

                print(f"    [{i+1}] {w}x{h}px  {pos}")
            except Exception as e:
                print(f"    [{i+1}] could not extract: {e}")

        # Show text blocks on the page (first 5)
        blocks = page.get_text("blocks")
        text_blocks = [b for b in blocks if b[6] == 0]  # type 0 = text
        if text_blocks:
            print(f"  Text blocks ({len(text_blocks)} total, showing first 5):")
            for b in text_blocks[:5]:
                preview = b[4].strip().replace("\n", " ")[:80]
                print(f"    \"{preview}\"")

    print(f"\n{'═'*60}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default to the glioma paper if no argument given
        pdf_path = "data/pdfs/2023 Therapeutic targeting of prenatal pontine ID1 signaling in diffuse midline glioma.pdf"
    else:
        pdf_path = sys.argv[1]

    if not Path(pdf_path).exists():
        print(f"❌ File not found: {pdf_path}")
        sys.exit(1)

    diagnose(pdf_path)
