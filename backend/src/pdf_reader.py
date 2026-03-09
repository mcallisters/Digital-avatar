"""
src/pdf_reader.py
─────────────────────────────────────────────────────────────────────────────
Extracts text and figures from academic PDF files.

Primary engine : PyMuPDF (fitz)  — fast, handles modern PDFs well
Fallback engine: pdfplumber      — better layout recovery for multi-column
                                   or heavily formatted pages

Outputs
───────
ExtractionResult dataclass containing:
  • full_text        : str  — complete document text, pages joined
  • pages            : list[PageResult] — per-page text + metadata
  • figures          : list[FigureResult] — extracted images + captions
  • metadata         : dict — title, authors, doi, year from PDF metadata
  • extraction_method: str  — "pymupdf" | "pdfplumber" | "mixed"
  • quality_score    : float 0–1  — rough estimate of extraction quality
"""

import os
import re
import json
import fitz          # PyMuPDF
import pdfplumber
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
from PIL import Image
import io


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class PageResult:
    page_number: int          # 1-based
    text: str
    word_count: int
    has_figures: bool
    extraction_method: str    # "pymupdf" | "pdfplumber"


@dataclass
class FigureResult:
    figure_id: str            # e.g. "fig_p3_0" — page 3, index 0
    page_number: int          # 1-based
    image_path: str           # saved PNG path
    caption: str              # detected caption text (may be empty)
    width_px: int
    height_px: int


@dataclass
class ExtractionResult:
    pdf_path: str
    paper_id: str
    full_text: str
    pages: list[PageResult] = field(default_factory=list)
    figures: list[FigureResult] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    extraction_method: str = "pymupdf"
    quality_score: float = 0.0
    page_count: int = 0
    word_count: int = 0


# ── Config ────────────────────────────────────────────────────────────────────

# Minimum average words-per-page before we try pdfplumber as fallback
MIN_WORDS_PER_PAGE = 50

# Minimum image size to bother saving (filters out logos, decorations, rules)
MIN_IMAGE_WIDTH  = 100
MIN_IMAGE_HEIGHT = 100

# Matches ONLY main figure captions e.g. "Figure 1.", "Figure 2A —"
# Explicitly excludes Supplemental / Extended Data figures
MAIN_FIGURE_RE = re.compile(
    r"^Figure\s+(\d+)[A-Za-z]?(?:[.\-\s–]|$)",
    re.IGNORECASE,
)


# ── Metadata extraction ───────────────────────────────────────────────────────

def _extract_metadata(doc: fitz.Document, pdf_path: str) -> dict:
    """
    Pull title, author, subject, doi from PDF metadata dict.
    Falls back to filename-derived values when fields are empty.
    """
    raw = doc.metadata or {}

    title  = (raw.get("title")  or "").strip()
    author = (raw.get("author") or "").strip()

    # Try to find a DOI in the first two pages of text
    doi = ""
    doi_pattern = re.compile(r"10\.\d{4,9}/[^\s]+")
    for page_num in range(min(2, len(doc))):
        page_text = doc[page_num].get_text()
        m = doi_pattern.search(page_text)
        if m:
            doi = m.group(0).rstrip(".,;)")
            break

    # Year: try CreationDate first, then search first page
    year = ""
    creation = raw.get("creationDate", "")
    year_match = re.search(r"(19|20)\d{2}", creation)
    if year_match:
        year = year_match.group(0)
    else:
        first_page_text = doc[0].get_text() if len(doc) > 0 else ""
        ym = re.search(r"\b(19|20)\d{2}\b", first_page_text)
        if ym:
            year = ym.group(0)

    # Fallback title from filename — strip year prefix and use as-is
    # since files are named "2023 Full Paper Title Here.pdf"
    if not title:
        stem  = Path(pdf_path).stem                        # "2023 Pan-cancer pharmacogenomic..."
        clean = re.sub(r'^\d{4}\s+', '', stem).strip()   # "Pan-cancer pharmacogenomic..."
        title = clean if len(clean.split()) >= 4 else stem.replace("_", " ").replace("-", " ").title()

    # Even if metadata title exists, prefer the filename when it looks like
    # a journal article ID (short, contains only numbers/hyphens/underscores)
    elif re.match(r'^[a-z0-9_\-]+$', Path(pdf_path).stem, re.I):
        stem  = Path(pdf_path).stem
        clean = re.sub(r'^\d{4}\s+', '', stem).strip()
        if len(clean.split()) >= 4:
            title = clean

    return {
        "title":  title,
        "author": author,
        "doi":    doi,
        "year":   year,
        "source": Path(pdf_path).name,
    }


# ── paper_id generator ────────────────────────────────────────────────────────

def _make_paper_id(metadata: dict, pdf_path: str) -> str:
    """
    Derives a filesystem-safe paper_id.
    Format: firstword_of_title__year   e.g. "transformer_clinical_nlp__2023"
    Falls back to PDF filename stem.
    """
    title = metadata.get("title", "")
    year  = metadata.get("year",  "")

    if title:
        slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
        slug = slug[:50]
    else:
        slug = Path(pdf_path).stem.lower()
        slug = re.sub(r"[^a-z0-9]+", "_", slug).strip("_")

    if year:
        return f"{slug}__{year}"
    return slug


# ── Quality scoring ───────────────────────────────────────────────────────────

def _quality_score(pages: list[PageResult], total_pages: int) -> float:
    """
    Heuristic quality score 0.0–1.0.

    Factors:
      - average words per page (academic papers: 250–400)
      - fraction of pages with at least some text
      - penalise if many pages look like scans (very low word count)
    """
    if not pages:
        return 0.0

    word_counts  = [p.word_count for p in pages]
    avg_words    = sum(word_counts) / len(word_counts)
    frac_ok      = sum(1 for w in word_counts if w >= 20) / len(word_counts)

    # Score avg_words: 0 at 0, 1.0 at 300+
    words_score = min(avg_words / 300.0, 1.0)

    return round((words_score * 0.6) + (frac_ok * 0.4), 3)


# ── Figure extraction ─────────────────────────────────────────────────────────

def _extract_figure_number(caption: str) -> str | None:
    """Extract the integer figure number from a caption string."""
    m = MAIN_FIGURE_RE.match(caption.strip())
    return m.group(1) if m else None


def _detect_caption_on_page(page: fitz.Page, bbox: fitz.Rect) -> str:
    """
    Search for a main figure caption on the page using three passes:

    Pass 1: Narrow band below the image (most common — caption beneath figure)
    Pass 2: Narrow band above the image (JCI style — caption above figure)
    Pass 3: Full page scan — catches right-justified captions embedded within
            the figure boundary or anywhere else on the page

    Returns the full matched caption text, or empty string if not found.
    Only returns captions matching MAIN_FIGURE_RE (no supplemental figures).
    """
    # Pass 1 & 2: progressively wider bands above and below
    for margin in (150, 350):
        for search_rect in (
            fitz.Rect(bbox.x0 - 20, bbox.y1,        bbox.x1 + 20, bbox.y1 + margin),
            fitz.Rect(bbox.x0 - 20, bbox.y0 - margin, bbox.x1 + 20, bbox.y0),
        ):
            for block in page.get_text("blocks", clip=search_rect):
                text = block[4].strip()
                if MAIN_FIGURE_RE.match(text):
                    return re.sub(r"\s+", " ", text)

    # Pass 3: full page scan — right-justified or embedded captions
    for block in page.get_text("blocks"):
        text = block[4].strip()
        if MAIN_FIGURE_RE.match(text):
            return re.sub(r"\s+", " ", text)

    return ""


def _vision_caption(image_path: str, figure_id: str, openai_client) -> str:
    """
    Use GPT-4o vision to extract the figure caption from an image.
    Called only when PyMuPDF text extraction found no caption.

    Returns the extracted caption string, or empty string on failure.
    """
    import base64

    try:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64}",
                            "detail": "high",
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "This is a figure from a scientific publication. "
                            "Extract the complete figure legend/caption text exactly as written. "
                            "The caption usually starts with 'Figure N.' or 'Fig. N.' "
                            "and may be positioned below, above, or to the right of the figure panels. "
                            "Return ONLY the caption text, nothing else. "
                            "If no caption is visible, return the single word: NONE"
                        ),
                    },
                ],
            }],
        )

        result = response.choices[0].message.content.strip()
        if result.upper() == "NONE" or not result:
            return ""
        return result

    except Exception as e:
        print(f"  ⚠️  Vision API failed for {figure_id}: {e}")
        return ""



def _is_composite_figure_page(page: fitz.Page) -> bool:
    """
    Detect pages that are predominantly figure content and likely need
    full-page vision analysis to extract captions.

    Covers several layouts seen in journals:
      1. Many small panel images (composite figure with individual panel xrefs)
      2. One large image covering >35% of page area
      3. Any image-containing page with low word count (<350 words)
         — catches figure pages where caption is embedded in the image frame
    """
    image_list = page.get_images(full=True)
    n_images   = len(image_list)
    word_count = len(page.get_text("text").split())
    page_area  = page.rect.width * page.rect.height

    if n_images == 0:
        return False

    # Pattern 1: many panel images (composite figure)
    if n_images >= 4 and word_count < 450:
        return True

    # Pattern 2: any image covering >35% of page
    for img_info in image_list:
        xref = img_info[0]
        try:
            rects = page.get_image_rects(xref)
            if rects:
                img_area = rects[0].width * rects[0].height
                if img_area > page_area * 0.35:
                    return True
        except Exception:
            continue

    # Pattern 3: any image on a low-word page (caption likely in image frame)
    if n_images >= 1 and word_count < 350:
        return True

    return False


def _render_page_as_image(page: fitz.Page, dpi: int = 150) -> Image.Image:
    """
    Render an entire PDF page as a PIL Image at the given DPI.
    150 DPI is sufficient for GPT-4o vision and keeps file size reasonable.
    """
    mat = fitz.Matrix(dpi / 72, dpi / 72)   # 72 DPI is PDF default
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def _vision_full_page(
    page_image: Image.Image,
    page_number: int,
    figures_dir: str,
    paper_id: str,
    openai_client,
) -> list[dict]:
    """
    Send a full rendered page to GPT-4o vision to extract figure information.
    Used when standard text-based caption detection fails (e.g. composite figures
    with captions inside a colored border frame).

    Returns a list of dicts, one per figure detected on the page:
        {
          "figure_num": "1",
          "caption":    "Figure 1. ...",
          "description": "Panel A shows... Panel B shows...",
          "image":      PIL.Image,
          "save_path":  str,
        }
    Returns empty list if no main figure detected.
    """
    import base64, io as _io

    # Save page render to bytes for API
    buf = _io.BytesIO()
    page_image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    prompt = (
        "This is a page from a scientific publication. "
        "It contains one or more figures which may be inside a colored border or frame. "
        "For each main figure (labeled Figure N or Fig. N) on this page:\n\n"
        "1. Extract the COMPLETE figure legend/caption text exactly as written\n"
        "2. Describe each panel (A, B, C etc) - what type of plot/image it is, "
        "what the axes show, and the key finding\n"
        "3. Summarize the overall message of the figure in 1-2 sentences\n\n"
        "Format your response EXACTLY as:\n"
        "FIGURE: <number>\n"
        "CAPTION: <full caption text>\n"
        "DESCRIPTION: <panel descriptions and overall message>\n\n"
        "If there is no main figure on this page, respond with: NONE"
    )

    try:
        response = openai_client.chat.completions.create(
            model      = "gpt-4o",
            max_tokens = 800,
            messages   = [{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url":    f"data:image/png;base64,{b64}",
                            "detail": "high",
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
        )

        text = response.choices[0].message.content.strip()
        if not text or text.upper() == "NONE":
            return []

        # Parse response — may contain multiple figures
        results = []
        # Split on FIGURE: blocks
        import re as _re
        blocks = _re.split(r"(?=^FIGURE:\s*\d)", text, flags=_re.MULTILINE)

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            fig_match     = _re.search(r"^FIGURE:\s*(\d+)", block, _re.MULTILINE)
            caption_match = _re.search(r"^CAPTION:\s*(.+?)(?=^DESCRIPTION:|$)",
                                       block, _re.MULTILINE | _re.DOTALL)
            desc_match    = _re.search(r"^DESCRIPTION:\s*(.+?)$",
                                       block, _re.MULTILINE | _re.DOTALL)

            if not fig_match:
                continue

            fig_num     = fig_match.group(1)
            caption     = caption_match.group(1).strip() if caption_match else ""
            description = desc_match.group(1).strip()   if desc_match    else ""

            # Ensure caption starts with "Figure N."
            if caption and not _re.match(r"^Figure\s+\d+", caption, _re.I):
                caption = f"Figure {fig_num}. {caption}"

            # Combined content for the chunk
            combined = caption
            if description:
                combined = (caption + "\n\n" + description) if caption else description

            # Save the page render as the figure image
            figure_id = f"figure_{fig_num}"
            filename  = f"{paper_id}__{figure_id}.png"
            save_path = os.path.join(figures_dir, filename)
            page_image.save(save_path, "PNG")

            results.append({
                "figure_num":  fig_num,
                "caption":     combined,
                "save_path":   save_path,
                "width_px":    page_image.width,
                "height_px":   page_image.height,
            })

        return results

    except Exception as e:
        print(f"  ⚠️  Vision full-page failed for p{page_number}: {e}")
        return []


def extract_figures(
    doc: fitz.Document,
    paper_id: str,
    figures_dir: str = "data/figures",
    openai_client=None,
) -> list[FigureResult]:
    """
    Extract MAIN figures only from the PDF.

    Strategy:
      1. Find all large embedded images (filters out logos/icons)
      2. For each image, search for a caption using _detect_caption_on_page
         - Searches below, above, and anywhere on the page
      3. Keep ONLY images with a "Figure N" caption (skips supplemental)
      4. If caption not found via text extraction AND openai_client provided,
         call GPT-4o vision as fallback
      5. Save as PNG, named by figure number (figure_1.png, figure_2.png)
      6. Deduplicate by figure number

    Args:
        doc           : open fitz.Document
        paper_id      : used to namespace saved filenames
        figures_dir   : where to save extracted PNGs
        openai_client : optional OpenAI client for vision fallback

    Returns:
        List of FigureResult sorted by figure number.
    """
    os.makedirs(figures_dir, exist_ok=True)
    figures = []
    seen_figure_numbers = set()

    for page_num, page in enumerate(doc):
        page_number = page_num + 1

        for img_info in page.get_images(full=True):
            xref = img_info[0]

            try:
                base_image = doc.extract_image(xref)
            except Exception:
                continue

            # Skip small images
            if (base_image.get("width", 0)  < MIN_IMAGE_WIDTH or
                base_image.get("height", 0) < MIN_IMAGE_HEIGHT):
                continue

            # Convert to PIL early so we can save if needed
            try:
                pil_img = Image.open(io.BytesIO(base_image["image"])).convert("RGB")
            except Exception:
                continue

            # ── Pass 1: PyMuPDF caption search — same page ──────────────────
            caption = ""
            try:
                img_rects = page.get_image_rects(xref)
                if img_rects:
                    caption = _detect_caption_on_page(page, img_rects[0])
            except Exception:
                pass

            # ── Pass 1b: Cross-page search — caption on next page ────────────
            # Handles figures where the image fills a page and the legend
            # continues onto the top of the following page (e.g. JCI Figure 7)
            if not caption and page_num + 1 < len(doc):
                next_page = doc[page_num + 1]
                # Check the first ~300pt of the next page for a figure caption
                next_page_rect = fitz.Rect(0, 0, next_page.rect.width, 300)
                for block in next_page.get_text("blocks", clip=next_page_rect):
                    block_text = block[4].strip()
                    if MAIN_FIGURE_RE.match(block_text):
                        caption = re.sub(r"\s+", " ", block_text)
                        break

            # ── Pass 2: Vision fallback for uncaptioned images ────────────────
            # We need to save the image first to send it to the API
            if not caption and openai_client:
                # Save to a temp path for vision call
                temp_id   = f"temp_{page_number}_{xref}"
                temp_path = os.path.join(figures_dir, f"{paper_id}__{temp_id}.png")
                pil_img.save(temp_path, "PNG")

                print(f"  🔍 Vision API: checking p{page_number} image for caption...")
                caption = _vision_caption(temp_path, temp_id, openai_client)

                # Clean up temp file if no caption found
                if not caption or not MAIN_FIGURE_RE.match(caption):
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass
                    caption = ""
                else:
                    # Will be renamed below — remove temp
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass

            # ── Only keep images confirmed as main figures ────────────────────
            if not caption or not MAIN_FIGURE_RE.match(caption):
                continue

            fig_num = _extract_figure_number(caption)
            if not fig_num or fig_num in seen_figure_numbers:
                continue
            seen_figure_numbers.add(fig_num)

            # ── Save with clean figure-number filename ────────────────────────
            figure_id = f"figure_{fig_num}"
            filename  = f"{paper_id}__{figure_id}.png"
            save_path = os.path.join(figures_dir, filename)
            pil_img.save(save_path, "PNG")

            figures.append(FigureResult(
                figure_id   = figure_id,
                page_number = page_number,
                image_path  = save_path,
                caption     = caption,
                width_px    = pil_img.width,
                height_px   = pil_img.height,
            ))

    # ── Pass 3: Full-page vision for composite/framed figures ───────────────
    # Runs after per-image passes on ALL pages that look like figure pages
    # and haven't yet yielded a confirmed figure (by figure number).
    # Decoupled from Pass 2 — a page that was visited by Pass 2 but produced
    # no figure is still eligible for Pass 3.
    if openai_client:
        # Pages that already produced a confirmed figure — skip these
        pages_with_figures = {f.page_number for f in figures}

        for page_num, page in enumerate(doc):
            page_number = page_num + 1

            # Skip pages that already successfully yielded a figure
            if page_number in pages_with_figures:
                continue

            # Only process pages that look like figure pages
            if not _is_composite_figure_page(page):
                continue

            print(f"  🔍 Vision API: full-page render for composite figure p{page_number}...")
            page_image = _render_page_as_image(page)

            page_results = _vision_full_page(
                page_image    = page_image,
                page_number   = page_number,
                figures_dir   = figures_dir,
                paper_id      = paper_id,
                openai_client = openai_client,
            )

            for r in page_results:
                fig_num = r["figure_num"]
                if fig_num in seen_figure_numbers:
                    continue
                seen_figure_numbers.add(fig_num)
                pages_with_figures.add(page_number)

                figures.append(FigureResult(
                    figure_id   = f"figure_{fig_num}",
                    page_number = page_number,
                    image_path  = r["save_path"],
                    caption     = r["caption"],
                    width_px    = r["width_px"],
                    height_px   = r["height_px"],
                ))

    # Sort by figure number
    figures.sort(key=lambda f: int(re.search(r"\d+", f.figure_id).group()))
    return figures


# ── Per-page text extraction ──────────────────────────────────────────────────

def _extract_columnar_text(page: fitz.Page, col_threshold: float = 0.45) -> str:
    """
    Extract text from a page respecting two-column journal layout.

    Sorts text blocks into three groups by x-position:
      1. Full-width blocks  — abstract box, figure captions spanning full page
      2. Left column blocks — x-start < 45% of page width
      3. Right column blocks— x-start >= 45% of page width

    Each group is sorted top-to-bottom, then concatenated in reading order:
    full-width → left column → right column.

    col_threshold: fraction of page width that divides columns (0.45 = 45%).
                   Works for standard two-column journal formats (JCI, Nature,
                   NEJM, etc.). Single-column pages are handled automatically
                   since all blocks land in "left column".
    """
    blocks    = page.get_text("blocks")  # (x0,y0,x1,y1,text,block_no,block_type)
    pw        = page.rect.width
    col_split = pw * col_threshold

    full_width, left_col, right_col = [], [], []

    for b in blocks:
        if b[6] != 0:                         # skip image blocks (type=1)
            continue
        x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], b[4]
        block_width = x1 - x0

        if block_width > pw * 0.65:           # spans >65% of page = full-width
            full_width.append((y0, text))
        elif x0 < col_split:                  # left column
            left_col.append((y0, text))
        else:                                  # right column
            right_col.append((y0, text))

    full_width.sort(key=lambda b: b[0])
    left_col.sort(  key=lambda b: b[0])
    right_col.sort( key=lambda b: b[0])

    ordered = [t for _, t in full_width + left_col + right_col]
    return "\n".join(ordered)


def _extract_page_pymupdf(page: fitz.Page, page_number: int) -> PageResult:
    """Extract text from a single page using column-aware PyMuPDF extraction."""
    text = _extract_columnar_text(page)
    text = _clean_text(text)
    return PageResult(
        page_number       = page_number,
        text              = text,
        word_count        = len(text.split()),
        has_figures       = len(page.get_images()) > 0,
        extraction_method = "pymupdf",
    )


def _extract_page_pdfplumber(plumber_page, page_number: int) -> PageResult:
    """Extract text from a single page using pdfplumber (better for columns)."""
    text = plumber_page.extract_text() or ""
    text = _clean_text(text)
    return PageResult(
        page_number      = page_number,
        text             = text,
        word_count       = len(text.split()),
        has_figures      = False,   # pdfplumber doesn't track images
        extraction_method = "pdfplumber",
    )


def _clean_text(text: str) -> str:
    """
    Light cleaning applied to every page's raw extracted text.

    - Remove null bytes and non-printable control chars (keep newlines/tabs)
    - Collapse runs of 3+ blank lines to 2
    - Strip trailing whitespace per line
    - Remove hyphenation at line breaks (common in justified academic text)
    """
    # Remove null bytes / control chars except newlines and tabs
    text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", text)

    # De-hyphenate line breaks: "meth-\nods" → "methods"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Strip trailing spaces from each line
    lines = [line.rstrip() for line in text.splitlines()]

    # Collapse 3+ consecutive blank lines to 2
    cleaned_lines = []
    blank_run = 0
    for line in lines:
        if line.strip() == "":
            blank_run += 1
            if blank_run <= 2:
                cleaned_lines.append(line)
        else:
            blank_run = 0
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


# ── Main extraction function ──────────────────────────────────────────────────

def extract_pdf(
    pdf_path: str,
    figures_dir: str = "data/figures",
    extract_figs: bool = True,
    openai_client=None,
) -> ExtractionResult:
    """
    Full extraction pipeline for a single PDF.

    Strategy:
      1. Open with PyMuPDF, extract text per page
      2. Compute quality score
      3. For any page below MIN_WORDS_PER_PAGE, re-extract with pdfplumber
      4. Extract figures (if extract_figs=True)
         - PyMuPDF caption search first (free)
         - GPT-4o vision fallback for uncaptioned images (if openai_client provided)
      5. Return ExtractionResult

    Args:
        pdf_path      : path to the PDF file
        figures_dir   : where to save extracted figure PNGs
        extract_figs  : set False to skip figure extraction (faster)
        openai_client : optional OpenAI client for vision caption fallback

    Returns:
        ExtractionResult
    """
    pdf_path = str(pdf_path)

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # ── Open with PyMuPDF ─────────────────────────────────────────────────────
    doc = fitz.open(pdf_path)
    page_count = len(doc)

    metadata = _extract_metadata(doc, pdf_path)
    paper_id = _make_paper_id(metadata, pdf_path)

    print(f"  📄 {page_count} pages | title: {metadata['title'][:60]}")

    # ── Pass 1: PyMuPDF extraction ────────────────────────────────────────────
    pages: list[PageResult] = []
    for page_num, page in enumerate(doc):
        pr = _extract_page_pymupdf(page, page_num + 1)
        pages.append(pr)

    initial_quality = _quality_score(pages, page_count)
    print(f"  📊 PyMuPDF quality score: {initial_quality:.2f}")

    # ── Pass 2: pdfplumber fallback for weak pages ────────────────────────────
    weak_pages = [p for p in pages if p.word_count < MIN_WORDS_PER_PAGE]
    used_plumber = False

    if weak_pages:
        print(f"  ⚠️  {len(weak_pages)} weak pages — trying pdfplumber fallback...")
        try:
            with pdfplumber.open(pdf_path) as plumber_doc:
                for pr in pages:
                    if pr.word_count < MIN_WORDS_PER_PAGE:
                        plumber_page = plumber_doc.pages[pr.page_number - 1]
                        improved = _extract_page_pdfplumber(plumber_page, pr.page_number)
                        if improved.word_count > pr.word_count:
                            pages[pr.page_number - 1] = improved
                            used_plumber = True
        except Exception as e:
            print(f"  ⚠️  pdfplumber fallback failed: {e}")

    # ── Figure extraction ─────────────────────────────────────────────────────
    figures: list[FigureResult] = []
    if extract_figs:
        print(f"  🖼️  Extracting figures...")
        figures = extract_figures(doc, paper_id, figures_dir, openai_client=openai_client)
        print(f"  🖼️  {len(figures)} figures extracted")

    doc.close()

    # ── Assemble result ───────────────────────────────────────────────────────
    full_text    = "\n\n".join(p.text for p in pages if p.text.strip())
    final_quality = _quality_score(pages, page_count)
    total_words  = sum(p.word_count for p in pages)

    if used_plumber:
        method = "mixed"
    else:
        method = "pymupdf"

    print(f"  ✅ Done | {total_words} words | quality: {final_quality:.2f} | method: {method}")

    return ExtractionResult(
        pdf_path         = pdf_path,
        paper_id         = paper_id,
        full_text        = full_text,
        pages            = pages,
        figures          = figures,
        metadata         = metadata,
        extraction_method = method,
        quality_score    = final_quality,
        page_count       = page_count,
        word_count       = total_words,
    )


# ── Save helpers ──────────────────────────────────────────────────────────────

def save_extracted_text(result: ExtractionResult, output_dir: str = "data/extracted_texts"):
    """
    Save full_text to a .txt file.
    Returns the saved file path.
    """
    os.makedirs(output_dir, exist_ok=True)
    filename  = f"{result.paper_id}.txt"
    file_path = os.path.join(output_dir, filename)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"PAPER ID : {result.paper_id}\n")
        f.write(f"TITLE    : {result.metadata.get('title','')}\n")
        f.write(f"AUTHOR   : {result.metadata.get('author','')}\n")
        f.write(f"YEAR     : {result.metadata.get('year','')}\n")
        f.write(f"DOI      : {result.metadata.get('doi','')}\n")
        f.write(f"PAGES    : {result.page_count}\n")
        f.write(f"WORDS    : {result.word_count}\n")
        f.write(f"QUALITY  : {result.quality_score}\n")
        f.write(f"METHOD   : {result.extraction_method}\n")
        f.write("\n" + "─" * 80 + "\n\n")
        f.write(result.full_text)

    print(f"  💾 Text saved → {file_path}")
    return file_path


def save_extraction_manifest(result: ExtractionResult, output_dir: str = "data/extracted_texts"):
    """
    Save a JSON manifest (metadata + per-page summary, no full text body).
    Useful for debugging and for the chunker to inspect structure.
    """
    os.makedirs(output_dir, exist_ok=True)
    filename  = f"{result.paper_id}__manifest.json"
    file_path = os.path.join(output_dir, filename)

    manifest = {
        "paper_id":          result.paper_id,
        "pdf_path":          result.pdf_path,
        "metadata":          result.metadata,
        "extraction_method": result.extraction_method,
        "quality_score":     result.quality_score,
        "page_count":        result.page_count,
        "word_count":        result.word_count,
        "pages": [
            {
                "page_number":       p.page_number,
                "word_count":        p.word_count,
                "has_figures":       p.has_figures,
                "extraction_method": p.extraction_method,
            }
            for p in result.pages
        ],
        "figures": [
            {
                "figure_id":   f.figure_id,
                "page_number": f.page_number,
                "image_path":  f.image_path,
                "caption":     f.caption,
                "width_px":    f.width_px,
                "height_px":   f.height_px,
            }
            for f in result.figures
        ],
    }

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"  📋 Manifest saved → {file_path}")
    return file_path


# ── Batch processing ──────────────────────────────────────────────────────────

def process_pdf_folder(
    pdf_dir: str = "data/pdfs",
    extracted_dir: str = "data/extracted_texts",
    figures_dir: str = "data/figures",
    archive_dir: str = "data/pdfs_archive",
    skip_existing: bool = True,
) -> list[ExtractionResult]:
    """
    Process all PDFs in a folder.

    Args:
        pdf_dir      : folder to scan for .pdf files
        extracted_dir: where to save .txt and manifest files
        figures_dir  : where to save figure PNGs
        archive_dir  : where to move PDFs after processing (None = don't move)
        skip_existing: skip PDFs whose .txt already exists

    Returns:
        List of ExtractionResult for each processed PDF.
    """
    pdf_files = sorted(Path(pdf_dir).glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return []

    print(f"\n📂 Found {len(pdf_files)} PDF(s) in {pdf_dir}\n")
    results = []

    for pdf_path in pdf_files:
        print(f"{'─'*60}")
        print(f"🔍 Processing: {pdf_path.name}")

        # Skip check
        if skip_existing:
            existing = list(Path(extracted_dir).glob(f"*{pdf_path.stem}*.txt"))
            if existing:
                print(f"  ⏭️  Already extracted — skipping")
                continue

        try:
            result = extract_pdf(
                str(pdf_path),
                figures_dir=figures_dir,
            )
            save_extracted_text(result, extracted_dir)
            save_extraction_manifest(result, extracted_dir)
            results.append(result)

            # Archive the PDF
            if archive_dir:
                os.makedirs(archive_dir, exist_ok=True)
                dest = os.path.join(archive_dir, pdf_path.name)
                os.rename(str(pdf_path), dest)
                print(f"  📦 Archived → {dest}")

        except Exception as e:
            print(f"  ❌ Error processing {pdf_path.name}: {e}")
            continue

    print(f"\n{'─'*60}")
    print(f"✅ Processed {len(results)} PDF(s)")
    return results
