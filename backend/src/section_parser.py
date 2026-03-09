"""
src/section_parser.py
─────────────────────────────────────────────────────────────────────────────
Detects section headers in extracted PDF text and normalizes them into
canonical chunk types, regardless of the journal's naming conventions.

Canonical sections (standardized output keys)
─────────────────────────────────────────────
  overview        — abstract / summary (often implicit, no header)
  introduction    — introduction / background / motivation
  methods         — methods / methodology / materials & methods / experimental
  results         — results / findings / outcomes / observations
  discussion      — discussion / interpretation
  conclusion      — conclusion / conclusions / summary
  references      — references / bibliography
  supplementary   — supplementary / appendix / extended data
  acknowledgements— acknowledgements / funding / author contributions
  other           — any header not matched (flagged for review)

Two-pass detection strategy
────────────────────────────
  Pass 1 — Regex alias matching against a comprehensive synonym table.
           Catches ~95% of real journal headers.
  Pass 2 — LLM fallback for any unrecognised headers (only fires when
           ambiguous headers are found and an OpenAI client is passed in).
"""

import re
import os
import json
from dataclasses import dataclass, field
from typing import Optional


# ── Canonical section alias table ─────────────────────────────────────────────
SECTION_ALIASES: dict[str, list[str]] = {
    "overview": [
        r"^abstract",
        r"^summary",
        r"^synopsis",
        r"^overview",
        r"^highlights",
        r"^graphical\s+abstract",
        r"^plain\s+language\s+summary",
        r"^lay\s+summary",
    ],
    "introduction": [
        r"^introduction",
        r"^background",
        r"^motivation",
        r"^rationale\s*$",
        r"^clinical\s+background",
        r"^scientific\s+background",
        r"^context",
        r"^preamble",
    ],
    "methods": [
        r"^methods?",
        r"^methodology",
        r"^materials?\s+and\s+methods?",
        r"^methods?\s+and\s+materials?",
        r"^patients?\s+and\s+methods?",
        r"^subjects?\s+and\s+methods?",
        r"^participants?\s+and\s+methods?",
        r"^experimental\s+(section|procedures?|design|methods?)\s*$",
        r"^study\s+design",
        r"^clinical\s+methods?",
        r"^statistical\s+(analysis|methods?)",
        r"^statistics$",                        # JCI uses bare "Statistics"
        r"^computational\s+methods?",
        r"^model\s+description",
        r"^data\s+collection",
        r"^study\s+approval",                   # ethics/IRB section in JCI
        r"^ethics",
        r"^irb",
        r"^animal\s+studies?",
        r"^cell\s+(lines?|culture)",
        r"^rna\s+(extraction|seq|sequencing)",
        r"^western\s+blot(\s+analysis)?\s*$",
        r"^immunofluorescence",
        r"^immunohistochemistry",
        r"^ihc$",
        r"^elisa$",
        r"^flow\s+cytometry",
        r"^antibod(ies|y)\s*$",
        r"^reagents?",
        r"^drug\s+(screening|treatment|combination)\s*$",
        r"^pharmacological",
        r"^htds\s*$",                               # high-throughput drug screening
        r"^pdx\s+(model|development|xenograft|tumor)\b",                                # patient-derived xenograft
        r"^murine\s+model",
        r"^multiplex",
        r"^spatial\s+(profiling|analysis)",
        r"^bioinformatics",
        r"^gene\s+expression\s+analys",
        r"^quantitative\s+rt",
        r"^qrt.pcr",
        r"^sex\s+as\s+a\s+biological",          # NIH-required methods section
    ],
    "results": [
        r"^results?\s*$",
        r"^findings?",
        r"^outcomes?",
        r"^observations?",
        r"^experimental\s+results?",
        r"^main\s+results?",
        r"^key\s+findings?",
    ],
    "discussion": [
        r"^discussion",
        r"^interpretation",
        r"^commentary",
        r"^implications?",
        r"^significance",
    ],
    "conclusion": [
        r"^conclusions?",
        r"^concluding\s+remarks?",
        r"^closing\s+remarks?",
        r"^summary\s+(and\s+conclusions?)?",
        r"^conclusions?\s+and\s+future",
        r"^final\s+remarks?",
        r"^perspectives?",
        r"^results\s+and\s+discussion",   # common combo section
    ],
    "references": [
        r"^references?\s*$",
        r"^bibliography",
        r"^works?\s+cited",
        r"^literature\s+cited",
        r"^citations?",
    ],
    "supplementary": [
        r"^supplementary",
        r"^supplemental(\s+(methods?|data|figures?|tables?|information))?\s*$",
        r"^appendix",
        r"^appendices",
        r"^extended\s+data",
        r"^supporting\s+information",
        r"^online\s+(methods?|supplementary)",
    ],
    "acknowledgements": [
        r"^acknowledgements?",
        r"^acknowledgments?",
        r"^funding",
        r"^conflict\s+of\s+interest",
        r"^author\s+contributions?",
        r"^data\s+availability",
        r"^code\s+availability",
        r"^declarations?",
        r"^competing\s+interests?",
        r"^disclosures?",
    ],
}

# Compile all alias patterns once at import time
_COMPILED_ALIASES: dict[str, list[re.Pattern]] = {
    canonical: [re.compile(pat, re.IGNORECASE) for pat in patterns]
    for canonical, patterns in SECTION_ALIASES.items()
}

# ── Header detection patterns ─────────────────────────────────────────────────
_HEADER_PATTERNS = [
    re.compile(r"^[A-Z][A-Z\s\-&/]{2,}$"),               # ALL CAPS
    re.compile(r"^[A-Z][a-zA-Z\s\-&/,()]{2,60}$"),        # Title Case
    re.compile(r"^\d+(\.\d+)*\.?\s+[A-Z]"),               # Numbered: 2. Methods
    re.compile(r"^([A-Z]\s){2,}[A-Z]$"),                  # Spaced: R E S U L T S
]

_HEADER_BLACKLIST = re.compile(
    r"""
    figure\s*\d+            |
    fig\.\s*\d+             |
    table\s*\d+             |
    scheme\s*\d+            |
    supplementary\s+fig     |
    extended\s+data\s+fig   |
    ^\d+$                   |   # lone numbers
    ^[ivxlcdm]+$            |   # roman numerals alone
    \©\s*20\d\d             |
    doi:                    |
    received:               |
    accepted:               |
    published:              |
    reference\s+information |   # "Reference information: J Clin Invest..."
    j\s+clin\s+invest       |   # journal citation lines
    correspondence          |
    ^\d+\.\s+[A-Z][a-z]     |   # "1. Siegel RL..." — numbered references
    ^\d+\.\s{2,}            |   # "12.        Ning B..." — spaced numbered refs
    et\s+al\.               |   # any line with "et al." is a reference
    [A-Z][a-z]+\s+[A-Z]{2}  |   # "Siegel RL" author pattern
    ^https?://                  # URLs
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Sentence-fragment indicators: lines starting with these words are body text
# NOTE: must NOT use VERBOSE mode here — spaces in alternation would be ignored
_SENTENCE_FRAGMENT_RE = re.compile(
    r"^(the|a|an|this|these|those|our|we|it|in|of|for|and|but|however|"
    r"although|despite|given|since|because|when|while|with|by|from|"
    r"including|following|using|based|compared|results?,|drug|cell|tumor|"
    r"cancer|patient|mouse|mice|model|data|analysis|expression|treatment|"
    r"combination|activity|effect|efficacy|response|resistance|pathway|"
    r"antibody|protein|gene|mrna|rna|dna|assay|concentration|dose|figure|"
    r"table|supplemental|pdx|htds|experimental|rationale|reference)\s",
    re.IGNORECASE,
)

_BOILERPLATE = re.compile(
    r"""
    journal\s+of            |
    r\s*e\s*s\s*e\s*a\s*r\s*c\s*h  |
    a\s*r\s*t\s*i\s*c\s*l\s*e      |
    open\s+access           |
    creative\s+commons      |
    all\s+rights\s+reserved |
    this\s+article\s+is     |
    downloaded\s+from       |
    ^https?://              |
    ^\s*\d{1,4}\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class DetectedSection:
    canonical:     str
    raw_header:    str
    content:       str
    start_line:    int
    confidence:    str      # "regex" | "llm" | "implicit" | "unknown"
    is_subsection: bool = False


@dataclass
class ParseResult:
    paper_id:          str
    sections:          dict[str, str]
    all_detected:      list[DetectedSection]
    unmatched:         list[str]
    implicit_abstract: bool = False
    title:             str = ""


# ── Utilities ─────────────────────────────────────────────────────────────────

def _normalise_spaces(text: str) -> str:
    """Collapse spaced-out capitals: 'R E S U L T S' → 'RESULTS'."""
    if re.match(r"^([A-Z]\s){2,}[A-Z]$", text.strip()):
        return text.replace(" ", "")
    return text


def _is_candidate_header(line: str) -> bool:
    """
    Return True if a line looks structurally like a section header.

    Gate order (fail-fast):
      1. Empty / too long → reject
      2. Boilerplate → reject
      3. Known section alias → accept immediately
      4. Truncated fragment / sentence fragment / inline citation → reject
      5. Blacklist patterns → reject
      6. Ends mid-sentence → reject
      7. Must match structural header pattern
    """
    stripped = line.strip()
    if not stripped:
        return False

    if len(stripped) > 80:
        return False

    if _BOILERPLATE.search(stripped):
        return False

    # Gate 3: known section aliases — accept immediately
    norm = _normalise_spaces(stripped)
    for canonical, patterns in _COMPILED_ALIASES.items():
        for pat in patterns:
            if pat.search(norm):
                return True

    words = stripped.split()

    # Gate 4a: sentence fragments starting with body-text words
    if _SENTENCE_FRAGMENT_RE.match(stripped):
        return False

    # Gate 4b: truncated fragments — end with preposition/article
    last_word = words[-1].lower() if words else ""
    fragment_endings = {
        "by","in","of","the","a","an","and","or","with","for","from",
        "to","at","on","as","is","are","was","were","be","been","has",
        "have","had","not","that","this","these","its","it","m","f",
    }
    if len(stripped) > 20 and last_word in fragment_endings:
        return False

    # Gate 4c: truncated mid-word endings (common in two-column PDF extraction)
    truncated = re.compile(
        r'(expressi|compar|analysi|profili|cultur|regulat|activat|inhibit|'
        r'resistan|combinat|concentrat|administrat|evaluat|identif|demonstrat|'
        r'characteriz|signific|associat|therapeut|pharmac|preclin|transcripto)$',
        re.I
    )
    if len(stripped) > 20 and truncated.search(stripped):
        return False

    # Gate 4d: mid-string period = two sentences = body text
    mid_text = stripped[stripped.find(' '):] if ' ' in stripped else ''
    if re.search(r'\w\.\s+[A-Z]', mid_text):
        return False

    # Gate 4e: inline parenthetical citations = body text
    if re.search(r'\(\d+\)|\(Figure|\(Supplemental|\(Table|\(Supp', stripped):
        return False

    # Gate 4f: dosing/concentration = methods body text
    if re.search(r'\d+\s*(mg|ug|ng|ml|ul|mM|uM|nM|%)(\s|/)', stripped, re.I):
        return False

    # Gate 4g: multiple commas = sentence or author list
    if stripped.count(",") >= 2:
        return False

    # Gate 5: blacklist
    if _HEADER_BLACKLIST.search(stripped):
        return False

    # Gate 6: ends with period + >5 words = sentence
    if stripped.endswith(".") and len(words) > 5:
        return False

    # Gate 7: must match structural header pattern
    return any(pat.match(norm) for pat in _HEADER_PATTERNS)


def _classify_header(header_text: str) -> tuple[str, str]:
    """Match a header against the alias table. Returns (canonical, confidence)."""
    norm = _normalise_spaces(header_text).strip()
    for canonical, patterns in _COMPILED_ALIASES.items():
        for pat in patterns:
            if pat.search(norm):
                return canonical, "regex"
    return "other", "unknown"


def _extract_title_from_first_page(first_page_text: str) -> str:
    """
    Heuristic title extraction from the first ~30 lines.
    Picks the longest plausible non-boilerplate line.
    """
    lines = first_page_text.strip().splitlines()[:30]
    candidates = []

    # Patterns that indicate body text / reagent lines — not a title
    body_text_signals = re.compile(
        r"kit\s*\(|using\s+the|were\s+(performed|obtained|prepared)|"
        r"sequencing\s+was|analysis\s+was|cells?\s+were|prepared\s+using",
        re.IGNORECASE,
    )

    for line in lines:
        stripped = line.strip()
        if len(stripped) < 15:
            continue
        if _BOILERPLATE.search(stripped):
            continue
        if re.match(r"^https?://|^doi:", stripped, re.IGNORECASE):
            continue
        if stripped.count(",") > 4:   # likely author list
            continue
        if stripped.isupper() and len(stripped.split()) > 3:
            continue
        if body_text_signals.search(stripped):
            continue
        candidates.append(stripped)

    if not candidates:
        return ""
    return max(candidates[:15], key=len)


def _llm_classify_headers(unmatched: list[str], client) -> dict[str, str]:
    """GPT-4o-mini fallback for headers regex couldn't classify."""
    canonical_list = ", ".join(SECTION_ALIASES.keys())
    headers_block  = "\n".join(f"- {h}" for h in unmatched)
    prompt = f"""You are classifying section headers from academic papers.
Map each header to one of these canonical types: {canonical_list}
If none fit, use "other".
Respond ONLY with a JSON object. Example: {{"Study Population": "methods"}}

Headers:
{headers_block}"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=300,
        )
        raw = re.sub(r"```json|```", "", response.choices[0].message.content).strip()
        return json.loads(raw)
    except Exception as e:
        print(f"  ⚠️  LLM header classification failed: {e}")
        return {}


def _clean_section_content(lines: list[str]) -> str:
    """Clean body lines: remove boilerplate, lone page numbers, excess blanks."""
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if _BOILERPLATE.search(stripped):
            continue
        if re.match(r"^\d{1,4}$", stripped):
            continue
        if re.match(r"^(figure|fig\.|table|scheme)\s*\d+\s*[.\-]?\s*$", stripped, re.IGNORECASE):
            continue
        cleaned.append(line)
    text = "\n".join(cleaned)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _strip_boilerplate_lines(text: str) -> str:
    lines = text.splitlines()
    clean = [l for l in lines if not _BOILERPLATE.search(l.strip())]
    return "\n".join(clean).strip()


# ── Core parser ───────────────────────────────────────────────────────────────

def parse_sections(
    full_text: str,
    paper_id: str = "unknown",
    llm_client=None,
    skip_sections: list[str] | None = None,
) -> ParseResult:
    """
    Parse extracted PDF text into canonical sections.

    Args:
        full_text     : raw text from pdf_reader.ExtractionResult.full_text
        paper_id      : used for labelling
        llm_client    : optional OpenAI client for fallback classification
        skip_sections : canonical names to exclude e.g. ["references", "acknowledgements"]

    Returns:
        ParseResult with .sections dict and .all_detected list
    """
    if skip_sections is None:
        skip_sections = []

    lines = full_text.splitlines()

    # Step 1: title from first page
    title = _extract_title_from_first_page("\n".join(lines[:40]))

    # Step 2: find all candidate header lines → (line_index, raw_text)
    header_positions: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        if _is_candidate_header(line):
            header_positions.append((i, line.strip()))

    # Step 3: classify each header and slice its content
    detected: list[DetectedSection] = []
    unmatched_headers: list[str] = []

    for idx, (line_idx, raw_header) in enumerate(header_positions):
        canonical, confidence = _classify_header(raw_header)
        next_line    = header_positions[idx + 1][0] if idx + 1 < len(header_positions) else len(lines)
        content      = _clean_section_content(lines[line_idx + 1 : next_line])

        if canonical == "other":
            unmatched_headers.append(raw_header)

        detected.append(DetectedSection(
            canonical  = canonical,
            raw_header = raw_header,
            content    = content,
            start_line = line_idx,
            confidence = confidence,
        ))

    # Step 4: LLM fallback
    if unmatched_headers and llm_client:
        print(f"  🤖 LLM classifying {len(unmatched_headers)} unmatched header(s)...")
        llm_mapping = _llm_classify_headers(unmatched_headers, llm_client)
        for section in detected:
            if section.canonical == "other" and section.raw_header in llm_mapping:
                section.canonical  = llm_mapping[section.raw_header]
                section.confidence = "llm"
                if section.raw_header in unmatched_headers:
                    unmatched_headers.remove(section.raw_header)

    # Step 5: implicit abstract — text before first detected header
    implicit_abstract = False
    first_header_line = header_positions[0][0] if header_positions else len(lines)
    pre_header_text   = _clean_section_content(lines[:first_header_line])
    clean_pre         = _strip_boilerplate_lines(pre_header_text)

    if clean_pre and len(clean_pre.split()) > 30:
        detected.insert(0, DetectedSection(
            canonical  = "overview",
            raw_header = "[implicit abstract]",
            content    = clean_pre,
            start_line = 0,
            confidence = "implicit",
        ))
        implicit_abstract = True

    # Step 6: subsection parent-inheritance
    # Any "other" section that sits between two instances of the same canonical
    # section (or after a known section with no intervening canonical change)
    # inherits the most recent known parent's canonical type.
    # This handles sub-headers like "Cell Lines and Culture Conditions" inside
    # Methods, or "Identification of DEGs" inside Results.
    last_known_canonical = None
    for sec in detected:
        if sec.canonical != "other":
            last_known_canonical = sec.canonical
        elif sec.canonical == "other" and last_known_canonical is not None:
            # Inherit parent — but never inherit references/acknowledgements
            # into what is likely a sub-result or sub-method
            if last_known_canonical not in ("references", "acknowledgements", "supplementary"):
                sec.canonical    = last_known_canonical
                sec.confidence   = f"inherited:{last_known_canonical}"
                sec.is_subsection = True

    # Step 7: merge sections sharing the same canonical name
    sections: dict[str, list[str]] = {}
    for sec in detected:
        if sec.canonical in skip_sections:
            continue
        if sec.content.strip():
            sections.setdefault(sec.canonical, []).append(sec.content)

    merged = {k: "\n\n".join(v) for k, v in sections.items()}

    final_unmatched = [
        sec.raw_header for sec in detected
        if sec.canonical == "other" and sec.content.strip()
        and not sec.is_subsection
    ]

    return ParseResult(
        paper_id          = paper_id,
        sections          = merged,
        all_detected      = detected,
        unmatched         = final_unmatched,
        implicit_abstract = implicit_abstract,
        title             = title,
    )


# ── Pretty printer ────────────────────────────────────────────────────────────

def print_parse_report(result: ParseResult):
    """Print a human-readable summary of the parse result."""
    print(f"\n{'─'*60}")
    print("SECTION PARSE REPORT")
    print(f"{'─'*60}")
    print(f"  paper_id         : {result.paper_id}")
    print(f"  title (detected) : {result.title[:70]}")
    print(f"  implicit abstract: {result.implicit_abstract}")
    print(f"  canonical keys   : {list(result.sections.keys())}")

    print(f"\n  {'CANONICAL':<18} {'RAW HEADER':<35} {'CONF':<22} WORDS")
    print(f"  {'─'*18} {'─'*35} {'─'*22} {'─'*5}")
    for sec in result.all_detected:
        words = len(sec.content.split())
        if words == 0:
            continue
        sub_flag = " ↳" if sec.is_subsection else "  "
        print(f"  {sec.canonical:<18} {sec.raw_header[:33]:<35} {sec.confidence:<22} {words}{sub_flag}")

    if result.unmatched:
        print(f"\n  ⚠️  UNMATCHED HEADERS — add these to SECTION_ALIASES if recurring:")
        for h in result.unmatched:
            print(f"     • {h}")
    else:
        print(f"\n  ✅ All headers matched")

    print(f"\n  CONTENT PREVIEW PER SECTION:")
    for key, content in result.sections.items():
        words   = len(content.split())
        preview = content[:100].replace("\n", " ")
        print(f"    [{key:<16}] {words:>5}w  {preview}…")
