# Publications Vector Store — How It Works

This document explains how academic papers are processed, chunked, embedded, and stored in ChromaDB for semantic search. It is intended as a reference for anyone working with or extending the publications RAG pipeline.

---

## Overview

```
PDF File
    ↓
[1] Text Extraction       pdf_reader.py
    ↓
[2] Section Parsing       section_parser.py
    ↓
[3] Chunking              chunker.py
    ↓
[4] Embedding             vector_store.py + OpenAI API
    ↓
[5] Storage               ChromaDB (data/chroma_db/)
    ↓
[6] Retrieval             semantic search at query time
    ↓
[7] Response              GPT-4o-mini with retrieved context
```

---

## Step 1 — Text Extraction

Each PDF is processed by `pdf_reader.py` using PyMuPDF with column-aware block sorting. The extractor handles standard two-column journal layouts (JCI, Nature, NEJM, etc.) by classifying text blocks as full-width, left-column, or right-column and reading them in the correct order.

Figures are extracted separately using a three-pass strategy:
- **Pass 1** — text caption search (free, no API)
- **Pass 2** — individual image GPT-4o vision (only if Pass 1 fails)
- **Pass 3** — full-page render GPT-4o vision (only for composite figure layouts)

Output: clean full text + figure images + metadata (title, authors, DOI, year).

---

## Step 2 — Section Parsing

`section_parser.py` detects section headers and normalizes them into canonical types regardless of how different journals label them.

| Canonical Type | Matches |
|---|---|
| `overview` | Abstract, Summary, implicit abstract |
| `introduction` | Introduction, Background, Rationale |
| `methods` | Methods, Materials & Methods, all sub-sections |
| `results` | Results, Findings |
| `discussion` | Discussion, Interpretation |
| `conclusion` | Conclusion, Concluding Remarks |
| `acknowledgements` | Acknowledgements, Funding, Author Contributions |

Sub-headers like "RNA Extraction" or "Western Blot Analysis" inherit their parent section type, allowing all methods sub-sections to be merged into a single `methods` chunk.

---

## Step 3 — Chunking

`chunker.py` assembles the final structured JSON from the extracted text and parsed sections. Each canonical section becomes one chunk.

**Example chunk:**
```json
{
  "paper_id":    "targeting_multiple_cannabinoid_anti_tumour_pathway__2014",
  "chunk_type":  "results",
  "chunk_id":    "targeting_multiple_cannabinoid__2014__results__003",
  "title":       "Targeting multiple cannabinoid anti-tumour pathways...",
  "authors":     "Scott D. McAllister, ...",
  "year":        "2014",
  "doi":         "10.1038/bjc.2014.438",
  "content":     "CBD increased survival rates (P < 0.006)...",
  "word_count":  1843
}
```

A typical paper produces 6–13 chunks. Figure captions are stored as separate figure chunks but are not embedded into the vector store (only text sections are indexed).

**Chunk splitting:** Chunks exceeding 4000 words are automatically split with 100-word overlap to stay within the embedding model's 8192-token limit. Scientific text averages ~1.87 tokens per word, so 4000 words ≈ 7480 tokens.

---

## Step 4 — Embedding

This is where text is converted into numbers.

Each chunk is sent to OpenAI's `text-embedding-3-small` model via the API. The model returns a vector of **1536 floating point numbers** that represents the semantic meaning of the text.

```
"CBD increased survival rates (P < 0.006) and was effective 
 at targeting metastatic foci that were 2mm or larger..."

        ↓  text-embedding-3-small

[0.023, -0.412, 0.887, 0.134, -0.291, 0.756, ...]
                    × 1536 numbers
```

**Where do these numbers come from?**

The embedding model was trained by OpenAI on hundreds of billions of words. During training it learned to place semantically similar text close together in 1536-dimensional space. The model is frozen — it does not learn from your papers. The same input text always produces the same output vector, on any machine, at any time.

**What the numbers represent:**

The numbers themselves are meaningless in isolation. What matters is their geometric relationship to other vectors. Two chunks with similar meaning will have vectors that point in similar directions in 1536-dimensional space — they have high *cosine similarity*. Two unrelated chunks will point in very different directions — low cosine similarity.

```
"CBD inhibits glioblastoma"         →  vector A
"cannabidiol brain tumor treatment" →  vector B  (close to A)
"patient scheduling software"       →  vector C  (far from A and B)
```

**Cost:** approximately $0.00002 per 1000 tokens. Embedding all 118 chunks costs a few cents. Once stored, vectors are never recomputed unless you explicitly rebuild the index.

---

## Step 5 — Storage (ChromaDB)

The vectors and their associated text/metadata are stored persistently in `data/chroma_db/`.

**Files on disk:**

| File | Contents |
|---|---|
| `chroma.sqlite3` | Chunk text, metadata, and index structure |
| `data_level0.bin` | Raw embedding vectors (binary format) |
| `header.bin` | Index metadata (dimensions, count, settings) |
| `length.bin` | Size of each stored vector |
| `link_lists.bin` | Graph connections between vectors for fast traversal |

ChromaDB uses an index structure called **HNSW** (Hierarchical Navigable Small World) to organize vectors into a graph where similar vectors are connected. This allows fast approximate nearest-neighbor search without comparing every vector to every other vector — critical at scale.

**Current index:**
- 22 papers
- 118 chunks
- 1536 dimensions per vector
- Similarity metric: cosine

---

## Step 6 — Retrieval (Query Time)

When a user asks a question, the same embedding model converts the question into a vector, and ChromaDB finds the chunks whose vectors are closest.

```
User: "What methods did you use in your cannabinoid breast cancer study?"

    ↓  text-embedding-3-small

[0.041, -0.389, 0.901, ...]   ← query vector

    ↓  ChromaDB cosine similarity search

Top 5 closest chunks returned:
  1. targeting_multiple_cannabinoid__2014__methods   (score: 0.91)
  2. pathways_mediating_cannabidiol__2012__methods   (score: 0.87)
  3. cannabidiol_enhances__2010__methods             (score: 0.84)
  4. targeting_multiple_cannabinoid__2014__results   (score: 0.79)
  5. reactive_oxygen_species__2015__methods          (score: 0.76)
```

**Why this is powerful:**

The query "What methods did you use?" will find methods chunks even if they never contain the word "methods" — because the meaning is encoded in the vector, not the keywords. A keyword search for "methods" would miss any chunk that uses "protocol", "procedure", or "approach" instead.

**Query routing:**

Before searching, the agent detects the intent of the query and filters by chunk type to improve precision:

| Query pattern | Chunk types searched |
|---|---|
| "how did you / what methods / protocol" | `methods` only |
| "what did you find / results / findings" | `results` + `discussion` |
| "what is this paper about / overview" | `overview` + `introduction` |
| ambiguous | all types |

---

## Step 7 — Response Generation

The top-5 retrieved chunks are formatted and injected into the GPT-4o-mini system prompt as context. The model is instructed to answer only from the provided excerpts, cite specific figures and statistics, and respond in first person as Sean.

```
System prompt:
  "Here are relevant excerpts from Sean's publications:
   [1] 'Targeting multiple cannabinoid...' (2014) — RESULTS
       CBD increased survival rates (P < 0.006)...
   [2] ...
   Answer the user's question using only these excerpts."

User: "What did you find about CBD and metastasis?"

Response: "In my 2014 breast cancer study, we found that CBD 
increased survival rates (P < 0.006) and was effective at 
targeting metastatic foci 2mm or larger..."
```

Temperature is set to 0.3 (lower than other agents) to keep the model faithful to the retrieved facts rather than elaborating beyond them.

---

## Adding New Papers

```bash
# 1. Name the PDF: "YEAR Full Paper Title.pdf"
cp "2024 My New Paper.pdf" data/pdfs/

# 2. Run the pipeline (skips already-processed papers)
python scripts/ingest_papers.py

# 3. Move to archive
mv "data/pdfs/2024 My New Paper.pdf" data/pdfs_archive/

# 4. Verify
python scripts/inspect_store.py

# 5. Commit updated vector store
git add data/chroma_db/ data/json_outputs/
git commit -m "Add new paper: 2024 My New Paper"
git push
```

Render redeploys automatically. The new paper is live within minutes.

---

## Rebuilding the Index

If you ever need to wipe and rebuild from scratch:

```bash
rm -rf data/chroma_db/
python scripts/ingest_papers.py --rebuild-index
```

This re-embeds all chunks from the existing JSON files in `data/json_outputs/`. It does not re-extract PDFs. Cost: a few cents for the embedding API calls.

---

## Key Facts

- **Embedding model:** OpenAI `text-embedding-3-small`
- **Vector dimensions:** 1536
- **Similarity metric:** Cosine similarity
- **Token limit per chunk:** 8192 tokens (~4000 words)
- **Chunks per paper:** 6–13 depending on section structure
- **Index type:** HNSW (Hierarchical Navigable Small World)
- **Storage:** Local SQLite + binary files in `data/chroma_db/`
- **Embedding cost:** ~$0.00002 per 1000 tokens (a few cents per paper)
- **The model is fixed:** OpenAI trained it once — your papers do not change it
