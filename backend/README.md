# Sean's Digital Twin — Backend API

A multi-agent AI system that acts as a digital twin, answering questions about work experience, projects, publications, calendar availability, and hobbies. Publications are powered by a semantic RAG pipeline backed by ChromaDB.

---

## Architecture

```
User Query
    ↓
Guardrails (Moderation, PII, Jailbreak)
    ↓
Classifier Agent
    ↓
Specialized Agents (General, Projects, Publications, Calendar, Hobbies)
    ↓
Response
```

**Publications agent** uses semantic RAG — queries are embedded and matched against 100+ chunks from 22 papers stored in ChromaDB, rather than a static JSON summary.

---

## Project Structure

```
backend/
├── main.py                      # FastAPI application, routes, startup
├── llm_client.py                # OpenAI client initialization
├── guardrails.py                # Content filtering (moderation, PII, jailbreak)
├── classifier.py                # Query classification agent
├── agents.py                    # Specialized agents (general, projects, etc.)
│
├── src/                         # PDF processing + RAG pipeline
│   ├── pdf_reader.py            # Step 1 — text extraction + figure detection
│   ├── section_parser.py        # Step 2 — header detection + normalization
│   ├── chunker.py               # Step 3 — standardized JSON chunks
│   ├── vector_store.py          # Step 4 — ChromaDB embedding + search
│   └── agents_publications.py  # Publications RAG agent (reference)
│
├── scripts/                     # Offline pipeline tools (run locally)
│   ├── ingest_papers.py         # Batch ingest PDFs end-to-end
│   ├── inspect_store.py         # Inspect ChromaDB contents
│   ├── diagnose_pdf.py          # Debug PDF layout issues
│   └── debug_title.py           # Debug title resolution
│
├── data/
│   ├── resume.json              # Work experience, education, skills
│   ├── projects.json            # Personal and community projects
│   ├── publications.json        # Publications list (fallback)
│   ├── hobbies.json             # Interests outside of work
│   ├── json_outputs/            # Chunked JSON per paper (22 papers)
│   ├── chroma_db/               # ChromaDB persistent vector store
│   ├── pdfs/                    # Drop new PDFs here for ingestion
│   └── pdfs_archive/            # Processed PDFs moved here after ingestion
│
├── requirements.txt
└── .env                         # API keys (not committed)
```

---

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd backend
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file:

```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:

```
OPENAI_API_KEY=sk-your-actual-api-key-here
```

### 5. Run the server locally

```bash
python main.py
```

Or using uvicorn directly:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

On startup you should see:

```
[startup] Initializing publications vector store...
[startup] Vector store ready.
```

The API will be available at `http://localhost:8000`

---

## Publications RAG Pipeline

Publications use semantic search rather than a static JSON file. On startup, `main.py` calls `init_vector_store()` which loads the pre-built ChromaDB index from `data/chroma_db/`.

**Current index:** 22 papers, 108 chunks, embedded with `text-embedding-3-small`.

### Adding a new paper

```bash
# 1. Name the PDF with year and full title
cp "2024 My New Paper Title.pdf" data/pdfs/

# 2. Run the ingest pipeline (processes new PDFs only)
python scripts/ingest_papers.py

# 3. Move processed PDF to archive
mv "data/pdfs/2024 My New Paper Title.pdf" data/pdfs_archive/

# 4. Verify indexing
python scripts/ingest_papers.py --stats
```

### Inspecting the vector store

```bash
# Summary of indexed papers and chunk counts
python scripts/inspect_store.py

# Test a semantic search query
python scripts/inspect_store.py --query "cannabidiol glioblastoma"

# Filter by chunk type
python scripts/inspect_store.py --type results
```

### How the publications agent works

1. Query intent is detected via regex (methods / results / overview / broad)
2. Top-5 semantically matched chunks are retrieved from ChromaDB
3. Chunks are injected into the GPT-4o-mini prompt as context
4. Agent responds in first person citing specific figures, data points, and DOIs
5. Falls back to `publications.json` summary if the vector store is unavailable

---

## API Endpoints

### POST /chat

Main chat endpoint for interacting with the digital twin.

**Request:**
```json
{
  "message": "What's your work experience?",
  "session_id": "optional-session-id"
}
```

**Response:**
```json
{
  "response": "I've worked as a Senior Data Scientist at...",
  "session_id": "uuid-here",
  "category": "general"
}
```

### DELETE /session/{session_id}

Clear conversation history for a session.

### GET /health

Health check endpoint.

---

## Deployment

### Deploy to Render

1. Create a new Web Service on [Render](https://render.com)
2. Connect your GitHub repository
3. Configure:
   - **Root Directory:** `backend`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Add environment variable:
   - Key: `OPENAI_API_KEY`
   - Value: Your OpenAI API key
5. Deploy

The `data/chroma_db/` and `data/json_outputs/` folders are committed to the repo and deployed with the app — no rebuild needed on Render.

**Note:** `data/pdfs/`, `data/pdfs_archive/`, `data/figures/`, and `data/extracted_texts/` are gitignored and never deployed. PDF processing runs locally only.

---

## Testing

```bash
# Health check
curl http://localhost:8000/health

# General query
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What projects have you worked on?"}'

# Publications RAG query
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What have you published on cannabidiol and cancer?"}'

# Methods-specific query (tests chunk routing)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How did you screen drugs in your pharmacogenomic study?"}'
```

Or use the interactive docs at `http://localhost:8000/docs`

---

## Customization

### Modify Agent Behavior

Edit the system messages in `agents.py` to change how each agent responds.

### Add New Categories

1. Add category to classifier in `classifier.py`
2. Create new agent function in `agents.py`
3. Update routing in `main.py`

### Adjust Guardrails

Modify detection logic in `guardrails.py`:
- Update jailbreak patterns
- Adjust PII detection sensitivity
- Customize blocked response messages

---

## Frontend Integration

The backend expects requests from your React frontend. Update CORS origins in `main.py`:

```python
allow_origins=["https://your-vercel-app.vercel.app"]
```

---

## Notes

- Conversation history is stored in memory (use Redis for production)
- History limited to last 10 messages per session (5 turns)
- Publications agent uses `gpt-4o-mini` at temperature 0.3 for factual accuracy
- All other agents use `gpt-4o-mini` at temperature 0.7
- Vector store initializes at startup — cold start adds ~2 seconds on Render free tier

---

## License

MIT
