# Sean's Digital Avatar

An AI-powered digital twin that answers questions about my research publications, work experience, projects, and hobbies. Built with a multi-agent RAG architecture — try it live at **[digital-avatar-one.vercel.app](https://digital-avatar-one.vercel.app/)**.

---

## What It Does

Ask it anything about my background:

- **Publications** — "What have you published on cannabidiol and cancer?", "List all your papers", "What methods did you use in your pharmacogenomic study?"
- **Work experience** — "What companies have you worked for?", "Tell me about your data science background"
- **Projects** — "What AI/ML projects have you built?", "Show me your GitHub projects"
- **Calendar** — "Can we schedule a meeting?"
- **Hobbies** — "What do you do outside of work?"

---

## Architecture

```
React Frontend (Vercel)
        ↓
FastAPI Backend (Render)
        ↓
Guardrails → Classifier → Specialized Agents
                               ↓
              ┌────────────────────────────────┐
              │  General   │  Projects         │
              │  Calendar  │  Hobbies          │
              │  Publications (RAG)            │
              └────────────────────────────────┘
                               ↓
                    ChromaDB Vector Store
                    (27 peer-reviewed papers)
```

The publications agent uses semantic RAG — user queries are embedded and matched against 27 peer-reviewed papers stored in ChromaDB using OpenAI `text-embedding-3-small`. The system handles listing queries, single scientific term lookups, and deep content questions with chunk-level retrieval.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React, deployed on Vercel |
| Backend | FastAPI (Python), deployed on Render |
| Vector Store | ChromaDB (persistent, committed to repo) |
| Embeddings | OpenAI `text-embedding-3-small` |
| LLM | GPT-4o-mini |
| PDF Pipeline | PyMuPDF, pdfplumber, OpenAI vision |
| Classification | GPT-4o-mini zero-shot classifier |

---

## Repository Structure

```
digital-avatar/
├── backend/                  # FastAPI API + RAG pipeline
│   ├── main.py               # App entrypoint + startup
│   ├── agents.py             # Specialized agents
│   ├── classifier.py         # Query routing
│   ├── guardrails.py         # Content filtering
│   ├── src/                  # PDF processing pipeline
│   ├── scripts/              # Local ingest + inspection tools
│   ├── data/                 # JSON data + ChromaDB index
│   └── README.md             # Backend setup + deployment guide
│
├── frontend/                 # React chat interface
│   ├── src/
│   │   ├── App.jsx           # Main chat component
│   │   └── App.css           # Styles
│   └── README.md             # Frontend setup + deployment guide
│
└── README.md                 # This file
```

---

## Quick Start

**Backend:**
```bash
cd backend
conda activate digital_avatar
pip install -r requirements.txt
python main.py
```

**Frontend:**
```bash
cd frontend
npm install
npm start
```

Full setup instructions in [`backend/README.md`](backend/README.md) and [`frontend/README.md`](frontend/README.md).

---

## Publications RAG Pipeline

Academic papers are processed through a 4-step local pipeline before deployment:

```
PDF → text extraction → section parsing → chunking → ChromaDB embedding
```

The resulting vector store is committed to the repo and loaded at startup — no rebuild needed on Render. Adding a new paper takes one command:

```bash
cp "2024 My New Paper.pdf" backend/data/pdfs/
python backend/scripts/ingest_papers.py
```

See [`backend/data/chroma_db_README.md`](backend/chroma_db_README.md) for a full explainer of how the vector store works.

---

## Deployment

| Service | Platform | URL |
|---|---|---|
| Frontend | Vercel | https://digital-avatar-one.vercel.app |
| Backend API | Render | https://your-app.onrender.com |

---

## License

MIT
