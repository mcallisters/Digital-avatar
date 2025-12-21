# Sean's Digital Twin - Backend API

A multi-agent AI system that acts as a digital twin, answering questions about work experience, projects, publications, calendar, and hobbies.

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

## Project Structure

```
backend/
├── main.py                 # FastAPI application and routes
├── llm_client.py          # OpenAI client initialization
├── guardrails.py          # Content filtering (moderation, PII, jailbreak)
├── classifier.py          # Query classification agent
├── agents.py              # Specialized agents (general, projects, etc.)
├── data/                  # JSON data files
│   ├── resume.json
│   ├── projects.json
│   ├── publications.json
│   └── hobbies.json
├── requirements.txt       # Python dependencies
└── .env                   # API keys (not committed)
```

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

Create a `.env` file in the root directory:

```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:

```
OPENAI_API_KEY=sk-your-actual-api-key-here
```

### 5. Prepare your data

Replace the sample data in the `data/` directory with your actual information:

- `resume.json` - Your work experience, education, skills
- `projects.json` - Your personal and community projects
- `publications.json` - Your research papers and publications
- `hobbies.json` - Your interests outside of work

You can convert PDFs to JSON using tools like:
- [pdf2json](https://www.npmjs.com/package/pdf2json)
- Custom Python scripts with PyPDF2
- Or manually structure your data

### 6. Run the server locally

```bash
python main.py
```

Or using uvicorn directly:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

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

## Deployment

### Deploy to Render

1. Create a new Web Service on [Render](https://render.com)
2. Connect your GitHub repository
3. Configure:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Add environment variable:
   - Key: `OPENAI_API_KEY`
   - Value: Your OpenAI API key
5. Deploy!

Your API will be available at: `https://your-app-name.onrender.com`

## Testing

Test the API with curl:

```bash
# Health check
curl http://localhost:8000/health

# Chat request
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What projects have you worked on?"}'
```

Or use the interactive docs at `http://localhost:8000/docs`

## Customization

### Modify Agent Behavior

Edit the system messages in `agents.py` to change how each agent responds.

### Add New Categories

1. Add category to classifier in `classifier.py`
2. Create new agent function in `agents.py`
3. Update routing in `main.py`
4. Create corresponding JSON data file

### Adjust Guardrails

Modify detection logic in `guardrails.py`:
- Update jailbreak patterns
- Adjust PII detection sensitivity
- Customize blocked response messages

## Frontend Integration

The backend expects requests from your React frontend. Update CORS origins in `main.py`:

```python
allow_origins=["https://your-vercel-app.vercel.app"]
```

## Notes

- Conversation history is stored in memory (use Redis for production)
- History limited to last 10 messages per session
- All agents use `gpt-4o-mini` by default (configurable in `llm_client.py`)
- Sample data is provided - replace with your actual information

## License

MIT