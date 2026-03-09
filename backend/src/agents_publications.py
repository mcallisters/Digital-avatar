"""
src/agents_publications.py
─────────────────────────────────────────────────────────────────────────────
Publications agent for the digital twin backend.

Drop-in replacement for publications_agent() in your existing agents.py.
Uses semantic retrieval from ChromaDB instead of loading the full
publications.json into the prompt.

Integration
───────────
1. Add to main.py startup (inside lifespan or startup event):
       from src.vector_store import init_vector_store
       init_vector_store()

2. In agents.py, replace the publications_agent() function with the one here.
   Import it at the top of agents.py:
       from src.agents_publications import publications_agent

Query routing
─────────────
Automatically routes to the most relevant chunk types:
  "how did you..." / "what methods..."  → methods only
  "what did you find..." / "results..." → results + discussion
  "what is the study about..."          → overview + introduction
  ambiguous / mixed                     → all types
"""

import re
from src.vector_store import search_publications


# ── Query intent detector ─────────────────────────────────────────────────────

_METHODS_RE = re.compile(
    r"\b(method|protocol|procedure|how\s+(did|was|were)|technique|assay|"
    r"sequencing|pcr|western\s+blot|ihc|elisa|animal\s+stud|statistic|"
    r"software|tool|platform|screen|dose|concentration|cell\s+line|"
    r"performed|conducted|measured|analyzed|quantif)\b",
    re.IGNORECASE,
)

_RESULTS_RE = re.compile(
    r"\b(result|finding|outcome|show|found|demonstrate|identif|efficacy|"
    r"response|synergy|combination|tumor|expression|significant|p.value|"
    r"figure|fold.change|upreg|downreg|pathway|data|observed|increased|"
    r"decreased|compared|higher|lower|greater)\b",
    re.IGNORECASE,
)

_OVERVIEW_RE = re.compile(
    r"\b(about|summary|overview|abstract|background|hypothesis|goal|aim|"
    r"purpose|what\s+is|describe\s+the|tell\s+me\s+about|focus\s+of)\b",
    re.IGNORECASE,
)


def _route_query(query: str) -> list[str] | None:
    """
    Return chunk_types to filter by, or None for a broad search.
    """
    is_methods  = bool(_METHODS_RE.search(query))
    is_results  = bool(_RESULTS_RE.search(query))
    is_overview = bool(_OVERVIEW_RE.search(query))

    if is_methods and not is_results:
        return ["methods"]
    if is_results and not is_methods:
        return ["results", "discussion"]
    if is_overview and not is_methods and not is_results:
        return ["overview", "introduction"]
    return None   # broad search


# ── Publications agent ────────────────────────────────────────────────────────

def publications_agent(user_query: str, conversation_history: list) -> str:
    """
    Answer questions about Sean's publications using semantic RAG.

    Args:
        user_query           : the user's natural language question
        conversation_history : list of {"role": ..., "content": ...} dicts

    Returns:
        Answer string.
    """

    # ── Retrieve relevant chunks ──────────────────────────────────────────────
    chunk_types = _route_query(user_query)
    hits = search_publications(query=user_query, n_results=5, chunk_types=chunk_types)

    # Fallback to broad search if targeted retrieval returned nothing
    if not hits and chunk_types:
        hits = search_publications(query=user_query, n_results=5)

    # ── Format context ────────────────────────────────────────────────────────
    if hits:
        context_lines = ["Relevant excerpts from Sean's publications:\n"]
        for i, hit in enumerate(hits, 1):
            context_lines.append(
                f"[{i}] \"{hit['title']}\" ({hit['year']}) — "
                f"{hit['chunk_type'].upper()} | DOI: {hit['doi']}"
            )
            context_lines.append(f"    {hit['content'][:1200]}")
            context_lines.append("")
        context = "\n".join(context_lines)
    else:
        context = "No matching publication excerpts found in the database."

    # ── Build system prompt ───────────────────────────────────────────────────
    system_prompt = f"""You are Sean McAllister's digital twin, answering questions \
about his academic publications and research at CPMC Research Institute.

Use ONLY the excerpts provided below to answer. Be specific and technical — \
the person asking likely has a research background.

Guidelines:
- If the answer is in the excerpts, answer directly and cite specific figures, \
data points, or statistics mentioned
- If the answer is not in the excerpts, say so clearly — do not speculate
- For methods questions, be precise about reagents, conditions, and instruments
- For results questions, include effect sizes, p-values, and key comparisons \
if present in the excerpts
- Suggest the DOI for the full paper if more detail is needed

{context}"""

    # ── Call LLM ─────────────────────────────────────────────────────────────
    messages = [{"role": "system", "content": system_prompt}]

    for turn in conversation_history[-6:]:   # last 3 exchanges for follow-up support
        messages.append(turn)

    messages.append({"role": "user", "content": user_query})

    response = client.chat.completions.create(   # `client` from your agents.py
        model       = "gpt-4o-mini",
        messages    = messages,
        temperature = 0.2,
        max_tokens  = 700,
    )

    return response.choices[0].message.content
