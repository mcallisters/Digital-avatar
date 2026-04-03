from llm_client import get_response
import json
import re
from pathlib import Path
from src.vector_store import search_publications, get_all_publications

# Path to data directory
DATA_DIR = Path(__file__).parent / "data"

def load_json_data(filename: str) -> dict:
    """
    Load JSON data from the data directory

    Args:
        filename: Name of the JSON file (e.g., "resume.json")

    Returns:
        dict: Parsed JSON data, or empty dict if file not found
    """
    filepath = DATA_DIR / filename

    try:
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            print(f"Warning: {filename} not found at {filepath}")
            return {}
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return {}


def general_agent(user_message: str, conversation_history: list) -> str:
    """
    Agent for answering general questions about Sean's work experience and resume
    """
    resume_data = load_json_data("resume.json")

    system_message = f"""You are Sean McAllister's digital twin. A user will chat with you and you must respond as if you ARE Sean - use first person ("I", "my", not "he" or "Sean").

You answer general questions about your previous work experience, education, skills, and professional background.

IMPORTANT RULES:
1. Always respond in first person as Sean
2. Keep answers short and conversational
3. Only use information from the context provided below
4. If you don't have specific information to answer a question, respond with: "Interesting question, you'll probably have to ask the real me for that :-)"
5. Do not make up or infer information not in the context

CONTEXT ABOUT SEAN:
{json.dumps(resume_data, indent=2)}
"""

    response = get_response(
        prompt=user_message,
        model="gpt-4o-mini",
        system_message=system_message,
        conversation_history=conversation_history,
        temperature=0.7
    )

    return response


def projects_agent(user_message: str, conversation_history: list) -> str:
    """
    Agent for answering questions about Sean's projects
    """
    projects_data = load_json_data("projects.json")

    system_message = f"""You are Sean McAllister's digital twin. A user will chat with you and you must respond as if you ARE Sean - use first person ("I", "my", not "he" or "Sean").

You answer questions about your projects and applications, including personal projects and community projects on the SuperDataScience Platform.

IMPORTANT RULES:
1. Always respond in first person as Sean
2. Keep answers short and conversational
3. Only use information from the context provided below
4. If you don't have specific information to answer a question, respond with: "Interesting question, you'll probably have to ask the real me for that :-)"
5. Do not make up or infer information not in the context
6. You can mention that more projects are available on your GitHub: https://github.com/mcallisters?tab=repositories

CONTEXT ABOUT SEAN'S PROJECTS:
{json.dumps(projects_data, indent=2)}
"""

    response = get_response(
        prompt=user_message,
        model="gpt-4o-mini",
        system_message=system_message,
        conversation_history=conversation_history,
        temperature=0.7
    )

    return response


# ── Query intent detectors ────────────────────────────────────────────────────

_METHODS_RE = re.compile(
    r"\b(method|protocol|procedure|how\s+(did|was|were)|technique|assay|"
    r"sequencing|pcr|western\s+blot|ihc|elisa|animal\s+stud|statistic|"
    r"software|tool|platform|screen|dose|concentration|cell\s+line|"
    r"performed|conducted|measured|analyzed|quantif)\b",
    re.IGNORECASE,
)

_RESULTS_RE = re.compile(
    r"\b(results?|findings?|outcomes?|show|found|demonstrate|identif|efficacy|"
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

# Detect listing intent — return full catalogue rather than semantic search
_LIST_RE = re.compile(
    r"\b(list|all|every|complete|full list|how many|count|show me all|"
    r"what have you published|all your papers|all publications|"
    r"all your publications|what papers|what are your publications|"
    r"publications do you have|papers do you have)\b",
    re.IGNORECASE,
)


def _route_query(query: str) -> list[str] | None:
    """Route the query to the most relevant chunk types."""
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


def _expand_query(query: str) -> str:
    """
    Expand short or narrow queries into richer descriptions for better
    semantic retrieval. Single words and very short phrases produce narrow
    embedding vectors that may miss relevant chunks — expanding them into
    a descriptive sentence improves recall significantly.

    Only expands queries of 3 words or fewer. Longer queries are returned
    unchanged since they already carry enough semantic signal.
    """
    if len(query.split()) > 3:
        return query

    try:
        expansion = get_response(
            prompt=(
                f"Expand this search query into one descriptive sentence "
                f"suitable for searching scientific research papers. "
                f"Keep it concise and factual. Query: '{query}'"
            ),
            model="gpt-4o-mini",
            temperature=0.0,
        )
        expanded = expansion.strip().strip('"').strip("'")
        print(f"[query expansion] '{query}' → '{expanded}'")
        return f"{query} {expanded}"
    except Exception as e:
        print(f"[query expansion] failed: {e}")
        return query


def publications_agent(user_message: str, conversation_history: list) -> str:
    """
    Agent for answering questions about Sean's publications.

    Uses semantic RAG — retrieves the most relevant chunks from ChromaDB
    and injects them into the prompt. Falls back to publications.json
    summary if the vector store is unavailable.
    """

    # ── Listing intent: return full catalogue from ChromaDB ──────────────────
    if _LIST_RE.search(user_message):
        all_pubs = get_all_publications()
        if all_pubs:
            lines = [f"Here are all {len(all_pubs)} of my publications, listed by year:\n"]
            for p in all_pubs:
                lines.append(
                    f"- **\"{p['title']}\"** ({p['year']}) — DOI: {p['doi']}"
                )
            listing_context = "\n".join(lines)
            system_message = f"""You are Sean McAllister's digital twin. Respond in first person as Sean ("I", "my").
You are listing all of Sean's publications from the vector store.
Present the list clearly and offer to provide details on any specific paper.
IMPORTANT: Always respond in first person as Sean. Never refer to Sean in third person.

{listing_context}"""
            return get_response(
                prompt=user_message,
                model="gpt-4o-mini",
                system_message=system_message,
                conversation_history=conversation_history,
                temperature=0.1,
            )

    # ── Step 1: Retrieve relevant chunks from ChromaDB ────────────────────────
    try:
        search_query = _expand_query(user_message)
        chunk_types  = _route_query(user_message)
        hits = search_publications(query=search_query, n_results=5,
                                   chunk_types=chunk_types)

        # Fallback to broad search if targeted search returned nothing
        if not hits and chunk_types:
            hits = search_publications(query=search_query, n_results=5)

    except Exception as e:
        print(f"[publications_agent] Vector store unavailable: {e}")
        hits = []

    # ── Step 2: Build context ─────────────────────────────────────────────────
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
        context_source = "rag"

    else:
        # Fallback: use publications.json summary (titles + abstracts only)
        publications_data = load_json_data("publications.json")
        all_pubs = publications_data if isinstance(publications_data, list) \
                   else publications_data.get("publications", [])

        summary = [{"title": p.get("title",""), "year": p.get("year",""),
                    "journal": p.get("journal","")} for p in all_pubs]
        context = (
            f"Publications list ({len(all_pubs)} total):\n"
            f"{json.dumps(summary, indent=2)}\n\n"
            f"Full list: https://pubmed.ncbi.nlm.nih.gov/?term=McAllister+SD"
        )
        context_source = "json"

    # ── Step 3: Build system prompt ───────────────────────────────────────────
    if context_source == "rag":
        system_message = f"""You are Sean McAllister's digital twin. Respond in first person as Sean ("I", "we", "my", "our").

You have direct access to Sean's publications through a semantic search system. You CAN and DO search publications — never say otherwise.

You answer questions about your academic publications and research using ONLY the excerpts provided below.

IMPORTANT RULES:
1. Always respond in first person as Sean
2. You HAVE access to Sean's publications — never say you cannot search or access them
3. When asked if you can search publications, confirm you can and immediately demonstrate it with relevant excerpts
4. Be specific and technical — the person asking likely has a research background
5. Cite specific figures, data points, or statistics from the excerpts when relevant
6. If the answer is not in the excerpts, say you don't have that level of detail here
   and suggest checking the full paper via the DOI
7. Do not speculate or invent details not in the excerpts
8. Keep responses conversational but precise

{context}"""

    else:
        system_message = f"""You are Sean McAllister's digital twin. Respond in first person as Sean ("I", "my", "our").

You answer questions about your publications. Start by listing titles, years, and journals.

IMPORTANT RULES:
1. Always respond in first person as Sean
2. List publications concisely (title, year, journal)
3. After listing, ask if they want more details about any specific paper
4. Full list available at: https://pubmed.ncbi.nlm.nih.gov/?term=McAllister+SD

{context}"""

    # ── Step 4: Call LLM ──────────────────────────────────────────────────────
    response = get_response(
        prompt=user_message,
        model="gpt-4o-mini",
        system_message=system_message,
        conversation_history=conversation_history,
        temperature=0.3,
    )

    return response


def calendar_agent(user_message: str, conversation_history: list) -> str:
    """
    Agent for handling calendar and meeting requests
    """
    system_message = """You are Sean McAllister's digital twin. A user will chat with you and you must respond as if you ARE Sean - use first person ("I", "my", not "he" or "Sean").

You help users schedule meetings with Sean.

IMPORTANT INSTRUCTIONS:
1. Always respond in first person as Sean
2. Provide this email: sean.david.mcallister@gmail.com
3. Ask them to propose TWO specific dates and times that work for them
4. Let them know you (Sean) will write back to confirm the meeting or coordinate another
5. Keep your tone friendly and professional

Example response:
"I'd be happy to meet with you! Please email me at sean.david.mcallister@gmail.com with two dates and times that work for you, and I'll get back to you to confirm or coordinate another time that works for both of us."
"""

    response = get_response(
        prompt=user_message,
        model="gpt-4o-mini",
        system_message=system_message,
        conversation_history=conversation_history,
        temperature=0.7
    )

    return response


def hobbies_agent(user_message: str, conversation_history: list) -> str:
    """
    Agent for answering questions about Sean's hobbies and interests outside science
    """
    hobbies_data = load_json_data("hobbies.json")

    system_message = f"""You are Sean McAllister's digital twin. A user will chat with you and you must respond as if you ARE Sean - use first person ("I", "my", not "he" or "Sean").

You answer questions about your hobbies, interests, and pursuits outside of professional/scientific work.

IMPORTANT RULES:
1. Always respond in first person as Sean
2. Keep answers short and conversational
3. Only use information from the context provided below
4. If you don't have specific information to answer a question, respond with: "Interesting question, you'll probably have to ask the real me for that :-)"
5. Do not make up or infer information not in the context

CONTEXT ABOUT SEAN'S HOBBIES:
{json.dumps(hobbies_data, indent=2)}
"""

    response = get_response(
        prompt=user_message,
        model="gpt-4o-mini",
        system_message=system_message,
        conversation_history=conversation_history,
        temperature=0.7
    )

    return response
