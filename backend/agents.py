from llm_client import get_response
import json
from pathlib import Path

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
    # Load resume data
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
    # Load projects data
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

def publications_agent(user_message: str, conversation_history: list) -> str:
    """
    Agent for answering questions about Sean's publications
    Hybrid approach: titles first, full details on request
    """
    # Load publications data
    publications_data = load_json_data("publications.json")
    
    # Handle both list format and dict format
    if isinstance(publications_data, list):
        all_publications = publications_data
    else:
        all_publications = publications_data.get("publications", [])
    
    # Check if user is asking for details about a specific paper
    asking_for_details = any(word in user_message.lower() for word in [
        "tell me more", "details", "about that", "abstract", "methods", 
        "results", "findings", "what did", "how did", "describe", "explain",
        "tell me about", "what was"
    ])
    
    # Try to identify specific paper by matching title keywords
    specific_paper = None
    for pub in all_publications:
        title_lower = pub.get("title", "").lower()
        message_lower = user_message.lower()
        # Get significant words from title (>5 chars, first 4 words)
        title_words = [w for w in title_lower.split() if len(w) > 5][:4]
        # Check if 2+ title words appear in message
        matches = sum(1 for word in title_words if word in message_lower)
        if matches >= 2:
            specific_paper = pub
            break
    
    # Build context based on summary vs details
    if specific_paper and asking_for_details:
        # Provide abstract and basic details only
        paper_details = {
            "title": specific_paper.get("title", ""),
            "authors": specific_paper.get("authors", [])[:10],  # First 10 authors
            "journal": specific_paper.get("journal", ""),
            "year": specific_paper.get("year", ""),
            "pmid": specific_paper.get("pmid", ""),
            "doi": specific_paper.get("doi", ""),
            "abstract": specific_paper.get("abstract", "")
        }
        
        system_message = f"""You are Sean McAllister's digital twin. Respond in first person as Sean ("I", "we", "my", "our").

The user is asking for details about a specific publication. Provide information based on the abstract.

IMPORTANT RULES:
1. Always respond in first person as Sean
2. Discuss the paper's findings and significance conversationally
3. Focus on what's in the abstract
4. If asked about methods, results, or conclusions not in the abstract, mention the full paper is available via PMID or PubMed
5. Keep responses informative but conversational

PUBLICATION DETAILS:
{json.dumps(paper_details, indent=2)}"""
    
    else:
        # Show titles and basic info only
        publications_summary = []
        for pub in all_publications:
            publications_summary.append({
                "title": pub.get("title", ""),
                "year": pub.get("year", ""),
                "journal": pub.get("journal", "")
            })
        
        system_message = f"""You are Sean McAllister's digital twin. Respond in first person as Sean ("I", "my", "our").

You answer questions about your publications. Start by listing titles, years, and journals.

IMPORTANT RULES:
1. Always respond in first person as Sean
2. List publications concisely (title, year, journal)
3. After listing, ASK if they want more details about any specific paper
4. Keep initial responses focused on titles
5. You have {len(all_publications)} publications total
6. Full list: https://pubmed.ncbi.nlm.nih.gov/?term=McAllister+SD&cauthor_id=34850900

PUBLICATIONS (Title, Year, Journal):
{json.dumps(publications_summary, indent=2)}"""
    
    response = get_response(
        prompt=user_message,
        model="gpt-4o",
        system_message=system_message,
        conversation_history=conversation_history,
        temperature=0.7
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
    # Load hobbies data
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
