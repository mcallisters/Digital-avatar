import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# === Load OpenAI API key ===
script_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()
load_dotenv(script_dir / ".env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")

# === Initialize OpenAI client lazily ===
openai_client = None

def get_client():
    """Get or create the OpenAI client instance"""
    global openai_client
    if openai_client is None:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        print("OpenAI client initialized successfully!")
    return openai_client

def get_response(
    prompt: str, 
    model: str = "gpt-4o-mini",
    system_message: str = None,
    conversation_history: list = None,
    temperature: float = 0.7
) -> str:
    """
    Get a response from OpenAI's chat completion API
    
    Args:
        prompt: The user's message/prompt
        model: The model to use (default: gpt-4o-mini)
        system_message: Optional system message to set context
        conversation_history: Optional list of previous messages for context
        temperature: Creativity level (0-2, default 0.7)
    
    Returns:
        The model's response text
    """
    messages = []
    
    # Add system message if provided
    if system_message:
        messages.append({"role": "system", "content": system_message})
    
    # Add conversation history if provided
    if conversation_history:
        messages.extend(conversation_history)
    
    # Add current user prompt
    messages.append({"role": "user", "content": prompt})
    
    try:
        client = get_client()
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        return response.choices[0].message.content
    
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        raise

def get_moderation(text: str) -> dict:
    """
    Check text against OpenAI's moderation API
    
    Args:
        text: The text to check
    
    Returns:
        Dictionary with moderation results
    """
    try:
        client = get_client()
        response = client.moderations.create(input=text)
        result = response.results[0]
        
        return {
            "flagged": result.flagged,
            "categories": result.categories.model_dump(),
            "category_scores": result.category_scores.model_dump()
        }
    
    except Exception as e:
        print(f"Error calling moderation API: {e}")
        # Fail open - don't block on moderation API errors
        return {"flagged": False, "categories": {}, "category_scores": {}}