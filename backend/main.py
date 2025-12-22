from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uuid

from guardrails import check_guardrails
from classifier import classify_query
from agents import (
    general_agent,
    projects_agent,
    publications_agent,
    calendar_agent,
    hobbies_agent
)

app = FastAPI(title="Sean's Digital Twin API")

# CORS configuration for Vercel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://digital-avatar-one.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory conversation storage (use Redis/DB for production)
conversations = {}

class QueryRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    response: str
    session_id: str
    category: Optional[str] = None

@app.get("/")
async def root():
    return {"message": "Sean's Digital Twin API is running"}

@app.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    """
    Main chat endpoint that orchestrates the entire agent workflow
    """
    try:
        # Generate or retrieve session ID
        session_id = request.session_id or str(uuid.uuid4())
        
        # Initialize conversation history if new session
        if session_id not in conversations:
            conversations[session_id] = []
        
        # Step 1: Guardrails check
        guardrails_result = check_guardrails(request.message)
        if not guardrails_result["passed"]:
            response_text = guardrails_result["message"]
            return QueryResponse(
                response=response_text,
                session_id=session_id,
                category="blocked"
            )
        
        # Step 2: Classify the query
        category = classify_query(request.message)
        
        # Step 3: Route to appropriate agent
        agent_response = route_to_agent(
            category=category,
            user_message=request.message,
            conversation_history=conversations[session_id]
        )
        
        # Step 4: Update conversation history
        conversations[session_id].append({
            "role": "user",
            "content": request.message
        })
        conversations[session_id].append({
            "role": "assistant",
            "content": agent_response
        })
        
        # Limit conversation history to last 10 messages (5 turns)
        if len(conversations[session_id]) > 10:
            conversations[session_id] = conversations[session_id][-10:]
        
        return QueryResponse(
            response=agent_response,
            session_id=session_id,
            category=category
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

def route_to_agent(category: str, user_message: str, conversation_history: list) -> str:
    """
    Routes the classified query to the appropriate specialized agent
    """
    # Handle greetings before classification
    greetings = ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening", "howdy", "yo"]
    message_lower = user_message.lower().strip()
    
    # Check if message is just a greeting (with or without punctuation)
    clean_message = message_lower.rstrip('!.,?')
    if clean_message in greetings:
        return "Hi! I'm Sean's digital twin. I can help you with questions about my work experience, projects, publications, calendar availability, or hobbies. What would you like to know?"
    
    # Handle follow-up responses (yes, tell me more, etc.)
    follow_ups = ["yes", "yeah", "yep", "sure", "ok", "okay", "tell me more", "more details", 
                  "continue", "go on", "please do", "i'd like to know more", "sounds good"]
    
    if clean_message in follow_ups and conversation_history:
        # Look at last assistant message to determine context
        for msg in reversed(conversation_history):
            if msg.get("role") == "assistant":
                content = msg.get("content", "").lower()
                # Check which topic was being discussed
                if "publication" in content or "paper" in content or "research" in content:
                    category = "publications"
                    break
                elif "project" in content or "github" in content or "application" in content:
                    category = "projects"
                    break
                elif "work" in content or "experience" in content or "company" in content:
                    category = "general"
                    break
                elif "meeting" in content or "calendar" in content or "schedule" in content:
                    category = "calendar"
                    break
                elif "hobby" in content or "hobbies" in content or "fun" in content:
                    category = "hobbies"
                    break
    
    agent_map = {
        "general": general_agent,
        "projects": projects_agent,
        "publications": publications_agent,
        "calendar": calendar_agent,
        "hobbies": hobbies_agent
    }
    
    if category == "else":
        return "I appreciate your question, but that's outside my scope. I'm here to discuss Sean's work experience, projects, publications, calendar availability, and hobbies. Feel free to ask me about any of those topics!"
    
    agent_function = agent_map.get(category)
    if not agent_function:
        return "I'm not sure how to help with that. Could you rephrase your question?"
    
    return agent_function(user_message, conversation_history)

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """
    Clear a conversation session
    """
    if session_id in conversations:
        del conversations[session_id]
        return {"message": "Session cleared successfully"}
    return {"message": "Session not found"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
