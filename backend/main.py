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
    allow_origins=["*"],  # Update with your Vercel domain in production
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
    agent_map = {
        "general": general_agent,
        "projects": projects_agent,
        "publications": publications_agent,
        "calendar": calendar_agent,
        "hobbies": hobbies_agent
    }
    
    if category == "else":
        return "I appreciate your question, but that's outside my scope. Do you have additional questions about Sean's work experience, projects, publications, calendar availability, and hobbies."
    
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
