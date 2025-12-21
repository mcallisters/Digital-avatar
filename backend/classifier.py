from llm_client import get_response
import json

def classify_query(user_message: str) -> str:
    """
    Classify user query into one of the predefined categories
    
    Categories:
    - general: questions about work experience and resume
    - projects: questions about personal/community projects
    - publications: questions about papers, articles, research
    - calendar: questions about schedule, availability, meetings
    - hobbies: questions about interests outside of science
    - else: anything that doesn't fit above categories
    
    Returns:
        str: category name
    """
    
    classifier_prompt = f"""
You are a classification agent for Sean's Digital Avatar.

Your task is to classify user queries into ONE of the following categories:

1. **general** - Questions about Sean's previous work experience, education, resume, career history, skills, or professional background
   Examples: "What companies have you worked for?", "Tell me about your experience", "What's your educational background?"

2. **projects** - Questions about Sean's personal projects, applications, code repositories, or community projects on SuperDataScience Platform
   Examples: "What projects have you built?", "Show me your GitHub repos", "Tell me about your applications"

3. **publications** - Questions about Sean's scientific papers, research output, articles, or academic publications
   Examples: "What papers have you published?", "Tell me about your research", "Do you have any publications on X?"

4. **calendar** - Questions about Sean's schedule, availability, meeting requests, or events
   Examples: "When are you available?", "Can we schedule a meeting?", "What's your availability next week?"

5. **hobbies** - Questions about Sean's interests, hobbies, or pursuits outside of professional/scientific work
   Examples: "What do you do for fun?", "What are your hobbies?", "What do you like besides work?"

6. **else** - Anything that doesn't clearly fit the above categories, including greetings, off-topic questions, or unclear queries

User query: "{user_message}"

Respond with ONLY a JSON object in this exact format:
{{"category": "general"}}

Do not include any explanation or additional text. Only output the JSON.
"""
    
    try:
        response = get_response(
            prompt=classifier_prompt,
            model="gpt-4o-mini",
            temperature=0.0
        )
        
        # Clean up response and parse JSON
        response_clean = response.strip()
        
        # Remove markdown code blocks if present
        if response_clean.startswith("```"):
            response_clean = response_clean.split("```")[1]
            if response_clean.startswith("json"):
                response_clean = response_clean[4:]
            response_clean = response_clean.strip()
        
        # Parse JSON
        result = json.loads(response_clean)
        category = result.get("category", "else").lower()
        
        # Validate category
        valid_categories = ["general", "projects", "publications", "calendar", "hobbies", "else"]
        if category not in valid_categories:
            print(f"Invalid category '{category}', defaulting to 'else'")
            return "else"
        
        print(f"Query classified as: {category}")
        return category
    
    except Exception as e:
        print(f"Classification error: {e}")
        print(f"Response was: {response}")
        # Default to else on errors
        return "else"