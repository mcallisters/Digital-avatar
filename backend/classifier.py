from llm_client import get_response
import re
import json

# ── Scientific term pre-classifier ───────────────────────────────────────────
# Single or two-word technical terms that are unambiguously publication-related.
# Catches queries like "pharmacogenomics", "EGFR", "CBD glioblastoma" before
# the LLM classifier sees them — the LLM tends to route these to "else" because
# they lack conversational context.

_SCIENCE_TERM_RE = re.compile(
    r"\b(pharmacogenomic|cannabidiol|glioblastoma|oncol|cancer|tumor|tumour|"
    r"melanoma|metastas|immunotherap|checkpoint|EGFR|BRAF|MEK|PDX|PDC|PDXC|"
    r"xenograft|sequencing|NGS|mutation|genomic|proteomic|transcriptomic|"
    r"biomarker|cytotoxic|apoptosis|proliferation|invasion|angiogenesis|"
    r"chemotherapy|immunology|antibody|antigen|cytokine|kinase|inhibitor|"
    r"receptor|ligand|pathway|signaling|preclinical|clinical\s+trial|"
    r"cell\s+line|in\s+vitro|in\s+vivo|orthotopic|xenograft|organoid|"
    r"CRISPR|siRNA|shRNA|PCR|western\s+blot|flow\s+cytometry|ELISA|IHC|"
    r"IC50|EC50|dose.response|survival\s+curve|kaplan.meier)\b",
    re.IGNORECASE,
)


def _is_short_science_query(message: str) -> bool:
    """
    Return True if the message is a short query (<=4 words) containing
    a scientific term — these should always route to publications.
    """
    return len(message.split()) <= 4 and bool(_SCIENCE_TERM_RE.search(message))


# ── Main classifier ───────────────────────────────────────────────────────────

def classify_query(user_message: str) -> str:
    """
    Classify user query into one of the predefined categories.

    Categories:
    - general:       questions about work experience and resume
    - projects:      questions about personal/community projects
    - publications:  questions about papers, articles, research
    - calendar:      questions about schedule, availability, meetings
    - hobbies:       questions about interests outside of science
    - else:          anything that doesn't fit above categories

    Returns:
        str: category name
    """

    # ── Pre-classifier: short scientific terms → publications ─────────────────
    if _is_short_science_query(user_message):
        print(f"Query classified as: publications (science term pre-classifier)")
        return "publications"

    # ── LLM classifier ────────────────────────────────────────────────────────
    classifier_prompt = f"""
You are a classification agent for Sean's Digital Avatar.

Your task is to classify user queries into ONE of the following categories:

1. **general** - Questions about Sean's previous work experience, education, resume, career history, skills, or professional background
   Examples: "What companies have you worked for?", "Tell me about your experience", "What's your educational background?"

2. **projects** - Questions about Sean's personal projects, applications, code repositories, or community projects on SuperDataScience Platform
   Examples: "What projects have you built?", "Show me your GitHub repos", "Tell me about your applications"

3. **publications** - Questions about Sean's scientific papers, research output, articles, or academic publications.
   This includes single scientific terms, drug names, biological concepts, or techniques that relate to research.
   Examples: "What papers have you published?", "Tell me about your research", "Do you have any publications on X?",
   "pharmacogenomics", "cannabidiol and cancer", "EGFR inhibitors", "what do you know about glioblastoma",
   "can you search your publications", "do you have papers on immunotherapy"

4. **calendar** - Questions about Sean's schedule, availability, meeting requests, or events
   Examples: "When are you available?", "Can we schedule a meeting?", "What's your availability next week?"

5. **hobbies** - Questions about Sean's interests, hobbies, or pursuits outside of professional/scientific work
   Examples: "What do you do for fun?", "What are your hobbies?", "What do you like besides work?"

6. **else** - Anything that doesn't clearly fit the above categories, including greetings, off-topic questions, or unclear queries
   Note: Scientific terms and research topics should go to publications, NOT else.

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
        return "else"
