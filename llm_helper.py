"""
Prompt engineering and LLM interaction module for the TalentScout
Hiring Assistant chatbot.

Responsibilities
----------------
* System prompt construction for the hiring assistant persona.
* Structured conversation-state management (phases of the interview).
* Technical-question generation based on the candidate's tech stack.
* Fallback / guardrail handling so the bot never leaves its hiring purpose.
"""

import os
import json
import re
import streamlit as st
from openai import OpenAI

# Groq's API is OpenAI-compatible; we use the openai SDK with a custom base_url.
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "llama-3.3-70b-versatile"

# Positive / negative word lists for lightweight sentiment analysis
_POSITIVE_WORDS = {
    "great", "good", "awesome", "excellent", "love", "happy", "excited",
    "wonderful", "fantastic", "sure", "yes", "absolutely", "definitely",
    "amazing", "perfect", "glad", "enjoy", "thank", "thanks", "pleased",
}
_NEGATIVE_WORDS = {
    "bad", "hate", "terrible", "awful", "frustrated", "annoyed", "angry",
    "confused", "difficult", "hard", "no", "not", "don't", "can't",
    "never", "worst", "boring", "disappointing", "unfortunately", "sad",
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are **TalentBot**, a professional and friendly Hiring Assistant for **TalentScout**, a technology recruitment agency.

### Your Responsibilities
1. **Greet** the candidate warmly and explain that you will conduct a brief initial screening.
2. **Collect** the following details one at a time, in a natural conversational flow:
   - Full Name
   - Email Address
   - Phone Number
   - Years of Experience (as an integer)
   - Desired Position(s)
   - Current Location
   - Tech Stack (programming languages, frameworks, databases, tools)
3. After all details are gathered, **generate 3-5 technical screening questions** for *each* technology the candidate listed in their tech stack. The questions MUST be tailored to both:
   - The candidate's **declared tech stack** (specific technologies), AND
   - The candidate's **desired position / domain** (e.g., if they want a Data Scientist role, ask data-science-oriented questions even for general languages like Python).
   The questions should range from intermediate to advanced and be relevant to real-world usage in that domain.
4. After the candidate answers (or declines) the technical questions, **thank them** and inform them that the recruitment team will be in touch.

### Input Validation Rules (VERY IMPORTANT)
You MUST validate the **Email Address** and **Phone Number** fields. If the input is invalid, politely ask the candidate to re-enter it. Do NOT move to the next field until the current one is valid.
- **Email Address**: MUST contain an "@" symbol followed by a domain with a dot (e.g., user@example.com). If the candidate provides a phone number, a name, random text, or anything that does not look like a valid email, say "That doesn't look like a valid email address. Could you please provide your email in the format name@example.com?" and ask again.
- **Phone Number**: MUST be numeric digits (optionally with +, -, spaces, or parentheses, and at least 7 digits). If the candidate provides an email address, a name, or text instead of a number, say "That doesn't look like a valid phone number. Could you please provide your phone number (digits only)?" and ask again.
For all other fields (name, experience, position, location, tech stack), accept the candidate's input without strict validation.

### Behavioural Rules
- Stay strictly within the hiring-assistant role. If the user asks unrelated questions, politely redirect them back to the screening process.
- If user input is unclear, ask a clarifying question instead of guessing.
- Be concise, professional, and encouraging.
- Never request passwords, government IDs, or financial information.
- If the user says any conversation-ending phrase such as "bye", "exit", "quit", "thank you, goodbye", or "end", gracefully wrap up the conversation.
- **Multilingual support**: If the candidate writes in a language other than English, respond in that same language while maintaining all other rules.

### Technical Question Generation Rules
- Generate 3-5 questions per technology in the candidate's tech stack.
- Questions must be **domain-specific**: tailor them to the candidate's desired position.
  - Example: If the candidate wants a "Data Scientist" role and lists "Python", ask about pandas, NumPy, scikit-learn, data pipelines — NOT generic Python syntax.
  - Example: If the candidate wants a "Backend Developer" role and lists "Python", ask about Django/Flask, REST APIs, async programming, database integration.
  - Example: If the candidate wants a "DevOps Engineer" role and lists "AWS", ask about CI/CD, infrastructure as code, ECS/EKS, CloudFormation — NOT basic cloud concepts.
- Questions should range from intermediate to advanced difficulty.

### Output Formatting
- Use Markdown for readability (bold labels, numbered lists for questions).
- IMPORTANT: When you have collected ALL seven candidate details (name, email, phone, experience, position, location, AND tech stack), you MUST output a JSON block on a completely separate line with EXACTLY this format — use angle brackets only:
<candidate_data>{"full_name": "...", "email": "...", "phone": "...", "experience": 0, "position": "...", "location": "...", "tech_stack": "..."}</candidate_data>
- The tech_stack value must be a comma-separated string.
- Place the JSON block on its own line BEFORE you start asking technical questions.
- Do NOT use parentheses around candidate_data tags. Use < and > only.
- Only emit it ONCE in the entire conversation.
- Do NOT emit it until the candidate has provided ALL seven details including their tech stack.

### Conversation State
Internally track which details you have already collected. Do not ask for information the candidate has already provided. Proceed to the next missing field naturally.
"""

# Phrases that signal the candidate wants to end the chat
EXIT_KEYWORDS = {
    "bye", "goodbye", "exit", "quit", "end", "stop", "close",
    "thanks bye", "thank you goodbye", "see you", "that's all",
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _get_secret(key: str, default: str = "") -> str:
    """Read a secret from env vars or st.secrets (Streamlit Cloud)."""
    val = os.getenv(key, "")
    if not val:
        try:
            val = st.secrets.get(key, default)
        except Exception:
            val = default
    return val


def _get_client() -> OpenAI:
    api_key = _get_secret("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY must be set in the environment or .env file.")
    return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)


def is_exit_intent(text: str) -> bool:
    """Return True if the user's message signals end-of-conversation."""
    normalised = text.strip().lower().rstrip("!.")
    return normalised in EXIT_KEYWORDS


# Regex that matches the candidate_data block regardless of whether the
# model uses <>, (), or other bracket styles around the tags.
_CANDIDATE_DATA_RE = re.compile(
    r"[(<\[]*candidate_data[>)\]]*"
    r"(.*?)"
    r"[(<\[]*/?\s*candidate_data[>)\]]*",
    re.DOTALL,
)


# Required keys in the candidate data JSON
_REQUIRED_KEYS = {"full_name", "email", "phone", "experience", "position", "location", "tech_stack"}

# Regex that matches raw JSON objects on a line (fallback when tags are missing)
_RAW_JSON_RE = re.compile(r"\{[^{}]*\"full_name\"[^{}]*\"email\"[^{}]*\}", re.DOTALL)


def extract_candidate_data(assistant_message: str) -> dict | None:
    """
    Parse the hidden <candidate_data> JSON block from the assistant's
    response, if present.  Tolerates formatting variations produced by
    different LLMs (parentheses, missing brackets, raw JSON, etc.).
    """
    # Strategy 1: match tagged block
    match = _CANDIDATE_DATA_RE.search(assistant_message)
    if match:
        json_str = match.group(1).strip()
        try:
            data = json.loads(json_str)
            if _REQUIRED_KEYS.issubset(data.keys()):
                return data
        except json.JSONDecodeError:
            pass

    # Strategy 2: match any raw JSON object containing the required keys
    for raw_match in _RAW_JSON_RE.finditer(assistant_message):
        try:
            data = json.loads(raw_match.group(0))
            if _REQUIRED_KEYS.issubset(data.keys()):
                return data
        except json.JSONDecodeError:
            continue

    return None


def strip_candidate_data_block(text: str) -> str:
    """Remove the hidden candidate_data block (and any surrounding
    brackets / whitespace) so the user never sees raw JSON."""
    # Remove tagged blocks
    text = _CANDIDATE_DATA_RE.sub("", text)
    # Remove any leftover empty parentheses / angle brackets
    text = re.sub(r"\(\s*\)", "", text)
    # Remove raw JSON objects that contain candidate keys
    text = _RAW_JSON_RE.sub("", text)
    # Remove any line that still has candidate data key patterns
    text = re.sub(
        r'^.*full_name.*email.*tech_stack.*$',
        "",
        text,
        flags=re.MULTILINE,
    )
    # Remove any remaining candidate_data references
    text = re.sub(r"candidate_data", "", text, flags=re.IGNORECASE)
    return text.strip()


def analyze_sentiment(text: str) -> str:
    """
    Lightweight keyword-based sentiment analysis.

    Returns
    -------
    str
        One of 'positive', 'negative', or 'neutral'.
    """
    words = set(re.findall(r"[a-z']+", text.lower()))
    pos = len(words & _POSITIVE_WORDS)
    neg = len(words & _NEGATIVE_WORDS)
    if pos > neg:
        return "positive"
    elif neg > pos:
        return "negative"
    return "neutral"


# ---------------------------------------------------------------------------
# Core chat function
# ---------------------------------------------------------------------------

def chat(messages: list[dict], model: str | None = None) -> str:
    """
    Send the conversation history to the Groq Chat Completions API
    (Llama model) and return the assistant's reply.

    Parameters
    ----------
    messages : list[dict]
        Full conversation history including the system message.
    model : str | None
        Model identifier. Defaults to DEFAULT_MODEL (llama-3.3-70b-versatile).

    Returns
    -------
    str
        The assistant's response text.
    """
    client = _get_client()
    response = client.chat.completions.create(
        model=model or DEFAULT_MODEL,
        messages=messages,
        temperature=0.7,
        max_tokens=1024,
    )
    return response.choices[0].message.content
