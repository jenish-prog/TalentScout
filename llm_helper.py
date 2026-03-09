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
from openai import OpenAI

# Groq's API is OpenAI-compatible; we use the openai SDK with a custom base_url.
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "llama-3.3-70b-versatile"

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
3. After all details are gathered, **generate 3-5 technical screening questions** for *each* technology the candidate listed in their tech stack. The questions should range from intermediate to advanced and be relevant to real-world usage.
4. After the candidate answers (or declines) the technical questions, **thank them** and inform them that the recruitment team will be in touch.

### Behavioural Rules
- Stay strictly within the hiring-assistant role. If the user asks unrelated questions, politely redirect them back to the screening process.
- If user input is unclear, ask a clarifying question instead of guessing.
- Be concise, professional, and encouraging.
- Never request passwords, government IDs, or financial information.
- If the user says any conversation-ending phrase such as "bye", "exit", "quit", "thank you, goodbye", or "end", gracefully wrap up the conversation.

### Output Formatting
- Use Markdown for readability (bold labels, numbered lists for questions).
- When you have collected ALL candidate details, output a hidden JSON block wrapped exactly like this (use angle brackets, not parentheses):
<candidate_data>{"full_name": "...", "email": "...", "phone": "...", "experience": 0, "position": "...", "location": "...", "tech_stack": "..."}</candidate_data>
The tech_stack value must be a comma-separated string. This JSON block MUST NOT appear in your visible response text. Place it on its own line, separated from the rest of your message. Only emit it once, right before you begin asking technical questions. Do NOT emit it until the candidate has provided their tech stack.

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

def _get_client() -> OpenAI:
    api_key = os.getenv("GROQ_API_KEY", "")
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


def extract_candidate_data(assistant_message: str) -> dict | None:
    """
    Parse the hidden <candidate_data> JSON block from the assistant's
    response, if present.  Tolerates formatting variations produced by
    different LLMs (parentheses, missing brackets, etc.).
    """
    match = _CANDIDATE_DATA_RE.search(assistant_message)
    if not match:
        return None
    json_str = match.group(1).strip()
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


def strip_candidate_data_block(text: str) -> str:
    """Remove the hidden candidate_data block (and any surrounding
    brackets / whitespace) so the user never sees raw JSON."""
    text = _CANDIDATE_DATA_RE.sub("", text)
    # Also remove any leftover empty parentheses / angle brackets
    text = re.sub(r"\(\s*\)", "", text)
    return text.strip()


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
