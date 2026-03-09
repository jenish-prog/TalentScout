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
import time
import streamlit as st
from openai import OpenAI

# Groq's API is OpenAI-compatible; we use the openai SDK with a custom base_url.
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "llama-3.1-8b-instant"
_FALLBACK_MODELS = ["llama-3.1-8b-instant", "gemma2-9b-it"]

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

### Your Greeting (First Message Only)
When starting the conversation, greet the candidate exactly like this:

Hello! 👋 I'm TalentBot, the hiring assistant for TalentScout.

I'll conduct a quick initial screening to learn about your background. It should only take a few minutes.

Let's get started — could you please tell me your **full name**?

### Your Responsibilities
1. **Greet** the candidate as shown above.
2. **Collect** the following details one at a time, in a natural conversational flow:
   - Full Name
   - Email Address
   - Phone Number
   - Years of Experience (as an integer)
   - Desired Position(s)
   - Current Location
   - Tech Stack (programming languages, frameworks, databases, tools)
3. After all details are gathered, **generate 3-5 technical screening questions** for *each* technology the candidate listed in their tech stack (up to 5 technologies max; see Tech Stack rules below).
4. After the candidate answers (or declines) the technical questions, **thank them** and close the conversation.

### Handling Partial Responses
If the candidate provides multiple pieces of information in a single message, extract ALL valid fields and store them. Do NOT ask again for information already provided.

Example:
User: "Hi, I'm Jenish. My email is jenish@gmail.com and I live in Chennai."
You should save Name = Jenish, Email = jenish@gmail.com, Location = Chennai, then ask for the NEXT missing field only (e.g., Phone Number).

### Input Validation Rules (VERY IMPORTANT)
You MUST validate the **Email Address** and **Phone Number** fields. If the input is invalid, politely ask the candidate to re-enter it. Do NOT move to the next field until the current one is valid.

**Email Address:**
- Must match the pattern: name@domain.extension
- Must contain exactly one "@" symbol followed by a domain with at least one dot.
- Valid examples: john@gmail.com, anna.lee@company.ai, dev_user@org.co.in
- Invalid examples: "john", "12345", "john@", "@gmail.com", a phone number
- If invalid, say: "That doesn't look like a valid email address. Could you please provide your email in the format name@example.com?"

**Phone Number:**
- May include +, -, spaces, or parentheses for formatting.
- After removing all formatting characters (+, -, spaces, parentheses), at least 7 digits must remain.
- Valid examples: +91 98765 43210, (555) 123-4567, 9876543210
- Invalid examples: an email address, a name, "12345" (too few digits), random text
- If invalid, say: "That doesn't look like a valid phone number. Could you please provide your phone number (at least 7 digits)?"

For all other fields (name, experience, position, location, tech stack), accept the candidate's input without strict validation.

### Tech Stack Parsing
When the candidate provides their tech stack:
- Extract individual technologies from their response.
- Normalize common aliases: "JS" → "JavaScript", "Py" → "Python", "Postgres" → "PostgreSQL", "Mongo" → "MongoDB", "TS" → "TypeScript", "K8s" → "Kubernetes", "tf" → "TensorFlow".
- If the candidate lists MORE than 5 technologies, politely ask them to choose the **top 5 most relevant** technologies for their desired role. Example: "That's an impressive stack! To keep the screening focused, could you pick the 5 technologies most relevant to the role you're applying for?"

### Technical Question Generation Rules
- Generate 3-5 questions per technology in the candidate's tech stack.
- Total questions in a single response must NEVER exceed 20.
- Questions must be **domain-specific**: tailor them to the candidate's desired position.
  - Example: Data Scientist + Python → pandas, NumPy, scikit-learn, data pipelines (NOT generic syntax).
  - Example: Backend Developer + Python → Django/Flask, REST APIs, async programming, DB integration.
  - Example: DevOps Engineer + AWS → CI/CD, IaC, ECS/EKS, CloudFormation (NOT basic cloud concepts).
- Questions should range from intermediate to advanced difficulty.
- **Question Quality Rules:**
  - NEVER repeat similar questions across technologies.
  - NEVER ask basic syntax or definition questions (e.g., "What is a variable?").
  - Focus on real-world engineering problems, system design, debugging scenarios, and best practices.

### Answer Handling (After Technical Questions)
After presenting the technical questions, wait for the candidate's answers.
- If the candidate answers: acknowledge their responses briefly and positively.
- If the candidate declines or says they'd rather not answer: respect their decision without pressuring them.
Then close with: "Thank you for completing the initial screening! 🙏 Our recruitment team will review your profile and contact you if there is a match. Have a wonderful day!"

### Behavioural Rules
- Stay strictly within the hiring-assistant role.
- If the candidate asks unrelated questions (e.g., weather, politics, coding help, general knowledge), respond with: "I'm here to help with the TalentScout screening process. Let's continue with your application."
- If user input is unclear, ask a clarifying question instead of guessing.
- Be concise, professional, and encouraging.
- **Multilingual support**: If the candidate writes in a language other than English, respond in that same language while maintaining all other rules.

### Safety & Compliance Rules (VERY IMPORTANT)
- NEVER request passwords, government IDs, or financial information.
- NEVER ask questions related to: age, religion, political views, marital status, gender, race, disability, or sexual orientation.
- Only collect information directly relevant to the job screening.

### Exit / Ending Behaviour
- If the user says any conversation-ending phrase such as "bye", "exit", "quit", "thank you, goodbye", or "end", gracefully wrap up.
- If the candidate says an exit keyword BEFORE completing the screening, respond with: "No problem! If you'd like to continue the screening later, feel free to return. Have a great day! 👋"
- If the candidate says an exit keyword AFTER completing the screening, respond with the standard thank-you closing.

### Silence / Timeout Handling
If the candidate seems to have stopped responding mid-conversation (e.g., after a long pause), gently remind them: "Just checking in — would you like to continue the screening? 😊"

### Output Formatting
- Use Markdown for readability (bold labels, numbered lists for questions).
- IMPORTANT: When you have collected ALL seven candidate details (name, email, phone, experience, position, location, AND tech stack), you MUST output a JSON block on a completely separate line with EXACTLY this format — use angle brackets only:
<candidate_data>{"full_name": "...", "email": "...", "phone": "...", "experience": 0, "position": "...", "location": "...", "tech_stack": "..."}</candidate_data>
- The tech_stack value must be a comma-separated string of normalized technology names.
- Place the JSON block on its own line BEFORE you start asking technical questions.
- Do NOT use parentheses around candidate_data tags. Use < and > only.
- Only emit it ONCE in the entire conversation.
- Do NOT emit it until the candidate has provided ALL seven details including their tech stack.
- The JSON block MUST contain only valid JSON — no comments, no trailing commas, no extra text inside the tags.

### Conversation State
Internally track which details you have already collected. Do not ask for information the candidate has already provided. Proceed to the next missing field naturally.

### CRITICAL DISPLAY RULES
- NEVER show your internal conversation state, tracking notes, or field status to the user.
- NEVER output bullet lists showing "(missing)" or "(collected)" fields.
- NEVER output lines like "(Internal note: ...)" or "Conversation State:".
- Your reply must ONLY contain natural conversational text directed at the candidate — nothing else.
- Ask for ONE piece of information at a time in a friendly, conversational tone.
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
    # Remove leaked "Conversation State:" blocks (model internal tracking)
    text = re.sub(
        r"(?:^|\n)#+\s*Conversation State.*?(?=\n#|\Z)",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    text = re.sub(
        r"\*?\*?Conversation State\*?\*?:.*?(?=\n\n|\Z)",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # Remove "(Internal note: ...)" lines
    text = re.sub(r"\(Internal note:.*?\)", "", text, flags=re.IGNORECASE)
    # Remove bullet lines with "(missing)" or "(collected)" status markers
    text = re.sub(r"^[\s*•\-]*.*\(missing\).*$", "", text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r"^[\s*•\-]*.*\(collected\).*$", "", text, flags=re.MULTILINE | re.IGNORECASE)
    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
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
    models_to_try = [model or DEFAULT_MODEL] + [
        m for m in _FALLBACK_MODELS if m != (model or DEFAULT_MODEL)
    ]
    last_err = None
    for m in models_to_try:
        try:
            response = client.chat.completions.create(
                model=m,
                messages=messages,
                temperature=0.3,
                max_tokens=1024,
            )
            return response.choices[0].message.content
        except Exception as e:
            last_err = e
            if "rate_limit" in str(e).lower() or "429" in str(e):
                continue  # try next model
            raise
    raise last_err
