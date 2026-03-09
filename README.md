# TalentScout Hiring Assistant 🤖

An intelligent, AI-powered Hiring Assistant chatbot built for **TalentScout**, a technology recruitment agency. The chatbot conducts initial candidate screenings by gathering essential information and generating tailored technical questions based on each candidate's declared tech stack.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Architecture](#architecture)
4. [Tech Stack](#tech-stack)
5. [Installation](#installation)
6. [Supabase Setup](#supabase-setup)
7. [Usage Guide](#usage-guide)
8. [Prompt Design](#prompt-design)
9. [Challenges & Solutions](#challenges--solutions)
10. [Future Enhancements](#future-enhancements)

---

## Project Overview

TalentScout's Hiring Assistant automates the first stage of the recruitment pipeline. When a candidate starts a conversation, the chatbot:

1. **Greets** them and explains the screening process.
2. **Collects** personal and professional details (name, email, phone, experience, desired position, location, tech stack).
3. **Generates** 3-5 technical screening questions per technology in the candidate's stack.
4. **Stores** candidate information securely in a **Supabase** (PostgreSQL) database.
5. **Ends** the conversation gracefully, informing the candidate about next steps.

The entire interaction is context-aware — the chatbot tracks which fields have been collected and never repeats questions.

---

## Features

| Feature | Description |
|---|---|
| **Conversational Screening** | Natural, multi-turn dialogue that collects candidate info one question at a time |
| **Dynamic Tech Questions** | Generates intermediate-to-advanced questions tailored to each declared technology |
| **Supabase Persistence** | All candidate data is stored in a PostgreSQL database via Supabase |
| **Context Handling** | Full conversation history is maintained so the bot never loses track |
| **Fallback / Guardrails** | Off-topic inputs are politely redirected back to the screening process |
| **Exit Detection** | Keywords like "bye", "exit", "quit" gracefully end the session |
| **Privacy-First Design** | No passwords, IDs, or financial data are requested; data handling follows GDPR principles |
| **Polished UI** | Custom-styled Streamlit interface with sidebar, progress bar, and "New Conversation" button |
| **Sentiment Analysis** | Keyword-based sentiment detection tracks candidate mood (positive/neutral/negative) with a live sidebar badge |
| **Multilingual Support** | The bot detects the candidate's language and responds in the same language automatically |
| **Screening Progress Bar** | Visual progress indicator in the sidebar showing how far along the info-gathering phase is |

---

## Architecture

```
┌──────────────────────────────────┐
│          Streamlit UI            │  ← app.py
│   (chat input / message display) │
└────────────┬─────────────────────┘
             │  user message
             ▼
┌──────────────────────────────────┐
│        LLM Helper Module         │  ← llm_helper.py
│  • System prompt construction    │
│  • Exit-intent detection         │
│  • Candidate-data extraction     │
│  • OpenAI Chat Completions call  │
└────────────┬─────────────────────┘
             │  assistant reply
             ▼
┌──────────────────────────────────┐
│      Supabase Helper Module      │  ← supabase_helper.py
│  • Authenticated client creation │
│  • Candidate row insertion       │
└──────────────────────────────────┘
```

**Data Flow:**

1. User types a message in the Streamlit chat input.
2. The full conversation history (including the system prompt) is sent to OpenAI's Chat Completions API via `llm_helper.chat()`.
3. The assistant's reply is scanned for a hidden `<candidate_data>` JSON block. If found, the data is persisted to Supabase via `supabase_helper.save_candidate()`.
4. The hidden JSON block is stripped before showing the reply to the user.

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| Frontend | Streamlit 1.41 |
| LLM | Llama 3.3 70B via Groq API (OpenAI-compatible, using `openai` Python SDK) |
| Database | Supabase (PostgreSQL) |
| Environment | python-dotenv for secret management |

---

## Installation

### Prerequisites

- Python 3.10 or higher
- A [Groq API key](https://console.groq.com/keys) (free tier available)
- A [Supabase project](https://supabase.com/) (free tier works)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/talentscout-hiring-assistant.git
cd talentscout-hiring-assistant

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env and fill in your real keys:
#   GROQ_API_KEY=gsk_...
#   SUPABASE_URL=https://xxxx.supabase.co
#   SUPABASE_KEY=eyJ...

# 5. Run the application
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## Supabase Setup

1. Create a new project at [supabase.com](https://supabase.com/).
2. Open the **SQL Editor** and run:

```sql
CREATE TABLE candidates (
    id          UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    full_name   TEXT NOT NULL,
    email       TEXT NOT NULL,
    phone       TEXT NOT NULL,
    experience  INTEGER NOT NULL,
    position    TEXT NOT NULL,
    location    TEXT NOT NULL,
    tech_stack  TEXT NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT now()
);

-- Enable Row-Level Security
ALTER TABLE candidates ENABLE ROW LEVEL SECURITY;

-- Allow inserts from the application
CREATE POLICY "Allow insert" ON candidates FOR INSERT WITH CHECK (true);
```

3. Copy the **Project URL** and **anon/public key** from *Settings → API* and paste them into your `.env` file.

---

## Usage Guide

1. **Start** the app (`streamlit run app.py`). The chatbot greets you automatically.
2. **Provide your details** when prompted — name, email, phone, experience, position, location, and tech stack.
3. **Answer technical questions** generated based on your tech stack.
4. **End the conversation** naturally or type "bye" / "exit" at any time.
5. Click **🔄 Start New Conversation** in the sidebar to reset.

### Screenshot Flow

```
Greeting  →  Info Collection  →  Tech Questions  →  Thank You
```

---

## Prompt Design

The system prompt (`SYSTEM_PROMPT` in `llm_helper.py`) was carefully crafted with several strategies:

### 1. Persona & Scope Constraint
The model is assigned the persona **TalentBot** with an explicit list of responsibilities and strict behavioural rules. This prevents it from answering off-topic questions or deviating from its hiring purpose.

### 2. Structured Data Extraction
The prompt instructs the model to emit a hidden `<candidate_data>` JSON block once all details are collected. This block is programmatically parsed by the application and never shown to the user, creating a clean separation between display output and structured data.

### 3. Single-Field-at-a-Time Collection
Rather than dumping all questions at once, the prompt directs the model to ask for one detail at a time, creating a natural conversational rhythm.

### 4. Tech-Stack Adaptive Questions
The model is instructed to generate 3-5 questions **per technology** in the candidate's stack, ranging from intermediate to advanced. This ensures diverse and relevant coverage regardless of the stack.

### 5. Guardrails
- **Off-topic redirection**: the model is told to politely redirect if the user goes off-topic.
- **Sensitive data refusal**: the model is explicitly forbidden from requesting passwords, IDs, or financial data.
- **Graceful exit**: exit-intent keywords are detected both at the application level and at the prompt level.

---

## Challenges & Solutions

| Challenge | Solution |
|---|---|
| **Extracting structured data from free-form LLM output** | Introduced a hidden `<candidate_data>` JSON block convention in the system prompt, with deterministic parsing in Python. |
| **Preventing the bot from going off-topic** | Added strict behavioural rules in the system prompt and an application-level exit-keyword detector. |
| **Handling diverse tech stacks** | The prompt is technology-agnostic — the LLM's broad training data covers virtually any technology, so no hard-coded question banks are needed. |
| **Conversation context management** | The full message history is sent with every API call, ensuring the model has complete context. Streamlit's `session_state` persists this across reruns. |
| **Sensitive data compliance** | The system prompt explicitly prohibits requesting sensitive information. Data stored in Supabase uses Row-Level Security, and environment secrets are never committed to version control. |

---

## Bonus Features Implemented

- **Sentiment Analysis** — Lightweight keyword-based sentiment detection tracks whether the candidate's mood is positive, neutral, or negative. A live emoji badge in the sidebar updates after each user message.
- **Multilingual Support** — The system prompt instructs the model to detect the candidate's language and respond in the same language, enabling seamless multilingual conversations.
- **Screening Progress Bar** — A visual progress bar in the sidebar shows how far along the info-gathering phase is, providing real-time feedback.
- **Enhanced UI** — Custom CSS styling including gradient header, dark-themed sidebar cards, sentiment badges, and responsive layout.

---

## Future Enhancements

- **Resume Upload** — Parse uploaded resumes to pre-fill candidate details.
- **Admin Dashboard** — A Streamlit page for recruiters to view and filter stored candidates.
- **Cloud Deployment** — Deploy on AWS / GCP / Streamlit Cloud for a live demo link.
- **Advanced Sentiment** — Use an LLM-based sentiment model for more nuanced emotion detection.

---

## License

This project is developed as an assignment / demo and is provided as-is.

---

*Built with ❤️ using Streamlit & Groq (Llama 3.3)*
