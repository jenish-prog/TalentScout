"""
TalentScout Hiring Assistant — Streamlit Application

Run with:
    streamlit run app.py
"""

import streamlit as st
from dotenv import load_dotenv

from llm_helper import (
    SYSTEM_PROMPT,
    chat,
    extract_candidate_data,
    is_exit_intent,
    strip_candidate_data_block,
    analyze_sentiment,
)
from supabase_helper import save_candidate

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()  # Load .env for API keys

st.set_page_config(
    page_title="TalentScout Hiring Assistant",
    page_icon="🤖",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Custom CSS for a polished look
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Header banner */
    .header-banner {
        background: linear-gradient(135deg, #1e3a5f 0%, #4a90d9 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
        text-align: center;
    }
    .header-banner h1 { margin: 0; font-size: 2rem; }
    .header-banner p  { margin: 0.3rem 0 0; opacity: 0.9; font-size: 1rem; }

    /* Chat message styling tweaks */
    .stChatMessage { border-radius: 12px; }

    /* Sidebar info card */
    .info-card {
        background: #1e293b;
        border-left: 4px solid #4a90d9;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        margin-bottom: 0.8rem;
        font-size: 0.9rem;
        color: #e2e8f0;
    }

    /* Sentiment badge */
    .sentiment-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 0.3rem;
    }
    .sentiment-positive { background: #166534; color: #bbf7d0; }
    .sentiment-negative { background: #991b1b; color: #fecaca; }
    .sentiment-neutral  { background: #374151; color: #d1d5db; }

    /* Progress bar container */
    .progress-container {
        background: #1e293b;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.8rem;
    }
    .progress-container p {
        color: #e2e8f0;
        margin: 0 0 0.4rem 0;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="header-banner">
        <h1>🤖 TalentScout Hiring Assistant</h1>
        <p>Your AI-powered initial screening companion</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/artificial-intelligence.png",
        width=80,
    )
    st.markdown("### About")
    st.markdown(
        '<div class="info-card">'
        "TalentScout's Hiring Assistant helps streamline the initial "
        "candidate screening process. Answer a few questions and we'll "
        "assess your technical proficiency."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("### How it works")
    st.markdown(
        '<div class="info-card">'
        "1️⃣ Share your basic details<br>"
        "2️⃣ Tell us your tech stack<br>"
        "3️⃣ Answer a few technical questions<br>"
        "4️⃣ Our team reviews & follows up"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Screening progress indicator
    st.markdown("### Screening Progress")

    st.markdown("---")

    # Sentiment analysis display
    st.markdown("### Candidate Mood")

    if st.button("🔄 Start New Conversation"):
        st.session_state.clear()
        st.rerun()

    st.caption("© 2026 TalentScout · Privacy-first screening")

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    # Conversation history sent to the LLM (includes system prompt)
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    # Display history shown to the user (excludes system prompt)
    st.session_state.display_messages = []
    # Whether candidate data has been saved already
    st.session_state.candidate_saved = False
    # Whether the conversation has ended
    st.session_state.ended = False
    # Sentiment tracking
    st.session_state.sentiment_history = []

# ---------------------------------------------------------------------------
# Sidebar dynamic content (after session state init)
# ---------------------------------------------------------------------------
with st.sidebar:
    # Progress bar
    if st.session_state.candidate_saved:
        st.success("✅ Details collected & saved!")
    elif st.session_state.ended:
        st.info("Conversation ended.")
    else:
        msg_count = len([m for m in st.session_state.display_messages if m["role"] == "user"])
        # Approximate: 7 fields to collect, so ~7 user messages before tech questions
        progress = min(msg_count / 7, 1.0)
        st.progress(progress, text=f"Info gathering: {int(progress * 100)}%")

    # Sentiment display
    if st.session_state.sentiment_history:
        latest = st.session_state.sentiment_history[-1]
        emoji = {"positive": "😊", "negative": "😟", "neutral": "😐"}[latest]
        color_class = f"sentiment-{latest}"
        st.markdown(
            f'<span class="sentiment-badge {color_class}">{emoji} {latest.capitalize()}</span>',
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Render existing chat history
# ---------------------------------------------------------------------------
for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# Generate initial greeting on first load
# ---------------------------------------------------------------------------
if len(st.session_state.display_messages) == 0 and not st.session_state.ended:
    with st.chat_message("assistant"):
        with st.spinner("Starting up…"):
            greeting = chat(st.session_state.messages)
            greeting_clean = strip_candidate_data_block(greeting)
        st.markdown(greeting_clean)
    st.session_state.messages.append({"role": "assistant", "content": greeting})
    st.session_state.display_messages.append(
        {"role": "assistant", "content": greeting_clean}
    )

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
if prompt := st.chat_input(
    "Type your response here…" if not st.session_state.ended else "Conversation ended.",
    disabled=st.session_state.ended,
):
    # Sentiment analysis on user input
    sentiment = analyze_sentiment(prompt)
    st.session_state.sentiment_history.append(sentiment)

    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.display_messages.append({"role": "user", "content": prompt})

    # Check for exit intent
    if is_exit_intent(prompt):
        farewell = (
            "Thank you for your time! 🙏 The TalentScout recruitment team "
            "will review your information and get back to you shortly. "
            "Have a wonderful day!"
        )
        with st.chat_message("assistant"):
            st.markdown(farewell)
        st.session_state.messages.append({"role": "assistant", "content": farewell})
        st.session_state.display_messages.append(
            {"role": "assistant", "content": farewell}
        )
        st.session_state.ended = True
        st.rerun()

    # Get assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            raw_reply = chat(st.session_state.messages)

        # Extract & save candidate data if present
        if not st.session_state.candidate_saved:
            candidate_data = extract_candidate_data(raw_reply)
            if candidate_data:
                try:
                    save_candidate(candidate_data)
                    st.session_state.candidate_saved = True
                    st.toast("✅ Candidate details saved!", icon="✅")
                except Exception as exc:
                    st.error(f"⚠️ Could not save to database: {exc}")

        # Strip hidden JSON and display clean reply
        clean_reply = strip_candidate_data_block(raw_reply)
        st.markdown(clean_reply)

    st.session_state.messages.append({"role": "assistant", "content": raw_reply})
    st.session_state.display_messages.append(
        {"role": "assistant", "content": clean_reply}
    )
    st.rerun()
