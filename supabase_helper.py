"""
Supabase helper module for storing candidate information securely.

Requires a Supabase table named 'candidates' with the following schema:

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

    -- Enable Row-Level Security (RLS)
    ALTER TABLE candidates ENABLE ROW LEVEL SECURITY;

    -- Policy: allow inserts only via the service/anon key
    CREATE POLICY "Allow insert" ON candidates FOR INSERT WITH CHECK (true);
"""

import os
import streamlit as st
from supabase import create_client, Client


def _get_secret(key: str, default: str = "") -> str:
    """Read a secret from env vars or st.secrets (Streamlit Cloud)."""
    val = os.getenv(key, "")
    if not val:
        try:
            val = st.secrets.get(key, default)
        except Exception:
            val = default
    return val


def _get_client() -> Client:
    """Return an authenticated Supabase client."""
    url = _get_secret("SUPABASE_URL")
    key = _get_secret("SUPABASE_KEY")
    if not url or not key:
        raise EnvironmentError(
            "SUPABASE_URL and SUPABASE_KEY must be set in the environment "
            "or in a .env file."
        )
    return create_client(url, key)


def save_candidate(data: dict) -> dict:
    """
    Persist candidate data to the Supabase 'candidates' table.

    Parameters
    ----------
    data : dict
        Must contain keys: full_name, email, phone, experience,
        position, location, tech_stack.

    Returns
    -------
    dict
        The inserted row returned by Supabase.
    """
    client = _get_client()
    row = {
        "full_name": data["full_name"],
        "email": data["email"],
        "phone": data["phone"],
        "experience": int(data["experience"]),
        "position": data["position"],
        "location": data["location"],
        "tech_stack": data["tech_stack"],
    }
    result = client.table("candidates").insert(row).execute()
    return result.data
