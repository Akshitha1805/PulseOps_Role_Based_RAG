import os
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client, Client

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings


# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not SUPABASE_URL:
    raise ValueError("SUPABASE_URL is missing from .env")

if not SUPABASE_KEY:
    raise ValueError("SUPABASE_KEY is missing from .env")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing from .env")


# --------------------------------------------------
# APPROVED APPLICATION ROLES
# --------------------------------------------------

VALID_ROLES = {
    "Admin",
    "Inventory Manager",
    "Procurement Manager",
    "Supplier Manager",
    "Delivery & Operations Manager",
    "Business Performance Manager",
}


# --------------------------------------------------
# SUPABASE CLIENT
# --------------------------------------------------

def create_supabase_client(access_token: str | None = None) -> Client:
    """
    Create a Supabase client.

    Without a token:
        Uses the project anon/publishable key.

    With a token:
        Uses the existing website user's Supabase JWT
        for authenticated database/RPC requests.
    """

    client = create_client(
        SUPABASE_URL,
        SUPABASE_KEY
    )

    if access_token:
        client.postgrest.auth(access_token)

    return client


# --------------------------------------------------
# VERIFY EXISTING WEBSITE USER
# --------------------------------------------------

def verify_user_token(access_token: str):
    """
    Verify the existing Supabase access token.

    The user's identity is obtained from Supabase Auth.
    Never accept a role supplied by the frontend.
    """

    if not access_token:
        raise PermissionError("Authentication token is missing.")

    client = create_supabase_client(access_token)

    response = client.auth.get_user(access_token)

    if not response or not response.user:
        raise PermissionError("Invalid or expired authentication token.")

    return client, response.user


# --------------------------------------------------
# EMBEDDINGS
# --------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# GROQ LLM
# --------------------------------------------------

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=GROQ_API_KEY
)


print("PulseOps RAG configuration loaded successfully.")
