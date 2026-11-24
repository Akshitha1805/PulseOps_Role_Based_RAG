import os
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client, Client

from langchain_groq import ChatGroq
import threading
from langchain_core.embeddings import Embeddings


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
# EMBEDDINGS — LAZY-LOADED FASTEMBED / ONNX
# --------------------------------------------------

class LazyFastEmbedEmbeddings(Embeddings):
    """
    Load the existing MiniLM embedding model only when
    document or query embeddings are first requested.

    This avoids loading the model during FastAPI startup.
    """

    def __init__(self):
        self._model = None
        self._model_lock = threading.Lock()

    def _get_model(self):
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    from langchain_community.embeddings import (
                        FastEmbedEmbeddings
                    )

                    self._model = FastEmbedEmbeddings(
                        model_name=(
                            "sentence-transformers/all-MiniLM-L6-v2"
                        ),
                        threads=1,
                        batch_size=16,
                        parallel=None,
                    )

                    print("FastEmbed MiniLM model loaded.")

        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._get_model().embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._get_model().embed_query(text)


embeddings = LazyFastEmbedEmbeddings()


# --------------------------------------------------
# GROQ LLM
# --------------------------------------------------

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=GROQ_API_KEY
)


print("PulseOps RAG configuration loaded successfully.")
