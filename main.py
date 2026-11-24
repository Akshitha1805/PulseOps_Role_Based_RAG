from ingest import ingest_documents

from datetime import datetime, timezone
from typing import Any
from threading import Lock

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import VALID_ROLES, verify_user_token

from src.rag_pipeline import (
    answer_question,
    retrieve_role_documents,
    build_document_context,
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="PulseOps Role-Based RAG API",
    description="Authenticated, role-based knowledge assistant for PulseOps.",
    version="1.0.0",
)


# Development configuration.
# Restrict this to your deployed website's origin before production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)


# Prevent simultaneous ingestion requests in this process.
ingestion_lock = Lock()


# ============================================================
# REQUEST MODELS
# ============================================================

class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=5000)
    conversation_history: Any = None


class RetrievalRequest(BaseModel):
    question: str = Field(min_length=1, max_length=5000)
    match_count: int = Field(default=8, ge=1, le=20)


# ============================================================
# AUTHENTICATION HELPERS
# ============================================================

def get_bearer_token(authorization: str | None) -> str:
    """Extract the existing Supabase access token from Authorization."""

    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Please sign in again.",
        )

    scheme, separator, token = authorization.partition(" ")

    if (
        not separator
        or scheme.lower() != "bearer"
        or not token.strip()
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid authorization header.",
        )

    return token.strip()


def get_approved_user(
    access_token: str,
) -> tuple[Any, Any, str]:
    """
    Verify the Supabase Auth token and retrieve the user's
    approved application role from public.users.

    The role is never accepted from the frontend.
    """

    try:
        supabase_client, auth_user = verify_user_token(
            access_token
        )

        profile_response = (
            supabase_client.table("users")
            .select("id, role, approval_status")
            .eq("id", str(auth_user.id))
            .limit(1)
            .execute()
        )

        profiles = profile_response.data or []

        if not profiles:
            raise HTTPException(
                status_code=403,
                detail="Your application user profile was not found.",
            )

        profile = profiles[0]

        role = profile.get("role")

        approval_status = str(
            profile.get("approval_status") or ""
        ).strip().lower()

        if approval_status != "approved":
            raise HTTPException(
                status_code=403,
                detail="Your account is not approved.",
            )

        if role not in VALID_ROLES:
            raise HTTPException(
                status_code=403,
                detail=(
                    "Your account does not have "
                    "a valid application role."
                ),
            )

        return supabase_client, auth_user, role

    except HTTPException:
        raise

    except Exception:
        # Do not expose tokens, database details, or internal errors.
        raise HTTPException(
            status_code=401,
            detail=(
                "Authentication could not be verified. "
                "Please sign in again."
            ),
        )


def format_conversation_history(history: Any) -> str:
    """Convert supported conversation-history formats into text."""

    if history is None:
        return ""

    if isinstance(history, str):
        return history[-12000:]

    if isinstance(history, list):
        formatted_messages = []

        for message in history[-20:]:

            if not isinstance(message, dict):
                continue

            speaker = str(
                message.get("role")
                or message.get("sender")
                or "user"
            )

            content = message.get("content")

            if content is None:
                content = message.get("text", "")

            if not isinstance(content, str):
                continue

            formatted_messages.append(
                f"{speaker}: {content[:3000]}"
            )

        return "\n".join(formatted_messages)[-12000:]

    return ""


# ============================================================
# ROOT AND HEALTH ENDPOINTS
# ============================================================

@app.get("/")
def root():

    return {
        "message": "PulseOps Role-Based RAG API is running.",
        "status": "ok",
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "service": "PulseOps RAG API",
    }


# ============================================================
# CHAT ENDPOINT
# ============================================================

@app.post("/chat")
def chat(
    request: ChatRequest,
    authorization: str | None = Header(default=None),
):
    """
    Authenticated chatbot endpoint.

    Requires the existing website Supabase access token.

    The database derives document access from auth.uid().
    No role or document filter is accepted from the caller.
    """

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Please enter a question.",
        )

    access_token = get_bearer_token(authorization)

    supabase_client, auth_user, role = get_approved_user(
        access_token
    )

    conversation_history = format_conversation_history(
        request.conversation_history
    )

    try:
        result = answer_question(
            supabase_client=supabase_client,
            question=question,
            role=role,
            conversation_history=conversation_history,
            live_data_context=None,
            match_count=8,
        )

        answer = result["answer"]

        return {
            "reply": answer,
            "answer": answer,
            "role": role,
            "name": getattr(auth_user, "email", None),
            "verified": True,
            "model": "openai/gpt-oss-20b",
            "sources": result["sources"],
            "documents_retrieved": result["documents_retrieved"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except PermissionError:
        raise HTTPException(
            status_code=403,
            detail=(
                "You are not authorized to access this knowledge."
            ),
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception:
        raise HTTPException(
            status_code=500,
            detail=(
                "The assistant could not process your question. "
                "Please try again."
            ),
        )


# ============================================================
# ROLE-BASED RETRIEVAL ENDPOINT
# ============================================================

@app.post("/retrieve")
def retrieve(
    request: RetrievalRequest,
    authorization: str | None = Header(default=None),
):
    """
    Retrieve role-authorized PDF context for the existing
    Node.js Gemini chatbot.

    Authentication and role are independently verified.
    No caller-supplied role or document filter is trusted.
    """

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Please enter a question.",
        )

    access_token = get_bearer_token(authorization)

    supabase_client, auth_user, role = get_approved_user(
        access_token
    )

    try:
        documents = retrieve_role_documents(
            supabase_client=supabase_client,
            question=question,
            match_count=request.match_count,
        )

        context = build_document_context(documents)

        sources = list(
            dict.fromkeys(
                document["source"]
                for document in documents
                if document.get("source")
            )
        )

        return {
            "verified": True,
            "role": role,
            "context": context,
            "sources": sources,
            "documents_retrieved": len(documents),
        }

    except PermissionError:
        raise HTTPException(
            status_code=403,
            detail=(
                "You are not authorized to access this knowledge."
            ),
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception:
        raise HTTPException(
            status_code=500,
            detail=(
                "Document retrieval failed. Please try again."
            ),
        )


# ============================================================
# ADMIN PDF INGESTION ENDPOINT
# ============================================================

@app.post("/admin/ingest")
def admin_ingest(
    authorization: str | None = Header(default=None),
):
    """
    Ingest the seven PDFs using the existing approved Admin session.

    Duplicate checking is performed inside ingest.py.
    Existing documents are not blindly inserted again.
    """

    access_token = get_bearer_token(authorization)

    # Verify the existing website user.
    supabase_client, auth_user, role = get_approved_user(
        access_token
    )

    # Only an approved Admin can initiate ingestion.
    if role != "Admin":
        raise HTTPException(
            status_code=403,
            detail=(
                "Only an approved Admin can ingest "
                "knowledge documents."
            ),
        )

    # Prevent two ingestion requests from running simultaneously.
    if not ingestion_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="Another PDF ingestion process is already running.",
        )

    try:
        print(
            "\nAdmin ingestion request received."
        )

        # ingest.py independently verifies the Admin JWT.
        result = ingest_documents(access_token)

        return {
            "message": "PDF ingestion process completed.",
            "verified": True,
            "role": role,
            "admin_user_id": str(auth_user.id),
            **result,
        }

    except PermissionError:
        raise HTTPException(
            status_code=403,
            detail="Approved Admin access is required.",
        )

    except HTTPException:
        raise

    except FileNotFoundError as error:
        print(f"PDF file error: {error}")

        raise HTTPException(
            status_code=500,
            detail=(
                "A required PDF file could not be found. "
                "Check the backend data folder."
            ),
        )

    except Exception as error:
        # Detailed errors stay in the backend terminal.
        print(
            f"PDF ingestion error: {type(error).__name__}: {error}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "PDF ingestion failed. "
                "Check the backend terminal for details."
            ),
        )

    finally:
        ingestion_lock.release()
