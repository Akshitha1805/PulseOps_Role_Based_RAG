import json
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, BaseMessage

from src.config import VALID_ROLES, embeddings, llm

from src.prompts import (
    intent_prompt,
    contextualize_prompt,
    answer_prompt,
    live_data_prompt,
    combined_answer_prompt,
    history_prompt,
    general_prompt,
    out_of_scope_prompt,
)


# ============================================================
# 1. CONSTANTS
# ============================================================

VALID_INTENTS = {
    "DOCUMENT",
    "LIVE_DATA",
    "BOTH",
    "HISTORY",
    "GENERAL",
    "OUT_OF_SCOPE",
}

NO_DOCUMENT_ANSWER = (
    "I don't know based on the available knowledge documents."
)

NO_LIVE_DATA_ANSWER = (
    "I don't have the required live operational data "
    "to answer that question."
)

NO_HISTORY_ANSWER = (
    "That information is not available in the conversation history."
)

OUT_OF_SCOPE_ANSWER = (
    "I can help with PulseOps documents, company policies "
    "and operational information. I don't have support for "
    "that question outside the PulseOps scope."
)


# ============================================================
# 2. CONVERSATION HISTORY NORMALIZATION
# ============================================================

def normalize_conversation_history(
    conversation_history: Any,
) -> list[BaseMessage]:
    """
    Convert supplied conversation history into LangChain messages.

    Supports:
    - A list of HumanMessage / AIMessage objects.
    - A list of dictionaries containing role/content.
    - A string containing User:/Human: and Assistant:/AI: lines.
    - A JSON string containing a list of message dictionaries.

    The history is used for contextualization and HISTORY questions.
    It is never treated as an independent factual source.
    """

    if not conversation_history:
        return []

    if isinstance(conversation_history, list):
        messages = []

        for item in conversation_history:
            if isinstance(item, BaseMessage):
                if isinstance(item, (HumanMessage, AIMessage)):
                    messages.append(item)
                continue

            if isinstance(item, dict):
                role = str(
                    item.get("role", item.get("type", ""))
                ).lower()

                content = item.get(
                    "content",
                    item.get("message", ""),
                )

                if not isinstance(content, str) or not content.strip():
                    continue

                if role in {"user", "human"}:
                    messages.append(HumanMessage(content=content))

                elif role in {"assistant", "ai"}:
                    messages.append(AIMessage(content=content))

        return messages

    if isinstance(conversation_history, str):
        history_text = conversation_history.strip()

        if not history_text:
            return []

        # Support JSON-formatted conversation history.
        try:
            parsed = json.loads(history_text)

            if isinstance(parsed, list):
                return normalize_conversation_history(parsed)

        except (json.JSONDecodeError, TypeError):
            pass

        # Support text such as:
        # User: What is the leave policy?
        # Assistant: The policy states...
        messages = []

        for line in history_text.splitlines():
            line = line.strip()

            if not line:
                continue

            match = re.match(
                r"^(User|Human|Assistant|AI)\s*:\s*(.*)$",
                line,
                flags=re.IGNORECASE,
            )

            if not match:
                continue

            speaker = match.group(1).lower()
            content = match.group(2).strip()

            if not content:
                continue

            if speaker in {"user", "human"}:
                messages.append(HumanMessage(content=content))

            else:
                messages.append(AIMessage(content=content))

        # Do not silently discard a history string that uses
        # a different format.
        if not messages:
            messages.append(
                HumanMessage(
                    content=(
                        "Previously supplied conversation history:\n"
                        + history_text
                    )
                )
            )

        return messages

    return []


# ============================================================
# 3. LLM RESPONSE EXTRACTION
# ============================================================

def extract_response_text(response: Any) -> str:
    """Extract readable text from a LangChain model response."""

    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []

        for part in content:
            if isinstance(part, dict):
                text = part.get("text")

                if text:
                    parts.append(str(text))
            else:
                parts.append(str(part))

        return "\n".join(parts).strip()

    return str(content).strip()


# ============================================================
# 4. SECURE ROLE-BASED DOCUMENT RETRIEVAL
# ============================================================

def retrieve_role_documents(
    supabase_client: Any,
    question: str,
    match_count: int = 8,
) -> list[dict[str, Any]]:
    """
    Retrieve documents through the secure Supabase RPC.

    IMPORTANT:
    - The user's role is NOT sent to this function.
    - Supabase derives the role from auth.uid() using the
      authenticated user's JWT.
    - Only that role's documents and shared Company Policy
      documents can be returned.
    """

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    query_embedding = embeddings.embed_query(question.strip())

    response = supabase_client.rpc(
        "match_documents",
        {
            "query_embedding": query_embedding,
            "match_count": min(max(match_count, 1), 20),
        },
    ).execute()

    rows = response.data or []

    documents = []

    for row in rows:
        metadata = row.get("metadata") or {}

        documents.append(
            {
                "id": row.get("id"),
                "content": row.get("content", ""),
                "metadata": metadata,
                "similarity": row.get("similarity"),
                "source": metadata.get(
                    "source",
                    metadata.get("file_name", "Knowledge document"),
                ),
            }
        )

    return documents


# ============================================================
# 5. DOCUMENT CONTEXT BUILDING
# ============================================================




def build_document_context(
    documents: list[dict[str, Any]],
) -> str:
    """Format retrieved chunks with accurate PDF source and page details."""

    if not documents:
        return "No relevant knowledge document chunks were found."

    context_parts = []

    for document in documents:
        metadata = document.get("metadata") or {}

        source = (
            metadata.get("source")
            or document.get("source")
            or "Knowledge document"
        )

        title = metadata.get("title")
        page_label = metadata.get("page_label")

        source_details = [str(source)]

        if title and title != source:
            source_details.append(f"Title: {title}")

        if page_label is not None:
            source_details.append(f"Page: {page_label}")

        content = document.get("content", "")

        if not isinstance(content, str) or not content.strip():
            continue

        context_parts.append(
            f"[Source: {' | '.join(source_details)}]\n{content}"
        )

    if not context_parts:
        return "No relevant knowledge document chunks were found."

    return "\n\n".join(context_parts)




# ============================================================
# 6. QUESTION CONTEXTUALIZATION
# ============================================================

def contextualize_question(
    question: str,
    chat_history: list[BaseMessage],
) -> str:
    """
    Rewrite a follow-up question into a standalone question.

    Conversation history can clarify references, but it cannot
    provide factual evidence for the final answer.
    """

    if not chat_history:
        return question.strip()

    prompt_value = contextualize_prompt.invoke(
        {
            "chat_history": chat_history,
            "input": question.strip(),
        }
    )

    response = llm.invoke(prompt_value)
    contextualized = extract_response_text(response)

    return contextualized or question.strip()


# ============================================================
# 7. INTENT CLASSIFICATION
# ============================================================

def classify_intent(
    question: str,
    chat_history: list[BaseMessage],
) -> str:
    """
    Classify the contextualized question.

    Invalid classifier output defaults to OUT_OF_SCOPE so that
    an uncertain classification cannot trigger unintended
    retrieval or an unrestricted response.
    """

    prompt_value = intent_prompt.invoke(
        {
            "chat_history": chat_history,
            "input": question.strip(),
        }
    )

    response = llm.invoke(prompt_value)
    result = extract_response_text(response).upper().strip()

    # Handle output such as: DOCUMENT.
    result = result.replace(".", "").strip()

    # Extract the first valid label.
    for token in result.split():
        if token in VALID_INTENTS:
            return token

    return "OUT_OF_SCOPE"


# ============================================================
# 8. ANSWER GENERATION
# ============================================================

def generate_answer(
    question: str,
    role: str,
    documents: list[dict[str, Any]],
    conversation_history: Any = "",
    live_data_context: str | None = None,
) -> str:
    """
    Generate a grounded answer from supplied document context
    and, when available, trusted backend-supplied live data.

    This function does not perform retrieval itself.
    """

    if role not in VALID_ROLES:
        raise PermissionError(
            "User has an invalid or unauthorized role."
        )

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    document_context = build_document_context(documents)

    live_context = (
        live_data_context.strip()
        if live_data_context and live_data_context.strip()
        else "No live operational data was supplied for this question."
    )

    if live_data_context and live_data_context.strip():
        selected_prompt = combined_answer_prompt

        prompt_values = {
            "role": role,
            "context": document_context,
            "live_context": live_context,
            "input": question.strip(),
        }

    else:
        selected_prompt = answer_prompt

        prompt_values = {
            "role": role,
            "context": document_context,
            "input": question.strip(),
        }

    prompt_value = selected_prompt.invoke(prompt_values)

    response = llm.invoke(prompt_value)

    return extract_response_text(response)


# ============================================================
# 9. COMPLETE ROLE-BASED RAG PIPELINE
# ============================================================

def answer_question(
    supabase_client: Any,
    question: str,
    role: str,
    conversation_history: Any = "",
    live_data_context: str | None = None,
    match_count: int = 8,
) -> dict[str, Any]:
    """
    Complete PulseOps RAG execution flow.

    1. Validate the authenticated role.
    2. Normalize conversation history.
    3. Contextualize the latest question.
    4. Classify the intent.
    5. Route to the appropriate prompt.
    6. Retrieve authorized documents only when required.
    7. Use only backend-supplied live operational data.
    8. Return the answer and source information.
    """

    if role not in VALID_ROLES:
        raise PermissionError(
            "User has an invalid or unauthorized role."
        )

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    chat_history = normalize_conversation_history(
        conversation_history
    )

    original_question = question.strip()

    # --------------------------------------------------------
    # STEP 1: CONTEXTUALIZE THE QUESTION
    # --------------------------------------------------------

    contextualized = contextualize_question(
        question=original_question,
        chat_history=chat_history,
    )

    # --------------------------------------------------------
    # STEP 2: CLASSIFY THE INTENT
    # --------------------------------------------------------

    intent = classify_intent(
        question=contextualized,
        chat_history=chat_history,
    )

    documents: list[dict[str, Any]] = []
    sources: list[str] = []

    # --------------------------------------------------------
    # STEP 3: HISTORY ROUTE
    # --------------------------------------------------------

    if intent == "HISTORY":

        if not chat_history:
            answer = NO_HISTORY_ANSWER

        else:
            prompt_value = history_prompt.invoke(
                {
                    "chat_history": chat_history,
                    "input": contextualized,
                }
            )

            response = llm.invoke(prompt_value)
            answer = extract_response_text(response)

    # --------------------------------------------------------
    # STEP 4: GENERAL ROUTE
    # --------------------------------------------------------

    elif intent == "GENERAL":

        prompt_value = general_prompt.invoke(
            {
                "chat_history": chat_history,
                "input": contextualized,
            }
        )

        response = llm.invoke(prompt_value)
        answer = extract_response_text(response)

    # --------------------------------------------------------
    # STEP 5: OUT-OF-SCOPE ROUTE
    # --------------------------------------------------------

    elif intent == "OUT_OF_SCOPE":

        prompt_value = out_of_scope_prompt.invoke(
            {
                "input": contextualized,
            }
        )

        response = llm.invoke(prompt_value)
        answer = extract_response_text(response)

    # --------------------------------------------------------
    # STEP 6: LIVE DATA ROUTE
    # --------------------------------------------------------

    elif intent == "LIVE_DATA":

        if not live_data_context or not live_data_context.strip():
            answer = NO_LIVE_DATA_ANSWER

        else:
            prompt_value = live_data_prompt.invoke(
                {
                    "role": role,
                    "live_context": live_data_context.strip(),
                    "input": contextualized,
                }
            )

            response = llm.invoke(prompt_value)
            answer = extract_response_text(response)

    # --------------------------------------------------------
    # STEP 7: DOCUMENT ROUTE
    # --------------------------------------------------------

    elif intent == "DOCUMENT":

        documents = retrieve_role_documents(
            supabase_client=supabase_client,
            question=contextualized,
            match_count=match_count,
        )

        sources = list(
            dict.fromkeys(
                document["source"]
                for document in documents
                if document.get("source")
            )
        )

        if not documents:
            answer = NO_DOCUMENT_ANSWER

        else:
            prompt_value = answer_prompt.invoke(
                {
                    "role": role,
                    "context": build_document_context(documents),
                    "input": contextualized,
                }
            )

            response = llm.invoke(prompt_value)
            answer = extract_response_text(response)

    # --------------------------------------------------------
    # STEP 8: COMBINED DOCUMENT + LIVE DATA ROUTE
    # --------------------------------------------------------

    elif intent == "BOTH":

        documents = retrieve_role_documents(
            supabase_client=supabase_client,
            question=contextualized,
            match_count=match_count,
        )

        sources = list(
            dict.fromkeys(
                document["source"]
                for document in documents
                if document.get("source")
            )
        )

        has_documents = bool(documents)

        has_live_data = bool(
            live_data_context
            and live_data_context.strip()
        )

        if not has_documents and not has_live_data:
            answer = (
                "I don't have sufficient authorized document "
                "or live operational information to answer "
                "that question."
            )

        else:
            prompt_value = combined_answer_prompt.invoke(
                {
                    "role": role,
                    "context": build_document_context(documents),
                    "live_context": (
                        live_data_context.strip()
                        if has_live_data
                        else (
                            "No live operational data was supplied "
                            "for this question."
                        )
                    ),
                    "input": contextualized,
                }
            )

            response = llm.invoke(prompt_value)
            answer = extract_response_text(response)

    # --------------------------------------------------------
    # STEP 9: RETURN RESULTS
    # --------------------------------------------------------

    return {
        "answer": answer,
        "sources": sources,
        "documents_retrieved": len(documents),
        "role": role,
        "intent": intent,
        "contextualized_question": contextualized,
    }
