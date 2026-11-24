"""
PulseOps LangChain chains.

Connects the prompts defined in prompts.py
to the Groq language model.
"""

from langchain_core.output_parsers import StrOutputParser

from src.config import llm

from src.prompts import (
    intent_prompt,
    contextualize_prompt,
    answer_prompt,
    live_data_prompt,
    combined_answer_prompt,
    history_prompt,
    general_prompt,
)


# ============================================================
# 1. OUTPUT PARSER
# ============================================================

output_parser = StrOutputParser()


# ============================================================
# 2. INTENT CLASSIFICATION CHAIN
# ============================================================

intent_chain = (
    intent_prompt
    | llm
    | output_parser
)


# ============================================================
# 3. QUESTION CONTEXTUALIZATION CHAIN
# ============================================================

contextualize_chain = (
    contextualize_prompt
    | llm
    | output_parser
)


# ============================================================
# 4. DOCUMENT ANSWER CHAIN
# ============================================================

document_chain = (
    answer_prompt
    | llm
    | output_parser
)


# ============================================================
# 5. LIVE OPERATIONAL DATA ANSWER CHAIN
# ============================================================

live_data_chain = (
    live_data_prompt
    | llm
    | output_parser
)


# ============================================================
# 6. COMBINED DOCUMENT + LIVE DATA ANSWER CHAIN
# ============================================================

combined_answer_chain = (
    combined_answer_prompt
    | llm
    | output_parser
)


# ============================================================
# 7. CONVERSATION HISTORY CHAIN
# ============================================================

history_chain = (
    history_prompt
    | llm
    | output_parser
)


# ============================================================
# 8. GENERAL CONVERSATION CHAIN
# ============================================================

general_chain = (
    general_prompt
    | llm
    | output_parser
)
