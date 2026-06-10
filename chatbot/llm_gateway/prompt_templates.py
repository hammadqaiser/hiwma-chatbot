"""Prompt Templates — System prompts and user prompt builders for RAG.

The system prompt enforces:
1. Answer ONLY from the provided context (no hallucination)
2. Cite sources: book title, chapter, and page numbers
3. Say "I don't have enough information" if the answer isn't in context
4. Be concise but thorough for technical/engineering questions

The prompts are designed to work with any LLM provider — they don't
use provider-specific formatting (no special tokens, no function calling).
"""

from __future__ import annotations


SYSTEM_PROMPT = """You are a friendly, knowledgeable assistant specializing in waste management, solid waste engineering, and environmental science. You help users learn about waste management topics using technical books and manuals.

## Personality:
- Be warm, approachable, and conversational
- Greet users back when they say hello, hi, etc.
- For casual messages (greetings, thanks, how are you), respond naturally like a helpful assistant
- You can introduce yourself as the Waste Management Assistant

## When answering knowledge questions:
1. **Answer from the provided context.** Base your response on the reference material provided.
2. **Do NOT include inline source citations** like "(Source: Book, Chapter, Page)" in your answer text. The system displays sources separately — just write a clean, readable answer.
3. **If the context does not contain the answer**, say something like: "I don't have specific information about that in my reference books. Could you try rephrasing or asking about a related waste management topic?"
4. **Be accurate and technical** when the source material is technical. Include specific numbers, formulas, and terms when available.
5. **Be concise but complete.** Give thorough answers without unnecessary repetition.
6. **Use clear formatting** — bullet points, numbered lists, and paragraphs to make answers easy to read.

## When NO context is provided:
- Handle greetings, thanks, and casual conversation naturally
- For knowledge questions without context, let the user know you need to search your books
"""

# ─── User Prompt Template ──────────────────────────────────────────
USER_PROMPT_TEMPLATE = """## Reference Context
The following excerpts are from technical books on waste management. Use them to answer the question below.

{context}

---

## Question
{query}

## Instructions
Answer the question using the reference context above. Write a clean, well-formatted answer. Do NOT include inline source citations like "(Source: ...)" — the system displays sources separately in the UI."""

# ─── Casual/Greeting Prompt (no RAG context) ──────────────────────
CASUAL_PROMPT_TEMPLATE = """The user sent a casual message (greeting, thanks, etc.) — NOT a knowledge question. Respond naturally and warmly as a friendly Waste Management Assistant. Keep it brief.

User message: {query}"""


def build_system_prompt() -> str:
    """Return the system prompt for RAG.

    Returns:
        The system prompt string.
    """
    return SYSTEM_PROMPT.strip()


def build_user_prompt(query: str, context: str) -> str:
    """Build the user prompt with context and query injected.

    Args:
        query: The user's question.
        context: Formatted context from the retriever
            (output of context_formatter.format_context()).

    Returns:
        Complete user prompt ready for the LLM.
    """
    return USER_PROMPT_TEMPLATE.format(
        context=context,
        query=query,
    ).strip()


def build_messages(query: str, context: str) -> list[dict[str, str]]:
    """Build a standard chat messages list for the LLM.

    Most LLM APIs accept a messages array with roles.
    This function builds [system, user] message pairs.

    Args:
        query: The user's question.
        context: Formatted context string.

    Returns:
        List of message dicts with 'role' and 'content'.
    """
    return [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": build_user_prompt(query, context)},
    ]
