# raml_tools.py
# ─────────────────────────────────────────────────────────────────────────────
# Pure pipeline functions — called directly and in order, no agent loop.
#
# Why not ReAct agent?
#   RAML generation is a fixed 4-step pipeline. An agent loop adds 3-4 extra
#   LLM round-trips just to "decide" what to do — and worse, loses structured
#   data between steps (lessons string gets dropped by the text chain).
#   Direct function calls: faster, reliable, lessons always injected correctly.
#
# Pipeline order (called from raml_agent.py):
#   1. fetch_context(query)            → RAG context string + sources
#   2. fetch_lessons(query)            → lessons block injected into system prompt
#   3. generate(llm, request, ...)     → dict with files, message, changed_files
#   4. save_lesson(llm, ...)           → lesson saved silently (feedback turns only)
# ─────────────────────────────────────────────────────────────────────────────

import json
import re

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from raml_prompts import RAML_GENERATION_PROMPT, LESSON_EXTRACTION_PROMPT


# ── Shared: robust JSON parser ────────────────────────────────────────────────

def parse_json_safe(text: str) -> dict:
    """
    Parse JSON from LLM output — never raises.
    Handles: plain JSON, markdown fences, preamble text, partial output.
    """
    text = text.strip()
    # 1. Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 2. Strip markdown fences then retry
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.MULTILINE)
    cleaned = re.sub(r"\n?```\s*$", "", cleaned, flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # 3. Bracket-count to extract outermost { ... } (handles preamble text)
    start = text.find("{")
    if start != -1:
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
    # Fallback — surface text as message, never crash
    return {"message": text, "files": [], "changed_files": [], "deleted_files": []}


def _clean_raml(path: str, content: str) -> str:
    """Strip fences and ensure every .raml file starts with #%RAML 1.0."""
    content = re.sub(r"^```[a-z]*\n?", "", content, flags=re.MULTILINE)
    content = re.sub(r"\n?```$",       "", content, flags=re.MULTILINE).strip()
    if path.endswith(".raml") and not content.startswith("#%RAML"):
        content = "#%RAML 1.0\n" + content
    return content


# ── Step 1: RAG retrieval ─────────────────────────────────────────────────────

def fetch_context(rag_retriever, query: str) -> tuple[str, list]:
    """
    Retrieve relevant RAML patterns from the knowledge base.
    Returns (context_string, sources_list).
    """
    if rag_retriever is None:
        return "", []
    try:
        raw     = rag_retriever.retrieve(query=query, top_k=5)
        context = rag_retriever.retrieve_for_llm(query=query, top_k=5)
        sources = [
            {
                "file":   r["source_file"],
                "type":   r["source_type"],
                "detail": r.get("resource_path") or r.get("section", ""),
                "score":  round(r["score"], 3),
            }
            for r in raw
        ]
        return context, sources
    except Exception as e:
        return f"[RAG error: {e}]", []


# ── Step 2: Lesson retrieval ──────────────────────────────────────────────────

def fetch_lessons(lesson_memory, query: str) -> tuple[str, list]:
    """
    Retrieve learned rules from Pinecone for the current query.
    Returns (lessons_block, raw_lessons_list).

    The lessons_block is prepended directly to the system prompt — this
    guarantees the rules are seen before generation. Passing lessons through
    an agent text chain is unreliable; this is the correct approach.
    """
    if lesson_memory is None:
        return "", []
    try:
        lessons = lesson_memory.retrieve(query=query)
        if not lessons:
            return "", []
        rules = "\n".join(
            f"{i}. [{l['category'].upper()}] {l['correction']}"
            for i, l in enumerate(lessons, 1)
        )
        block = (
            "<learned_rules>\n"
            "MANDATORY — follow these rules learned from past corrections. "
            "Violating them is not allowed:\n\n"
            f"{rules}\n"
            "</learned_rules>"
        )
        return block, lessons
    except Exception as e:
        return "", []


# ── Step 3: RAML generation ───────────────────────────────────────────────────

def generate(
    llm:           ChatAnthropic,
    request:       str,
    context:       str,
    lessons_block: str,
    current_files: dict,
) -> dict:
    """
    Single LLM call to generate or update RAML project files.

    lessons_block is prepended to the system prompt so it is read before
    the base rules — this is the only reliable way to enforce learned rules.

    Returns dict: {message, files, changed_files, deleted_files}.
    """
    # Lessons first → base rules second (order matters for attention)
    system = "\n\n".join(p for p in [lessons_block, RAML_GENERATION_PROMPT] if p)

    # Build user message
    parts = []
    if context:
        parts.append(f"<retrieved_context>\n{context}\n</retrieved_context>")
    if current_files:
        files_summary = "\n\n".join(
            f"=== {p} ===\n{c[:1000]}{'...(truncated)' if len(c) > 1000 else ''}"
            for p, c in current_files.items()
        )
        parts.append(f"Current project files:\n{files_summary}")
    parts.append(f"User request: {request}")

    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content="\n\n---\n\n".join(parts)),
    ])
    parsed = parse_json_safe(response.content)

    # Ensure all RAML files are clean
    for f in parsed.get("files", []):
        f["content"] = _clean_raml(f["path"], f["content"])

    return parsed


# ── Step 4: Lesson extraction (feedback turns only) ───────────────────────────

def save_lesson(
    llm:           ChatAnthropic,
    lesson_memory,
    last_message:  str,
    user_feedback: str,
    project_name:  str,
) -> dict | None:
    """
    Background call: detect if user feedback is a correction, save lesson silently.
    Returns the saved lesson dict, or None if not a correction.
    """
    if lesson_memory is None:
        return None
    try:
        response = llm.bind(max_tokens=150).invoke([
            SystemMessage(content=LESSON_EXTRACTION_PROMPT),
            HumanMessage(content=(
                f"Agent's last response:\n{last_message[:500]}\n\n"
                f"User follow-up:\n{user_feedback}"
            )),
        ])
        result = parse_json_safe(response.content)

        if not result.get("is_correction"):
            return None

        mistake    = result.get("mistake", "").strip()
        correction = result.get("correction", "").strip()
        if not mistake or not correction:
            return None

        lesson_id = lesson_memory.save(
            mistake      = mistake,
            correction   = correction,
            category     = result.get("category", "general"),
            project_name = project_name,
        )
        return {
            "id":         lesson_id,
            "mistake":    mistake,
            "correction": correction,
            "category":   result.get("category", "general"),
        }
    except Exception:
        return None