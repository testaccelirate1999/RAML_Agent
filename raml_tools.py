# raml_tools.py
# Tool definitions using the modern @tool decorator (langchain >= 1.0).
# Each tool is a plain Python function — no class, no boilerplate.
# Import build_tools() into raml_agent.py and pass the returned list to create_agent().

import json
import re
from langchain.tools import tool
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from raml_prompts import RAML_GENERATION_PROMPT, LESSON_EXTRACTION_PROMPT


# ── Shared helpers ────────────────────────────────────────────────────────────

def parse_json_safe(text: str) -> dict:
    """
    Parse JSON from an LLM response — never raises.
    Handles markdown fences, preamble text, and partial output.
    """
    text = text.strip()
    # Try 1: direct
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try 2: strip fences
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.MULTILINE)
    cleaned = re.sub(r"\n?```\s*$", "", cleaned, flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Try 3: bracket-counting to find outermost { }
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
    # Fallback: return text as message so UI still renders something
    return {"message": text, "files": [], "changed_files": [], "deleted_files": []}


def clean_raml(path: str, content: str) -> str:
    """Strip fences and ensure .raml files start with #%RAML 1.0."""
    content = re.sub(r"^```[a-z]*\n?", "", content, flags=re.MULTILINE)
    content = re.sub(r"\n?```$", "", content, flags=re.MULTILINE).strip()
    if path.endswith(".raml") and not content.startswith("#%RAML"):
        content = "#%RAML 1.0\n" + content
    return content


# ── Tool factory ──────────────────────────────────────────────────────────────
# We use a factory because tools need injected dependencies (retriever, lessons, llm).
# The @tool decorator is applied inside the factory so closures capture the deps.

def build_tools(rag_retriever, lesson_memory, llm: ChatAnthropic) -> list:
    """
    Build all @tool-decorated functions with injected dependencies.
    Returns a plain list ready to pass to create_agent().
    """

    @tool
    def retrieve_context(query: str) -> str:
        """
        Retrieve relevant RAML patterns from the knowledge base for the given query.
        Call this first on every generation turn.
        """
        if rag_retriever is None:
            return "Knowledge base not connected."
        try:
            return rag_retriever.retrieve_for_llm(query=query, top_k=5)
        except Exception as e:
            return f"Retrieval error: {e}"

    @tool
    def retrieve_lessons(query: str) -> str:
        """
        Retrieve learned rules from past corrections relevant to this query.
        Call this after retrieve_context, before generating files.
        Returns a block of rules the agent must follow, or empty string if none.
        """
        if lesson_memory is None:
            return ""
        try:
            lessons = lesson_memory.retrieve(query=query)
            if not lessons:
                return ""
            rules = "\n".join(
                f"{i}. [{l['category'].upper()}] {l['correction']}"
                for i, l in enumerate(lessons, 1)
            )
            return f"MANDATORY RULES from past corrections:\n{rules}"
        except Exception as e:
            return f"Lesson retrieval error: {e}"

    @tool
    def generate_raml(request: str, context: str = "", lessons: str = "", current_files: str = "") -> str:
        """
        Generate or update RAML project files.
        Args:
            request:       The user's API description or feedback message.
            context:       Output from retrieve_context tool.
            lessons:       Output from retrieve_lessons tool.
            current_files: JSON string of existing files for feedback turns (empty for first turn).
        Returns JSON with: message, files, changed_files, deleted_files.
        """
        # Build system prompt — learned rules on top, base rules below
        system = "\n\n".join(filter(None, [lessons, RAML_GENERATION_PROMPT]))

        # Build user message
        parts = []
        if context:
            parts.append(f"<retrieved_context>\n{context}\n</retrieved_context>")
        if current_files:
            parts.append(f"Current project files:\n{current_files}")
        parts.append(f"User request: {request}")

        messages = [
            SystemMessage(content=system),
            HumanMessage(content="\n\n---\n\n".join(parts)),
        ]
        response = llm.invoke(messages)
        parsed   = parse_json_safe(response.content)

        # Clean RAML content in every returned file
        for f in parsed.get("files", []):
            f["content"] = clean_raml(f["path"], f["content"])

        return json.dumps(parsed)

    @tool
    def extract_lesson(agent_response: str, user_feedback: str, project_name: str = "") -> str:
        """
        Detect if the user is correcting a mistake and silently save the lesson.
        Call this only on feedback turns, after generate_raml.
        Args:
            agent_response: The agent's previous reply (what the user is reacting to).
            user_feedback:  The user's correction or feedback message.
            project_name:   Name of the current project (for display in lesson list).
        Returns JSON: {"saved": true, ...lesson fields} or {"saved": false}.
        """
        if lesson_memory is None:
            return json.dumps({"saved": False})

        prompt = f"Agent's response:\n{agent_response[:500]}\n\nUser follow-up:\n{user_feedback}"
        messages = [
            SystemMessage(content=LESSON_EXTRACTION_PROMPT),
            HumanMessage(content=prompt),
        ]
        # Cheap call — small output, no streaming needed
        response = llm.bind(max_tokens=150).invoke(messages)
        result   = parse_json_safe(response.content)

        if not result.get("is_correction"):
            return json.dumps({"saved": False})

        mistake    = result.get("mistake", "").strip()
        correction = result.get("correction", "").strip()
        if not mistake or not correction:
            return json.dumps({"saved": False, "reason": "incomplete extraction"})

        lesson_id = lesson_memory.save(
            mistake      = mistake,
            correction   = correction,
            category     = result.get("category", "general"),
            project_name = project_name,
        )
        return json.dumps({
            "saved":      True,
            "id":         lesson_id,
            "mistake":    mistake,
            "correction": correction,
            "category":   result.get("category", "general"),
        })

    return [retrieve_context, retrieve_lessons, generate_raml, extract_lesson]