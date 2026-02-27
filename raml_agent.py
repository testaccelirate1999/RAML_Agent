# raml_agent.py
# RAML generation agent using the latest LangChain API (langchain >= 1.0).
#
# create_agent() is the current recommended API — no AgentExecutor,
# no hub.pull(), no initialize_agent(). Just model + tools list.
#
# File structure:
#   raml_prompts.py  — all prompts as constants
#   raml_tools.py    — @tool functions + build_tools() factory
#   raml_agent.py    — THIS: agent init, sessions, chat()
#   lesson_memory.py — Pinecone lesson store (unchanged)
#   raml_server.py   — FastAPI wrapper (unchanged)

import os
import re
import json
import zipfile
import shutil
from io import BytesIO
from pathlib import Path
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic

load_dotenv()

from retriever     import RAGRetriever
from lesson_memory import LessonMemory
from raml_tools    import build_tools, parse_json_safe

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR   = Path(os.getenv("RAML_OUTPUT_DIR", "output"))
INDEX_NAME   = os.getenv("PINECONE_INDEX_NAME", "raml-knowledge-base")
CLAUDE_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")


# ── Session ───────────────────────────────────────────────────────────────────

class RAMLSession:
    """Holds all state for one project — files on disk + conversation history."""

    def __init__(self, session_id: str, project_name: str):
        self.session_id   = session_id
        self.project_name = project_name
        self.history: list[dict]     = []   # [{role, content}, ...]
        self.files:   dict[str, str] = {}   # path → file content
        self.created_at  = datetime.now().isoformat()
        self.project_dir = OUTPUT_DIR / session_id

    def to_dict(self) -> dict:
        return {
            "session_id":   self.session_id,
            "project_name": self.project_name,
            "created_at":   self.created_at,
            "file_count":   len(self.files),
            "files":        list(self.files.keys()),
            "turn_count":   len([h for h in self.history if h["role"] == "user"]),
        }


# ── Agent ─────────────────────────────────────────────────────────────────────

class RAMLAgent:
    """
    RAML generation agent — latest LangChain API.

    create_agent(model, tools=[...]) is all that's needed.
    Tools are plain @tool-decorated functions from raml_tools.py.
    """

    def __init__(
        self,
        index_name:   str  = INDEX_NAME,
        output_dir         = OUTPUT_DIR,
        claude_model: str  = CLAUDE_MODEL,
        verbose:      bool = False,
    ):
        self.output_dir  = Path(output_dir)
        self.verbose     = verbose
        self.sessions: dict[str, RAMLSession] = {}
        self.output_dir.mkdir(parents=True, exist_ok=True)

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

        # LLM — ChatAnthropic is a drop-in for ChatOpenAI
        llm = ChatAnthropic(
            model             = claude_model,
            temperature       = 0,
            max_tokens        = 8096,
            anthropic_api_key = api_key,
        )

        # RAG retriever
        rag = None
        self._rag_ready = False
        try:
            rag = RAGRetriever(index_name=index_name, verbose=verbose)
            self._rag_ready = True
        except Exception as e:
            if verbose: print(f"[RAMLAgent] RAG unavailable: {e}")

        # Lesson memory
        lessons = None
        self._lessons_ready = False
        try:
            lessons = LessonMemory(index_name=index_name, verbose=verbose)
            self._lessons_ready = True
        except Exception as e:
            if verbose: print(f"[RAMLAgent] Lessons unavailable: {e}")

        # Store for lesson surfacing
        self._lesson_memory = lessons

        # Build tools and create agent — that's it, no AgentExecutor needed
        tools = build_tools(rag_retriever=rag, lesson_memory=lessons, llm=llm)
        self.agent = create_agent(llm, tools=tools)

    # ── Session management ────────────────────────────────────────────────────

    def create_session(self, project_name: str) -> RAMLSession:
        ts         = datetime.now().strftime("%Y%m%d-%H%M%S")
        slug       = re.sub(r"[^a-z0-9]+", "-", project_name.lower()).strip("-")
        session_id = f"{slug}-{ts}"
        session    = RAMLSession(session_id=session_id, project_name=project_name)
        session.project_dir.mkdir(parents=True, exist_ok=True)
        self.sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[RAMLSession]:
        return self.sessions.get(session_id)

    def list_sessions(self) -> list:
        return [s.to_dict() for s in self.sessions.values()]

    # ── Core: one chat turn ───────────────────────────────────────────────────

    def chat(self, session_id: str, message: str) -> dict:
        """
        Run one turn through the agent.
        Agent autonomously calls tools in order: retrieve_context →
        retrieve_lessons → generate_raml → extract_lesson (feedback turns).
        Returns a dict compatible with raml_server.py.
        """
        session  = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session '{session_id}' not found")

        is_first = len(session.history) == 0
        query    = self._build_query(message, session, is_first)

        # Run agent — create_agent returns a standard runnable
        try:
            result     = self.agent.invoke({"messages": [("human", query)]})
            # Latest API: result is a dict with "messages" list;
            # the last AI message is the final answer
            raw_output = result["messages"][-1].content
        except Exception as e:
            if self.verbose: print(f"[RAMLAgent] agent error: {e}")
            raw_output = json.dumps({"message": f"Error: {e}", "files": [], "changed_files": [], "deleted_files": []})

        # Parse JSON from the generate_raml tool output surfaced as final answer
        parsed        = parse_json_safe(raw_output)
        changed_files = parsed.get("changed_files", [])
        deleted_files = parsed.get("deleted_files", [])
        new_files     = parsed.get("files", [])

        # Write new/updated files to disk
        for f in new_files:
            path, content = f["path"], f["content"]
            session.files[path] = content
            full = session.project_dir / path
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_text(content, encoding="utf-8")

        # Delete removed files
        for path in deleted_files:
            session.files.pop(path, None)
            full = session.project_dir / path
            if full.exists():
                full.unlink()
            try:
                full.parent.rmdir()
            except OSError:
                pass

        if not changed_files and new_files:
            changed_files = [f["path"] for f in new_files]

        # Update conversation history
        session.history.append({"role": "user",      "content": message})
        session.history.append({"role": "assistant", "content": parsed.get("message", "")})

        # Surface lesson_saved if extract_lesson ran this turn
        lesson_saved = self._find_lesson_in_result(result)

        return {
            "message":       parsed.get("message", "Done."),
            "files":         dict(session.files),
            "changed_files": changed_files,
            "deleted_files": deleted_files,
            "sources":       [],
            "lessons_used":  [],
            "lesson_saved":  lesson_saved,
            "tokens_used":   {"input": 0, "output": 0},
            "is_first_turn": is_first,
            "session":       session.to_dict(),
        }

    # ── Query builders ────────────────────────────────────────────────────────

    def _build_query(self, message: str, session: RAMLSession, is_first: bool) -> str:
        """
        Build an explicit step-by-step task for the agent.
        Mirrors the AML.py style: numbered steps tell the agent exactly what to do.
        """
        if is_first:
            return f"""
Complete these steps in order:
1. Call retrieve_context("{message}")
2. Call retrieve_lessons("{message}")
3. Call generate_raml with:
   - request = "{message}"
   - context = <output of step 1>
   - lessons = <output of step 2>
   - current_files = ""
Return the JSON output from generate_raml as your final answer.
"""
        # Feedback turn — pass current files and trigger lesson extraction
        files_json = json.dumps(
            {p: c[:600] + ("..." if len(c) > 600 else "") for p, c in session.files.items()},
            indent=2
        )
        last_reply = next(
            (h["content"] for h in reversed(session.history) if h["role"] == "assistant"), ""
        )
        return f"""
Complete these steps in order:
1. Call retrieve_context("{message}")
2. Call retrieve_lessons("{message}")
3. Call generate_raml with:
   - request = "{message}"
   - context = <output of step 1>
   - lessons = <output of step 2>
   - current_files = {files_json}
4. Call extract_lesson with:
   - agent_response = {json.dumps(last_reply[:400])}
   - user_feedback = "{message}"
   - project_name = "{session.project_name}"
Return the JSON output from generate_raml as your final answer.
"""

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _find_lesson_in_result(self, result: dict) -> Optional[dict]:
        """
        Scan agent messages for an ExtractLesson tool call result.
        create_agent returns all messages including ToolMessages in result["messages"].
        """
        try:
            for msg in result.get("messages", []):
                # ToolMessage has name = tool name, content = tool output
                if getattr(msg, "name", "") == "extract_lesson":
                    data = parse_json_safe(msg.content)
                    if data.get("saved"):
                        return data
        except Exception:
            pass
        return None

    # ── File ops ──────────────────────────────────────────────────────────────

    def get_zip(self, session_id: str) -> bytes:
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session '{session_id}' not found")
        buf = BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for path, content in session.files.items():
                zf.writestr(f"{session.project_name}/{path}", content)
        return buf.getvalue()

    def get_file(self, session_id: str, path: str) -> str:
        session = self.get_session(session_id)
        if not session or path not in session.files:
            raise FileNotFoundError(f"{path} not found")
        return session.files[path]

    def delete_session(self, session_id: str):
        session = self.sessions.pop(session_id, None)
        if session and session.project_dir.exists():
            shutil.rmtree(session.project_dir)