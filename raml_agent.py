# raml_agent.py
# ─────────────────────────────────────────────────────────────────────────────
# RAML Generation Agent — direct pipeline, LangChain for LLM interface only.
#
# Architecture:
#   - No ReAct agent loop — pipeline calls each step directly in order
#   - LangChain ChatAnthropic for LLM calls (same interface as ChatOpenAI)
#   - RAMLSession holds per-project files + conversation history
#   - LessonMemory lessons are injected into system prompt (not passed via agent)
#
# One chat() turn = exactly 2 LLM calls:
#   1. generate()     — main RAML generation (big call)
#   2. save_lesson()  — lesson extraction (tiny background call, feedback turns only)
#
# File layout:
#   raml_prompts.py  — all prompts as constants
#   raml_tools.py    — fetch_context, fetch_lessons, generate, save_lesson
#   raml_agent.py    — THIS: sessions, pipeline orchestration, file I/O
#   lesson_memory.py — Pinecone lesson store
#   raml_server.py   — FastAPI wrapper
# ─────────────────────────────────────────────────────────────────────────────

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
from langchain_anthropic import ChatAnthropic

load_dotenv()

from retriever     import RAGRetriever
from lesson_memory import LessonMemory
from raml_tools    import fetch_context, fetch_lessons, generate, save_lesson, parse_json_safe

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR   = Path(os.getenv("RAML_OUTPUT_DIR", "output"))
INDEX_NAME   = os.getenv("PINECONE_INDEX_NAME", "raml-knowledge-base")
CLAUDE_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")


# ── Session ───────────────────────────────────────────────────────────────────

class RAMLSession:
    """Holds all state for one project — files + conversation history."""

    def __init__(self, session_id: str, project_name: str):
        self.session_id   = session_id
        self.project_name = project_name
        self.history: list[dict]     = []   # [{role, content}, ...]
        self.files:   dict[str, str] = {}   # path → file content (live state)
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
    Orchestrates the 4-step RAML generation pipeline.
    Uses LangChain ChatAnthropic as the LLM — drop-in for ChatOpenAI.
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

        # LangChain LLM — same interface as ChatOpenAI
        self.llm = ChatAnthropic(
            model             = claude_model,
            temperature       = 0.3,
            max_tokens        = 8096,
            anthropic_api_key = api_key,
        )

        # RAG retriever
        self._rag = None
        self._rag_ready = False
        try:
            self._rag = RAGRetriever(index_name=index_name, verbose=verbose)
            self._rag_ready = True
        except Exception as e:
            if verbose: print(f"[RAMLAgent] RAG unavailable: {e}")

        # Lesson memory
        self._lesson_memory = None
        self._lessons_ready = False
        try:
            self._lesson_memory = LessonMemory(index_name=index_name, verbose=verbose)
            self._lessons_ready = True
        except Exception as e:
            if verbose: print(f"[RAMLAgent] Lessons unavailable: {e}")

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
        Run the 4-step pipeline for one conversation turn.

        Steps:
          1. fetch_context  — RAG lookup (no LLM call)
          2. fetch_lessons  — Pinecone lookup (no LLM call)
          3. generate       — 1 LLM call: lessons in system prompt, context in user msg
          4. save_lesson    — 1 cheap LLM call (feedback turns only)

        Total: 1–2 LLM calls per turn. Same speed as the original direct SDK version.
        """
        session  = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session '{session_id}' not found")

        is_first = len(session.history) == 0

        # ── Step 1: RAG context ───────────────────────────────────────────────
        context, sources = fetch_context(self._rag, message)

        # ── Step 2: Lessons (injected directly into system prompt) ────────────
        lessons_block, lessons_used = fetch_lessons(self._lesson_memory, message)

        if self.verbose and lessons_used:
            print(f"[RAMLAgent] Injecting {len(lessons_used)} lessons into system prompt")
            for l in lessons_used:
                print(f"  [{l['category']}] {l['correction'][:70]}")

        # ── Step 3: Generate RAML ─────────────────────────────────────────────
        result = generate(
            llm           = self.llm,
            request       = message,
            context       = context,
            lessons_block = lessons_block,   # prepended to system prompt
            current_files = session.files if not is_first else {},
        )

        changed_files = result.get("changed_files", [])
        deleted_files = result.get("deleted_files", [])
        new_files     = result.get("files", [])

        # Write / update files on disk
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
        agent_message = result.get("message", "Done.")
        session.history.append({"role": "user",      "content": message})
        session.history.append({"role": "assistant", "content": agent_message})

        # ── Step 4: Save lesson silently (feedback turns only) ────────────────
        lesson_saved = None
        if not is_first:
            last_reply = session.history[-2]["content"] if len(session.history) >= 2 else ""
            lesson_saved = save_lesson(
                llm           = self.llm,
                lesson_memory = self._lesson_memory,
                last_message  = last_reply,
                user_feedback = message,
                project_name  = session.project_name,
            )
            if self.verbose and lesson_saved:
                print(f"[RAMLAgent] Lesson saved: [{lesson_saved['category']}] {lesson_saved['correction'][:60]}")

        return {
            "message":       agent_message,
            "files":         dict(session.files),
            "changed_files": changed_files,
            "deleted_files": deleted_files,
            "sources":       sources,
            "lessons_used":  [
                {"id": l["id"], "correction": l["correction"], "category": l["category"]}
                for l in lessons_used
            ],
            "lesson_saved":  lesson_saved,
            "tokens_used":   {"input": 0, "output": 0},
            "is_first_turn": is_first,
            "session":       session.to_dict(),
        }

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