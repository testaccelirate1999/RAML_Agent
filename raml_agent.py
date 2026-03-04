# raml_agent.py
# ─────────────────────────────────────────────────────────────────────────────
# RAML Generation Agent — direct pipeline, LangChain for LLM interface only.
#
# Session persistence:
#   Each session is saved to output/{session_id}/.session.json on every write.
#   On startup, all existing session directories are scanned and restored.
#   This means sessions survive server restarts and --reload cycles.
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
SESSION_FILE = ".session.json"   # saved inside each project directory


# ── Session ───────────────────────────────────────────────────────────────────

class RAMLSession:
    """
    Holds all state for one project — files + conversation history.
    Persists to output/{session_id}/.session.json so restarts don't lose sessions.
    """

    def __init__(self, session_id: str, project_name: str, created_at: str = None):
        self.session_id   = session_id
        self.project_name = project_name
        self.history: list[dict]     = []
        self.files:   dict[str, str] = {}
        self.created_at  = created_at or datetime.now().isoformat()
        self.project_dir = OUTPUT_DIR / session_id

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self):
        """Write session metadata + history to .session.json (files are already on disk)."""
        self.project_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "session_id":   self.session_id,
            "project_name": self.project_name,
            "created_at":   self.created_at,
            "history":      self.history,
            "file_paths":   list(self.files.keys()),   # paths only; content read from disk
        }
        (self.project_dir / SESSION_FILE).write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, project_dir: Path) -> Optional["RAMLSession"]:
        """
        Restore a session from its .session.json + files on disk.
        Returns None if the directory is not a valid saved session.
        """
        meta_path = project_dir / SESSION_FILE
        if not meta_path.exists():
            return None
        try:
            data    = json.loads(meta_path.read_text(encoding="utf-8"))
            session = cls(
                session_id   = data["session_id"],
                project_name = data["project_name"],
                created_at   = data.get("created_at"),
            )
            session.history = data.get("history", [])
            # Re-read file contents from disk
            for rel_path in data.get("file_paths", []):
                full = project_dir / rel_path
                if full.exists():
                    session.files[rel_path] = full.read_text(encoding="utf-8")
            return session
        except Exception as e:
            print(f"[RAMLSession] Could not load {project_dir}: {e}")
            return None

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
    Sessions are persisted to disk and restored on startup.
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

        self.llm = ChatAnthropic(
            model             = claude_model,
            temperature       = 0,
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

        # Restore sessions from disk on startup
        self._restore_sessions()

    # ── Session persistence ───────────────────────────────────────────────────

    def _restore_sessions(self):
        """
        Scan output/ for saved sessions and reload them into memory.
        Called once on startup — this is why sessions survive --reload.
        """
        count = 0
        for child in sorted(self.output_dir.iterdir()):
            if not child.is_dir():
                continue
            session = RAMLSession.load(child)
            if session:
                self.sessions[session.session_id] = session
                count += 1
        if count and self.verbose:
            print(f"[RAMLAgent] Restored {count} session(s) from disk")

    # ── Session management ────────────────────────────────────────────────────

    def create_session(self, project_name: str) -> RAMLSession:
        ts         = datetime.now().strftime("%Y%m%d-%H%M%S")
        slug       = re.sub(r"[^a-z0-9]+", "-", project_name.lower()).strip("-")
        session_id = f"{slug}-{ts}"
        session    = RAMLSession(session_id=session_id, project_name=project_name)
        session.project_dir.mkdir(parents=True, exist_ok=True)
        session.save()   # create .session.json immediately
        self.sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[RAMLSession]:
        return self.sessions.get(session_id)

    def list_sessions(self) -> list:
        return [s.to_dict() for s in self.sessions.values()]

    # ── Core: one chat turn ───────────────────────────────────────────────────

    def chat(self, session_id: str, message: str) -> dict:
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session '{session_id}' not found")

        is_first = len(session.history) == 0

        # Step 1: RAG context
        context, sources = fetch_context(self._rag, message)

        # Step 2: Lessons → system prompt
        lessons_block, lessons_used = fetch_lessons(self._lesson_memory, message)
        if self.verbose and lessons_used:
            print(f"[RAMLAgent] Injecting {len(lessons_used)} lessons into system prompt")
            for l in lessons_used:
                print(f"  [{l['category']}] {l['correction'][:70]}")

        # Step 3: Generate RAML
        result = generate(
            llm           = self.llm,
            request       = message,
            context       = context,
            lessons_block = lessons_block,
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

        # Update history
        agent_message = result.get("message", "Done.")
        session.history.append({"role": "user",      "content": message})
        session.history.append({"role": "assistant", "content": agent_message})

        # Persist session after every turn
        session.save()

        # Step 4: Save lesson silently (feedback turns only)
        lesson_saved = None
        if not is_first:
            last_reply = session.history[-4]["content"] if len(session.history) >= 4 else ""
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