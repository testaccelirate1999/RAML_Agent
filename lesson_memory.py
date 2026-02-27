# lesson_memory.py
# ─────────────────────────────────────────────────────────────────────────────
# Global Agent Lesson Memory
#
# Design decisions:
#   - Fully automatic: no user confirmation needed, lessons save silently
#   - Global scope: all projects share the same lesson pool
#   - Shared: every team member benefits from every correction
#   - One-way trust: if a lesson was extracted, it's immediately active
#
# Storage: same Pinecone index as RAML knowledge, but "lessons" namespace
#   so they never pollute document retrieval results.
#
# Per-lesson metadata:
#   id:           "lesson-{uuid12}"
#   mistake:      what the agent did wrong (one sentence)
#   correction:   the rule it must follow instead (one sentence, starts with verb)
#   category:     structure | auth | types | endpoints | naming | examples | general
#   project_name: which project surfaced this (for display only)
#   created_at:   ISO timestamp
# ─────────────────────────────────────────────────────────────────────────────

import os
import uuid
from datetime import datetime
from dotenv import load_dotenv

import voyageai
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

LESSONS_NAMESPACE = "lessons"
VOYAGE_MODEL      = os.getenv("VOYAGE_MODEL", "voyage-code-2")
MIN_SCORE         = 0.72   # high bar — only very relevant lessons injected
TOP_K_LESSONS     = 6


class LessonMemory:
    """
    Shared, persistent lesson store backed by Pinecone.
    All lessons are immediately active — no confirmation step.
    """

    def __init__(self, index_name: str = None, verbose: bool = False):
        self.verbose    = verbose
        self.index_name = index_name or os.getenv("PINECONE_INDEX_NAME", "raml-knowledge-base")

        voyage_key = os.getenv("VOYAGE_API_KEY")
        if not voyage_key:
            raise ValueError("VOYAGE_API_KEY not set")
        self._voyage = voyageai.Client(api_key=voyage_key)

        pc_key = os.getenv("PINECONE_API_KEY")
        if not pc_key:
            raise ValueError("PINECONE_API_KEY not set")
        pc = Pinecone(api_key=pc_key)

        existing = [i.name for i in pc.list_indexes()]
        if self.index_name not in existing:
            pc.create_index(
                name      = self.index_name,
                dimension = 1536,
                metric    = "cosine",
                spec      = ServerlessSpec(
                    cloud  = os.getenv("PINECONE_CLOUD", "aws"),
                    region = os.getenv("PINECONE_REGION", "us-east-1"),
                )
            )
        self._index = pc.Index(self.index_name)

        if verbose:
            print(f"[LessonMemory] ready — index='{self.index_name}' ns='{LESSONS_NAMESPACE}'")

    # ── Embed ─────────────────────────────────────────────────────────────────

    def _embed(self, text: str, input_type: str = "document") -> list[float]:
        r = self._voyage.embed(texts=[text], model=VOYAGE_MODEL,
                               input_type=input_type, truncation=True)
        return r.embeddings[0]

    # ── Write ─────────────────────────────────────────────────────────────────

    def save(self, mistake: str, correction: str,
             category: str = "general", project_name: str = "") -> str:
        """
        Silently save a lesson. Immediately active for all future sessions.
        Returns the lesson_id.
        """
        lesson_id   = f"lesson-{uuid.uuid4().hex[:12]}"
        lesson_text = f"[RULE — {category.upper()}] {correction}"

        vec = self._embed(lesson_text, input_type="document")
        self._index.upsert(
            vectors=[{
                "id":     lesson_id,
                "values": vec,
                "metadata": {
                    "text":         lesson_text,
                    "mistake":      mistake,
                    "correction":   correction,
                    "category":     category,
                    "project_name": project_name,
                    "created_at":   datetime.now().isoformat(),
                }
            }],
            namespace=LESSONS_NAMESPACE,
        )
        if self.verbose:
            print(f"[LessonMemory] saved: [{category}] {correction[:70]}")
        return lesson_id

    # ── Read ──────────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = TOP_K_LESSONS) -> list[dict]:
        """
        Retrieve the most relevant lessons for the current query.
        Returns list sorted by relevance, filtered by MIN_SCORE.
        """
        qvec     = self._embed(query, input_type="query")
        response = self._index.query(
            vector=qvec, top_k=top_k,
            include_metadata=True, namespace=LESSONS_NAMESPACE,
        )
        results = []
        for m in response.matches:
            if m.score < MIN_SCORE:
                continue
            md = m.metadata
            results.append({
                "id":           m.id,
                "score":        round(m.score, 3),
                "mistake":      md.get("mistake", ""),
                "correction":   md.get("correction", ""),
                "category":     md.get("category", "general"),
                "project_name": md.get("project_name", ""),
                "created_at":   md.get("created_at", ""),
            })
        if self.verbose:
            print(f"[LessonMemory] retrieved {len(results)} lessons")
        return results

    def format_for_prompt(self, lessons: list[dict]) -> str:
        """
        Format lessons as a <learned_rules> block to prepend to the system prompt.
        Each rule is a hard constraint the agent MUST follow.
        """
        if not lessons:
            return ""
        lines = [
            "<learned_rules>",
            "MANDATORY RULES from past corrections — violating these is not allowed:",
            "",
        ]
        for i, l in enumerate(lessons, 1):
            lines.append(f"{i}. [{l['category'].upper()}] {l['correction']}")
        lines.append("</learned_rules>")
        return "\n".join(lines)

    # ── List / Delete (for UI management) ────────────────────────────────────

    def list_all(self) -> list[dict]:
        """Return all lessons, newest first."""
        dummy = self._embed("RAML API design rules corrections mistakes", "query")
        resp  = self._index.query(
            vector=dummy, top_k=100,
            include_metadata=True, namespace=LESSONS_NAMESPACE,
        )
        lessons = []
        for m in resp.matches:
            md = m.metadata
            lessons.append({
                "id":           m.id,
                "mistake":      md.get("mistake", ""),
                "correction":   md.get("correction", ""),
                "category":     md.get("category", "general"),
                "project_name": md.get("project_name", ""),
                "created_at":   md.get("created_at", ""),
            })
        lessons.sort(key=lambda x: x["created_at"], reverse=True)
        return lessons

    def delete(self, lesson_id: str) -> bool:
        try:
            self._index.delete(ids=[lesson_id], namespace=LESSONS_NAMESPACE)
            return True
        except Exception:
            return False

    @property
    def count(self) -> int:
        try:
            stats = self._index.describe_index_stats()
            return (stats.namespaces or {}).get(LESSONS_NAMESPACE, {}).get("vector_count", 0)
        except Exception:
            return 0