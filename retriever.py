# retriever.py
# ─────────────────────────────────────────────────────────────────────────────
# Reusable RAG Retriever — drop this into any agent project.
#
# Usage:
#   from retriever import RAGRetriever
#
#   retriever = RAGRetriever(index_name="raml-knowledge-base")
#   context   = retriever.retrieve_for_llm("How do I define OAuth2 in RAML?")
#   # Inject `context` into your LLM system prompt
#
# The retriever is stateless — safe to reuse across calls / threads.
# ─────────────────────────────────────────────────────────────────────────────

import os
from typing import List, Dict, Any, Optional

import voyageai
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()


# ── Config Defaults ───────────────────────────────────────────────────────────

DEFAULT_INDEX_NAME   = os.getenv("PINECONE_INDEX_NAME", "raml-knowledge-base")
DEFAULT_VOYAGE_MODEL = os.getenv("VOYAGE_MODEL", "voyage-code-2")
DEFAULT_TOP_K        = int(os.getenv("TOP_K_FINAL", 5))
DEFAULT_MIN_SCORE    = float(os.getenv("MIN_SCORE", 0.65))

SOURCE_RAML = "raml_file"
SOURCE_TEXT = "text_doc"


# ── Core Retriever ────────────────────────────────────────────────────────────

class RAGRetriever:
    """
    Plug-and-play RAG retriever backed by Pinecone + Voyage AI.

    Designed to be dropped into any agent with a single import.
    Handles embedding, querying, score filtering, and context formatting.

    Args:
        index_name:   Pinecone index to query (default: "raml-knowledge-base")
        voyage_model: Voyage AI model for embeddings (default: "voyage-code-2")
        min_score:    Minimum cosine similarity score to include a result (0–1)
        verbose:      Print debug logs (default: False)
    """

    def __init__(
        self,
        index_name:   str   = DEFAULT_INDEX_NAME,
        voyage_model: str   = DEFAULT_VOYAGE_MODEL,
        min_score:    float = DEFAULT_MIN_SCORE,
        verbose:      bool  = False,
    ):
        self.index_name   = index_name
        self.voyage_model = voyage_model
        self.min_score    = min_score
        self.verbose      = verbose

        # Init Voyage AI
        voyage_api_key = os.getenv("VOYAGE_API_KEY")
        if not voyage_api_key:
            raise ValueError("VOYAGE_API_KEY not set in environment / .env")
        self._voyage = voyageai.Client(api_key=voyage_api_key)

        # Init Pinecone
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            raise ValueError("PINECONE_API_KEY not set in environment / .env")
        pc = Pinecone(api_key=pinecone_api_key)

        existing = [idx.name for idx in pc.list_indexes()]
        if index_name not in existing:
            raise ValueError(
                f"Pinecone index '{index_name}' not found. "
                f"Run the ingestion pipeline first."
            )
        self._index = pc.Index(index_name)

        if self.verbose:
            print(f"[RAGRetriever] Ready — index='{index_name}', model='{voyage_model}'")

    # ── Embedding ─────────────────────────────────────────────────────────────

    def _embed_query(self, query: str) -> List[float]:
        result = self._voyage.embed(
            texts=[query],
            model=self.voyage_model,
            input_type="query",
            truncation=True,
        )
        return result.embeddings[0]

    # ── Raw Retrieval ─────────────────────────────────────────────────────────

    def retrieve(
        self,
        query:       str,
        top_k:       int            = DEFAULT_TOP_K,
        source_type: Optional[str]  = None,
        min_score:   Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant chunks for a query.

        Args:
            query:       Natural language question or task description
            top_k:       Max results to return
            source_type: Optional filter — SOURCE_RAML | SOURCE_TEXT | None (all)
            min_score:   Override instance min_score for this call

        Returns:
            List of result dicts: {text, source_type, source_file, score, ...}
        """
        threshold = min_score if min_score is not None else self.min_score

        if self.verbose:
            print(f"[RAGRetriever] Query: '{query[:80]}...'")

        query_vec = self._embed_query(query)

        filter_dict = {"source_type": {"$eq": source_type}} if source_type else None

        # Fetch more candidates than needed, then filter by score
        fetch_k  = max(top_k * 2, 8)
        response = self._index.query(
            vector=query_vec,
            top_k=fetch_k,
            include_metadata=True,
            filter=filter_dict,
        )

        results = []
        for match in response.matches:
            if match.score < threshold:
                continue
            results.append({
                "score":         match.score,
                "text":          match.metadata.get("text", ""),
                "source_type":   match.metadata.get("source_type", ""),
                "source_file":   match.metadata.get("source_file", ""),
                "chunk_type":    match.metadata.get("chunk_type", ""),
                "section":       match.metadata.get("section", ""),
                "resource_path": match.metadata.get("resource_path", ""),
            })
            if len(results) >= top_k:
                break

        if self.verbose:
            print(f"[RAGRetriever] {len(results)} results (threshold={threshold})")
            for i, r in enumerate(results):
                detail = r.get("resource_path") or r.get("section", "")
                print(f"  [{i+1}] {r['score']:.3f} | {r['source_file']} | {detail}")

        return results

    # ── Formatted Context ─────────────────────────────────────────────────────

    def retrieve_for_llm(
        self,
        query:       str,
        top_k:       int           = DEFAULT_TOP_K,
        source_type: Optional[str] = None,
    ) -> str:
        """
        Retrieve chunks and format them as a ready-to-inject LLM context block.

        Inject the returned string directly into your system prompt:

            system_prompt = f\"""
            You are a RAML expert assistant.

            {retriever.retrieve_for_llm(user_query)}

            Answer based on the context above.
            \"""

        Returns:
            A <retrieved_context>...</retrieved_context> XML block.
        """
        results = self.retrieve(query=query, top_k=top_k, source_type=source_type)

        if not results:
            return "<retrieved_context>\nNo relevant context found.\n</retrieved_context>"

        parts = []
        for i, r in enumerate(results, 1):
            detail = r.get("resource_path") or r.get("section") or r.get("chunk_type", "")
            label  = (
                f"[Source {i} — {r['source_type']}: {r['source_file']}"
                + (f" | {detail}" if detail else "")
                + f" | relevance: {r['score']:.2f}]"
            )
            parts.append(f"{label}\n{r['text']}")

        body = "\n\n---\n\n".join(parts)
        return f"<retrieved_context>\n{body}\n</retrieved_context>"

    def retrieve_mixed(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        raml_k: int = 3,
        text_k: int = 3,
    ) -> str:
        """
        Retrieve from RAML files AND text docs separately, then merge by score.
        Useful when you want both concrete RAML examples AND best-practice docs.

        Returns:
            A <retrieved_context>...</retrieved_context> XML block.
        """
        raml_results = self.retrieve(query=query, top_k=raml_k, source_type=SOURCE_RAML)
        text_results = self.retrieve(query=query, top_k=text_k, source_type=SOURCE_TEXT)

        combined = sorted(raml_results + text_results, key=lambda x: x["score"], reverse=True)
        top      = combined[:top_k]

        if not top:
            return "<retrieved_context>\nNo relevant context found.\n</retrieved_context>"

        parts = []
        for i, r in enumerate(top, 1):
            detail = r.get("resource_path") or r.get("section", "")
            label  = (
                f"[Source {i} — {r['source_type']}: {r['source_file']}"
                + (f" | {detail}" if detail else "")
                + f" | relevance: {r['score']:.2f}]"
            )
            parts.append(f"{label}\n{r['text']}")

        body = "\n\n---\n\n".join(parts)
        return f"<retrieved_context>\n{body}\n</retrieved_context>"

    # ── Convenience Properties ────────────────────────────────────────────────

    @property
    def stats(self) -> Dict[str, Any]:
        """Return Pinecone index statistics."""
        return self._index.describe_index_stats()

    @property
    def vector_count(self) -> int:
        """Total number of vectors in the index."""
        return self.stats.get("total_vector_count", 0)