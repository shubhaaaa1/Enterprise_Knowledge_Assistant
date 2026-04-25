"""Generator component — wraps Ollama /api/generate for grounded answer generation."""

from __future__ import annotations

import logging
from typing import Iterator, List

import requests

from enterprise_rag.models import ScoredChunk, Turn

logger = logging.getLogger(__name__)

INSUFFICIENT_INFO_RESPONSE = (
    "I don't have sufficient information in the provided context to answer this question."
)

_PROMPT_HEADER = (
    "You are a helpful assistant. Answer the user's question using ONLY the context below.\n"
    "For every factual claim, cite the chunk number in brackets, e.g. [1].\n"
    "If the context does not contain sufficient information, say so explicitly.\n"
)


class Generator:
    """Generates grounded answers via a local Ollama instance."""

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "llama3",
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 2048,
        context_window: int = 4096,
    ) -> None:
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.context_window = context_window

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate(
        self,
        query: str,
        chunks: List[ScoredChunk],
        history: List[Turn],
        stream: bool = True,
    ) -> Iterator[str] | str:
        """Generate an answer from *chunks* for *query*.

        Returns an ``Iterator[str]`` of tokens when *stream* is ``True``,
        or a complete ``str`` when *stream* is ``False``.

        Raises:
            TimeoutError: if Ollama does not respond within 30 seconds.
            ConnectionError: if Ollama is unreachable.
        """
        if not chunks:
            if stream:
                return iter([INSUFFICIENT_INFO_RESPONSE])
            return INSUFFICIENT_INFO_RESPONSE

        # Sort chunks by relevance_score descending (highest first).
        sorted_chunks = sorted(chunks, key=lambda c: c.relevance_score, reverse=True)

        # Enforce context window — remove lowest-ranked chunks until prompt fits.
        sorted_chunks = self._truncate_to_context(sorted_chunks, query, history)

        prompt = self._build_prompt(query, sorted_chunks, history)

        if stream:
            return self._stream(prompt)
        return self._complete(prompt)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        query: str,
        sorted_chunks: List[ScoredChunk],
        history: List[Turn],
    ) -> str:
        """Build the full prompt string from sorted chunks and history."""
        parts: List[str] = [_PROMPT_HEADER, "\nContext:\n"]
        for i, chunk in enumerate(sorted_chunks, start=1):
            parts.append(f"[{i}] {chunk.text}\n")

        if history:
            parts.append("\nConversation history:\n")
            for turn in history:
                parts.append(f"{turn.role}: {turn.original_query}\n")
                if turn.answer:
                    parts.append(f"assistant: {turn.answer}\n")

        parts.append(f"\nQuestion: {query}\n")
        return "".join(parts)

    def _count_tokens(self, text: str) -> int:
        """Approximate token count as word count (simple whitespace split)."""
        return len(text.split())

    def _truncate_to_context(
        self,
        sorted_chunks: List[ScoredChunk],
        query: str,
        history: List[Turn],
    ) -> List[ScoredChunk]:
        """Remove lowest-ranked chunks until the prompt fits within context_window."""
        working = list(sorted_chunks)
        while working:
            prompt = self._build_prompt(query, working, history)
            if self._count_tokens(prompt) <= self.context_window:
                break
            removed = working.pop()  # remove lowest-ranked (last in sorted list)
            logger.info(
                "Context window truncation: removed chunk %s (score=%.4f); "
                "%d chunks remaining.",
                removed.chunk_id,
                removed.relevance_score,
                len(working),
            )
        return working

    def _stream(self, prompt: str) -> Iterator[str]:
        """Yield tokens from Ollama using streaming mode."""
        import json

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "num_predict": self.max_tokens,
            },
        }
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                stream=True,
                timeout=120,
            )
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    token = data.get("response", "")
                    if token:
                        yield token
                    if data.get("done", False):
                        break
        except requests.exceptions.Timeout as exc:
            raise TimeoutError("Ollama generation timed out after 120 seconds.") from exc
        except requests.exceptions.ConnectionError as exc:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.ollama_url}."
            ) from exc

    def _complete(self, prompt: str) -> str:
        """Return the full response from Ollama (non-streaming)."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "num_predict": self.max_tokens,
            },
        }
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                stream=False,
                timeout=120,
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except requests.exceptions.Timeout as exc:
            raise TimeoutError("Ollama generation timed out after 120 seconds.") from exc
        except requests.exceptions.ConnectionError as exc:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.ollama_url}."
            ) from exc
