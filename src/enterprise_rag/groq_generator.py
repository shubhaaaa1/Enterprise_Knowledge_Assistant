"""Groq-based Generator component for ultra-fast LLM inference."""

from __future__ import annotations

import logging
import os
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


class GroqGenerator:
    """Generates grounded answers via Groq's ultra-fast API."""

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.1-8b-instant",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        context_window: int = 4096,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.context_window = context_window
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"

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
            TimeoutError: if Groq does not respond within 30 seconds.
            ConnectionError: if Groq is unreachable.
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
        """Yield tokens from Groq using streaming mode."""
        import json

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                self.api_url,
                json=payload,
                headers=headers,
                stream=True,
                timeout=30,
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        data_str = line_str[6:]  # Remove 'data: ' prefix
                        if data_str.strip() == '[DONE]':
                            break
                        try:
                            data = json.loads(data_str)
                            delta = data.get('choices', [{}])[0].get('delta', {})
                            token = delta.get('content', '')
                            if token:
                                yield token
                        except json.JSONDecodeError:
                            continue
                            
        except requests.exceptions.Timeout as exc:
            raise TimeoutError("Groq generation timed out after 30 seconds.") from exc
        except requests.exceptions.ConnectionError as exc:
            raise ConnectionError("Cannot connect to Groq API.") from exc
        except requests.exceptions.HTTPError as exc:
            if exc.response.status_code == 401:
                raise ConnectionError("Invalid Groq API key.") from exc
            elif exc.response.status_code == 429:
                raise ConnectionError("Groq rate limit exceeded.") from exc
            raise ConnectionError(f"Groq API error: {exc}") from exc

    def _complete(self, prompt: str) -> str:
        """Return the full response from Groq (non-streaming)."""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                self.api_url,
                json=payload,
                headers=headers,
                timeout=30,
            )
            logger.info(f"Groq API response status: {response.status_code}")
            response.raise_for_status()
            data = response.json()
            logger.info(f"Groq API response received successfully")
            return data['choices'][0]['message']['content']
            
        except requests.exceptions.Timeout as exc:
            logger.error(f"Groq timeout: {exc}")
            raise TimeoutError("Groq generation timed out after 30 seconds.") from exc
        except requests.exceptions.ConnectionError as exc:
            logger.error(f"Groq connection error: {exc}")
            raise ConnectionError("Cannot connect to Groq API.") from exc
        except requests.exceptions.HTTPError as exc:
            logger.error(f"Groq HTTP error: {exc}, status: {exc.response.status_code}, body: {exc.response.text[:200]}")
            if exc.response.status_code == 401:
                raise ConnectionError("Invalid Groq API key.") from exc
            elif exc.response.status_code == 429:
                raise ConnectionError("Groq rate limit exceeded.") from exc
            raise ConnectionError(f"Groq API error: {exc}") from exc
        except Exception as exc:
            logger.error(f"Unexpected Groq error: {type(exc).__name__}: {exc}")
            raise ConnectionError(f"Unexpected Groq API error: {exc}") from exc
