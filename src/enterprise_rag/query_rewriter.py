"""Query rewriter component for the Enterprise RAG System.

Rewrites/expands user queries using Groq API to improve retrieval recall.
Incorporates the last 5 turns of conversation history for context resolution.
"""

from __future__ import annotations

import logging
from typing import List

import requests

from enterprise_rag.models import Turn

logger = logging.getLogger(__name__)


class QueryRewriter:
    """Rewrites user queries using Groq API.

    On timeout or any error, falls back to returning the original query
    so retrieval can always proceed.
    """

    _PROMPT_TEMPLATE = (
        "Given the conversation history and the user's question, generate "
        "{max_variants} alternative phrasings of the question that preserve "
        "the original meaning. Return one variant per line, no numbering.\n\n"
        "History:\n{history}\n\n"
        "Question: {query}\n\n"
        "Variants:"
    )

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.1-8b-instant",
        timeout: float = 5.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rewrite(
        self,
        query: str,
        history: List[Turn],
        max_variants: int = 3,
    ) -> List[str]:
        """Return a list of query variants starting with the original query.

        Args:
            query: The user's original query text.
            history: Full conversation history; only the last 5 turns are used.
            max_variants: Number of alternative phrasings to request from Groq.

        Returns:
            A list with at least one element (the original query).
            On success, the original query is prepended to the LLM variants.
        """
        prompt = self._build_prompt(query, history, max_variants)

        try:
            response = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            raw_text: str = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            variants = self._parse_variants(raw_text, max_variants)
            # Always prepend the original query as the first variant
            return [query] + variants

        except requests.Timeout:
            logger.warning(
                "QueryRewriter timeout after %.1fs for query=%r",
                self.timeout,
                query,
            )
            return [query]

        except requests.HTTPError as exc:
            if exc.response.status_code == 401:
                logger.error("QueryRewriter: Invalid Groq API key")
            elif exc.response.status_code == 429:
                logger.error("QueryRewriter: Groq rate limit exceeded")
            else:
                logger.error("QueryRewriter HTTP error for query=%r: %s", query, exc)
            return [query]

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "QueryRewriter error for query=%r: %s",
                query,
                exc,
            )
            return [query]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        query: str,
        history: List[Turn],
        max_variants: int,
    ) -> str:
        last_5 = history[-5:] if len(history) > 5 else history
        history_text = "\n".join(
            f"{turn.role}: {turn.original_query}" for turn in last_5
        )
        return self._PROMPT_TEMPLATE.format(
            max_variants=max_variants,
            history=history_text,
            query=query,
        )

    @staticmethod
    def _parse_variants(raw_text: str, max_variants: int) -> List[str]:
        lines = [line.strip() for line in raw_text.splitlines()]
        non_empty = [line for line in lines if line]
        return non_empty[:max_variants]
