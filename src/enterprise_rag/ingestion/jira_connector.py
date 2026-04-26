"""Jira REST API source connector."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Iterator, List, Optional

import requests

from enterprise_rag.ingestion.base import SourceConnector
from enterprise_rag.models import Document

logger = logging.getLogger(__name__)


class JiraConnector(SourceConnector):
    """Fetches issues from a Jira project via the Jira REST API."""

    def __init__(
        self,
        source_id: str,
        base_url: str,
        username: str,
        api_token: str,
        project_key: str,
        permission_tags: Optional[List[str]] = None,
    ) -> None:
        self.source_id = source_id
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.api_token = api_token
        self.project_key = project_key
        self.permission_tags: List[str] = permission_tags or []

    @property
    def _auth(self):
        return (self.username, self.api_token)

    def _fetch_comments(self, issue_key: str) -> str:
        url = f"{self.base_url}/rest/api/2/issue/{issue_key}/comment"
        try:
            resp = requests.get(url, auth=self._auth, timeout=30)
            resp.raise_for_status()
            comments = resp.json().get("comments", [])
            return "\n".join(c.get("body", "") for c in comments)
        except Exception as exc:
            logger.warning("Failed to fetch comments for %s: %s", issue_key, exc)
            return ""

    def _search_issues(self, jql: str) -> Iterator[dict]:
        start_at = 0
        max_results = 100
        while True:
            url = (
                f"{self.base_url}/rest/api/2/search"
                f"?jql={jql}&maxResults={max_results}&startAt={start_at}"
            )
            try:
                resp = requests.get(url, auth=self._auth, timeout=30)
                resp.raise_for_status()
            except Exception as exc:
                logger.error("Jira search failed (jql=%s): %s", jql, exc)
                return

            data = resp.json()
            issues = data.get("issues", [])
            yield from issues

            total = data.get("total", 0)
            start_at += len(issues)
            if start_at >= total or not issues:
                break

    def _make_document(self, issue: dict) -> Document:
        fields = issue.get("fields", {})
        key = issue.get("key", "")
        summary = fields.get("summary", key)
        description = fields.get("description") or ""
        comments = self._fetch_comments(key)
        content = description
        if comments:
            content = f"{description}\n\nComments:\n{comments}"

        updated_str = fields.get("updated", "")
        try:
            modified_at = datetime.strptime(updated_str[:19], "%Y-%m-%dT%H:%M:%S")
        except (ValueError, TypeError):
            modified_at = datetime.now(timezone.utc)

        return Document(
            doc_id=str(uuid.uuid4()),
            source_type="jira",
            source_id=self.source_id,
            title=summary,
            url=f"{self.base_url}/browse/{key}",
            content=content,
            permission_tags=list(self.permission_tags),
            modified_at=modified_at,
        )

    def fetch_all(self) -> Iterator[Document]:
        jql = f"project={self.project_key}"
        for issue in self._search_issues(jql):
            yield self._make_document(issue)

    def fetch_incremental(self, since: datetime) -> Iterator[Document]:
        since_str = since.strftime("%Y-%m-%d")
        jql = f'project={self.project_key} AND updated >= "{since_str}"'
        for issue in self._search_issues(jql):
            yield self._make_document(issue)
