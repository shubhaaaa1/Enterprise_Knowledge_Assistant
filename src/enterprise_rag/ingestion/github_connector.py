"""GitHub API source connector."""

from __future__ import annotations

import base64
import logging
import uuid
from datetime import datetime
from typing import Iterator, List, Optional

import requests

from enterprise_rag.ingestion.base import SourceConnector
from enterprise_rag.models import Document

logger = logging.getLogger(__name__)


class GitHubConnector(SourceConnector):
    """Fetches files from a GitHub repository via the GitHub REST API."""

    def __init__(
        self,
        source_id: str,
        repo: str,
        token: str,
        permission_tags: Optional[List[str]] = None,
        base_url: str = "https://api.github.com",
    ) -> None:
        self.source_id = source_id
        self.repo = repo
        self.token = token
        self.permission_tags: List[str] = permission_tags or []
        self.base_url = base_url.rstrip("/")

    @property
    def _headers(self) -> dict:
        return {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json",
        }

    def _get_blob_content(self, blob_url: str) -> Optional[str]:
        try:
            resp = requests.get(blob_url, headers=self._headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            encoding = data.get("encoding", "")
            raw = data.get("content", "")
            if encoding == "base64":
                return base64.b64decode(raw).decode("utf-8", errors="replace")
            return raw
        except Exception as exc:
            logger.warning("Failed to fetch blob %s: %s", blob_url, exc)
            return None

    def _make_document(self, path: str, content: str, modified_at: datetime) -> Document:
        return Document(
            doc_id=str(uuid.uuid4()),
            source_type="github",
            source_id=self.source_id,
            title=path.split("/")[-1],
            url=f"https://github.com/{self.repo}/blob/HEAD/{path}",
            content=content,
            permission_tags=list(self.permission_tags),
            modified_at=modified_at,
        )

    def fetch_all(self) -> Iterator[Document]:
        tree_url = f"{self.base_url}/repos/{self.repo}/git/trees/HEAD?recursive=1"
        try:
            resp = requests.get(tree_url, headers=self._headers, timeout=30)
            resp.raise_for_status()
        except Exception as exc:
            logger.error("Failed to fetch tree for %s: %s", self.repo, exc)
            return

        tree = resp.json().get("tree", [])
        for item in tree:
            if item.get("type") != "blob":
                continue
            path = item["path"]
            blob_url = item.get("url", "")
            content = self._get_blob_content(blob_url)
            if content is None:
                continue
            yield self._make_document(path, content, datetime.utcnow())

    def fetch_incremental(self, since: datetime) -> Iterator[Document]:
        since_str = since.strftime("%Y-%m-%dT%H:%M:%SZ")
        commits_url = (
            f"{self.base_url}/repos/{self.repo}/commits?since={since_str}&per_page=100"
        )
        try:
            resp = requests.get(commits_url, headers=self._headers, timeout=30)
            resp.raise_for_status()
        except Exception as exc:
            logger.error("Failed to fetch commits for %s: %s", self.repo, exc)
            return

        changed_paths: dict = {}
        for commit in resp.json():
            sha = commit.get("sha", "")
            commit_url = f"{self.base_url}/repos/{self.repo}/commits/{sha}"
            try:
                c_resp = requests.get(commit_url, headers=self._headers, timeout=30)
                c_resp.raise_for_status()
                commit_data = c_resp.json()
                commit_date_str = (
                    commit_data.get("commit", {})
                    .get("committer", {})
                    .get("date", "")
                )
                try:
                    commit_date = datetime.strptime(commit_date_str, "%Y-%m-%dT%H:%M:%SZ")
                except ValueError:
                    commit_date = datetime.utcnow()
                for f in commit_data.get("files", []):
                    fpath = f.get("filename", "")
                    if fpath and fpath not in changed_paths:
                        changed_paths[fpath] = {
                            "blob_url": f.get("blob_url", ""),
                            "modified_at": commit_date,
                        }
            except Exception as exc:
                logger.warning("Failed to fetch commit %s: %s", sha, exc)
                continue

        for path, meta in changed_paths.items():
            blob_url = meta["blob_url"]
            if not blob_url:
                continue
            content = self._get_blob_content(blob_url)
            if content is None:
                continue
            yield self._make_document(path, content, meta["modified_at"])
