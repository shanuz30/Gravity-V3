"""
[CHAT HISTORY & SEARCH]
Persists session interactions and provides keyword search over recent chats.
Integrates with session_loop to record each interaction automatically.
"""

import json
import os
import time
from dataclasses import dataclass, asdict, field
from typing import Optional


@dataclass
class ChatEntry:
    timestamp: float
    user_input: str
    result: str
    status: str  # "SUCCESS" | "ESCALATED" | "INSUFFICIENT"
    session_id: Optional[str] = None


class ChatHistoryStore:
    """
    Persists ChatEntry records to a JSON file and supports keyword search.
    Thread-safety is not guaranteed — intended for single-process use.
    """

    def __init__(self, storage_path: str = "chat_history.json"):
        self.storage_path = storage_path
        if not os.path.exists(self.storage_path):
            self._write([])

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def record(self, entry: ChatEntry) -> None:
        entries = self._read()
        entries.append(asdict(entry))
        self._write(entries)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_recent(self, limit: int = 10) -> list[ChatEntry]:
        """Return the *limit* most recent entries, newest first."""
        entries = self._read()
        entries.sort(key=lambda e: e["timestamp"], reverse=True)
        return [ChatEntry(**e) for e in entries[:limit]]

    def search(self, query: str, limit: int = 10) -> list[ChatEntry]:
        """
        Case-insensitive keyword search over user_input and result fields.
        Returns the *limit* most recent matches, newest first.
        """
        if not query or not query.strip():
            return self.get_recent(limit)

        q = query.strip().lower()
        entries = self._read()
        matches = [
            e for e in entries
            if q in e["user_input"].lower() or q in e["result"].lower()
        ]
        matches.sort(key=lambda e: e["timestamp"], reverse=True)
        return [ChatEntry(**e) for e in matches[:limit]]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read(self) -> list[dict]:
        with open(self.storage_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _write(self, entries: list[dict]) -> None:
        with open(self.storage_path, "w", encoding="utf-8") as fh:
            json.dump(entries, fh, indent=2)


def _resolve_status(result) -> str:
    if isinstance(result, dict):
        return result.get("status", "ESCALATED")
    if isinstance(result, str) and "INSUFFICIENT VARIABLES:" in result:
        return "INSUFFICIENT"
    return "SUCCESS"


def record_session_result(
    store: ChatHistoryStore,
    user_input: str,
    result,
    session_id: Optional[str] = None,
) -> None:
    """Convenience wrapper used by session_loop to persist an interaction."""
    result_str = json.dumps(result) if isinstance(result, dict) else str(result)
    entry = ChatEntry(
        timestamp=time.time(),
        user_input=user_input,
        result=result_str,
        status=_resolve_status(result),
        session_id=session_id,
    )
    store.record(entry)
