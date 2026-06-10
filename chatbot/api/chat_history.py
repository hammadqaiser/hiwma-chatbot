"""Chat History — SQLite-based session storage.

Persists chat conversations with session management.
Each session has a unique ID and stores message history.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path


class ChatHistory:
    """SQLite-backed chat history storage.

    Thread-safe for use with FastAPI's async workers.
    Creates the database and tables automatically.
    """

    def __init__(self, database_path: str = "./data/chat_history.db"):
        self.database_path = database_path
        Path(database_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Create tables if they don't exist."""
        with sqlite3.connect(self.database_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    sources TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)
            conn.commit()

    def create_session(self) -> str:
        """Create a new chat session and return its ID."""
        session_id = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()
        with sqlite3.connect(self.database_path) as conn:
            conn.execute(
                "INSERT INTO sessions (session_id, created_at, updated_at) VALUES (?, ?, ?)",
                (session_id, now, now),
            )
            conn.commit()
        return session_id

    def session_exists(self, session_id: str) -> bool:
        """Check if a session exists."""
        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.execute(
                "SELECT 1 FROM sessions WHERE session_id = ?",
                (session_id,),
            )
            return cursor.fetchone() is not None

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        sources: list[dict] | None = None,
    ) -> None:
        """Add a message to a session.

        Args:
            session_id: The session to add to.
            role: 'user' or 'assistant'.
            content: Message text.
            sources: Optional source citations (for assistant messages).
        """
        now = datetime.now(UTC).isoformat()
        sources_json = json.dumps(sources) if sources else None

        with sqlite3.connect(self.database_path) as conn:
            conn.execute(
                "INSERT INTO messages (session_id, role, content, sources, created_at) VALUES (?, ?, ?, ?, ?)",
                (session_id, role, content, sources_json, now),
            )
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (now, session_id),
            )
            conn.commit()

    def get_messages(self, session_id: str, limit: int = 50) -> list[dict]:
        """Get message history for a session.

        Args:
            session_id: The session to retrieve.
            limit: Max messages to return.

        Returns:
            List of message dicts with role, content, sources, created_at.
        """
        with sqlite3.connect(self.database_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT role, content, sources, created_at FROM messages "
                "WHERE session_id = ? ORDER BY id ASC LIMIT ?",
                (session_id, limit),
            )
            rows = cursor.fetchall()

        return [
            {
                "role": row["role"],
                "content": row["content"],
                "sources": json.loads(row["sources"]) if row["sources"] else None,
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def get_session_count(self) -> int:
        """Return total number of sessions."""
        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM sessions")
            return cursor.fetchone()[0]
