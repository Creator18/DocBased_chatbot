# =============================================================================
# session.py — Multi-User Session Manager
# =============================================================================
# Manages multi-user conversation sessions in memory.
# Each session stores a flat list of messages in chronological order
# with user_id, role, content, and timestamp.
#
# Provides helpers for:
#   - Conversation history formatted for Ollama
#   - Recent message retrieval for trigger detection
#   - Throttling metrics (gap since last bot response, bot message ratio)
#
# For production: swap the in-memory dict for Redis or SQLite.
#
# Extracted from: Notebook 2, Cell 3

import time
import uuid

from config import MAX_HISTORY_TURNS, RATIO_WINDOW, AGENT_USER_ID


class MultiUserSessionManager:
    """
    Manages multi-user conversation sessions.

    Each session stores a flat list of messages in chronological order:
        [
            {"role": "user",      "user_id": "alice", "content": "...", "timestamp": float},
            {"role": "user",      "user_id": "bob",   "content": "...", "timestamp": float},
            {"role": "assistant", "user_id": "agent",  "content": "...", "timestamp": float},
            ...
        ]

    "role" follows Ollama's convention (user/assistant) so history can be
    passed directly to /api/chat. "user_id" distinguishes who said what.
    """

    def __init__(self, max_history_turns: int = MAX_HISTORY_TURNS):
        self.sessions: dict[str, list[dict]] = {}
        self.max_history_turns = max_history_turns

    def create_session(self) -> str:
        """Create a new session and return its ID."""
        session_id = uuid.uuid4().hex[:12]
        self.sessions[session_id] = []
        return session_id

    def add_user_message(self, session_id: str, user_id: str, content: str):
        """Record a message from a human user."""
        if session_id not in self.sessions:
            raise KeyError(f"Session '{session_id}' not found")

        self.sessions[session_id].append({
            "role": "user",
            "user_id": user_id,
            "content": content,
            "timestamp": time.time()
        })

    def add_bot_response(self, session_id: str, content: str):
        """Record a response from the agent."""
        if session_id not in self.sessions:
            raise KeyError(f"Session '{session_id}' not found")

        self.sessions[session_id].append({
            "role": "assistant",
            "user_id": AGENT_USER_ID,
            "content": content,
            "timestamp": time.time()
        })

    def get_recent_messages(self, session_id: str, n: int) -> list[dict]:
        """
        Get the last n messages (all roles) in chronological order.
        Used by trigger detection to see recent conversation context.
        """
        if session_id not in self.sessions:
            return []
        return self.sessions[session_id][-n:]

    def get_history_for_llm(self, session_id: str) -> list[dict]:
        """
        Get conversation history formatted for Ollama's /api/chat.
        Returns the last max_history_turns * 2 messages as
        [{"role": ..., "content": ...}] — strips user_id and timestamp.

        User messages are prefixed with [username] so the LLM knows
        who said what in the discussion.
        """
        if session_id not in self.sessions:
            return []

        messages = self.sessions[session_id]
        recent = messages[-(self.max_history_turns * 2):]

        formatted = []
        for msg in recent:
            if msg["role"] == "assistant":
                formatted.append({
                    "role": "assistant",
                    "content": msg["content"]
                })
            else:
                formatted.append({
                    "role": "user",
                    "content": f"[{msg['user_id']}]: {msg['content']}"
                })

        return formatted

    def messages_since_last_bot_response(self, session_id: str) -> int:
        """
        Count user messages since the bot last spoke.
        Returns a large number if the bot has never spoken.
        Used by response throttling.
        """
        if session_id not in self.sessions:
            return 999

        count = 0
        for msg in reversed(self.sessions[session_id]):
            if msg["role"] == "assistant":
                break
            count += 1

        return count

    def get_bot_message_ratio(self, session_id: str, window: int = RATIO_WINDOW) -> float:
        """
        Fraction of the last `window` messages that are bot responses.
        Used by response throttling to detect if the bot is dominating.
        """
        if session_id not in self.sessions:
            return 0.0

        recent = self.sessions[session_id][-window:]
        if not recent:
            return 0.0

        bot_count = sum(1 for m in recent if m["role"] == "assistant")
        return bot_count / len(recent)

    def get_full_log(self, session_id: str) -> list[dict]:
        """Return the complete message log for a session."""
        return self.sessions.get(session_id, [])

    def list_sessions(self) -> dict:
        """Return all sessions with message counts."""
        return {sid: len(msgs) for sid, msgs in self.sessions.items()}

    def delete_session(self, session_id: str):
        """Delete a session entirely."""
        self.sessions.pop(session_id, None)