# =============================================================================
# agent.py — Trigger Detection, Query Reformulation, Adaptive Throttling
# =============================================================================
# The discussion agent's decision-making layer. Three components:
#
#   1. Trigger detection (Cell 4) — should the bot respond to this message?
#   2. Query reformulation (Cell 5) — expand vague messages for better retrieval
#   3. Adaptive throttling (Cell 7) — prevent the bot from dominating
#
# All three are stateless functions that take a session_manager instance
# as an argument. The AdaptiveThrottler class holds per-session state
# for gap learning and trigger queuing.
#
# Extracted from: Notebook 2, Cells 4, 5, 7

import re
import time

from config import (
    BOT_NAMES, NON_TRIGGER_PHRASES, TRIGGER_CONTEXT_WINDOW,
    REFORMULATION_CONTEXT_TURNS, MIN_SPECIFIC_WORDS,
    VAGUE_PHRASE_SIGNALS, VAGUE_WORD_SIGNALS,
    MIN_RESPONSE_GAP, MAX_RESPONSE_GAP, MAX_BOT_RATIO,
    RATIO_WINDOW, GAP_HISTORY_SIZE,
    RAPID_TEMPO_THRESHOLD, TEMPO_WINDOW,
    SUMMARY_TRIGGER_PHRASES
)
from llm import chat_with_system
from session import MultiUserSessionManager


# =============================================================================
# LLM PROMPTS
# =============================================================================

TRIGGER_CLASSIFIER_SYSTEM_PROMPT = """You are a conversation monitor. Your job is to decide whether a new message in a group discussion requires a factual response from a document-aware assistant.

The assistant has access to an uploaded document and should ONLY respond when document knowledge would be helpful.

Respond with EXACTLY one line in this format:
DECISION: YES or NO
TYPE: DIRECT_TAG | QUESTION | UNCERTAINTY | DISAGREEMENT | CLARIFICATION | NONE
REASON: one short phrase (max 5 words)

Triggers (respond YES):
- Direct mention of the bot/assistant
- Questions about document content (methods, results, tools, data, architecture)
- Uncertainty or speculation about something the document could clarify ("I think they used X but not sure")
- Factual disagreement between users that the document could resolve
- Requests for explanation of something covered in the document

Non-triggers (respond NO):
- Opinions and preferences ("I think this approach is better")
- Agreement or acknowledgment ("yeah", "makes sense", "good point")
- Casual chat, greetings, meta-discussion ("let's take a break", "should we move on")
- Questions unrelated to any uploaded document content"""

REFORMULATION_SYSTEM_PROMPT = """You are a search query optimizer. Your job is to rewrite a conversational message into a clear, self-contained search query for retrieving information from an academic/technical document.

You will see recent conversation context and a new message. Rewrite the new message so it:
1. Replaces pronouns (it, they, that, this) with the actual subjects from context
2. Includes key topic terms from the discussion
3. Is a single, concise query (under 20 words)
4. Works as a standalone search — someone reading ONLY the query should understand what's being asked

Respond with ONLY the reformulated query. No explanation, no quotes, no preamble."""

SUMMARY_SYSTEM_PROMPT = "You are a helpful meeting assistant. Summarize discussions concisely."


# =============================================================================
# TRIGGER DETECTION
# =============================================================================

def _rule_based_check(message: str) -> tuple[str, str] | None:
    """
    Fast-path rule-based trigger detection. No LLM call needed.

    Returns:
        ("trigger", trigger_type) if definitely a trigger
        ("skip", "NONE") if definitely NOT a trigger
        None if ambiguous — needs LLM classification
    """
    msg_lower = message.strip().lower()

    # Direct tag: always trigger
    for bot_name in BOT_NAMES:
        if bot_name in msg_lower:
            return ("trigger", "DIRECT_TAG")

    # Very short messages: almost never triggers
    if len(msg_lower.split()) <= 2:
        if msg_lower.endswith("?"):
            return None  # Ambiguous — let LLM decide
        return ("skip", "NONE")

    # Known non-trigger phrases
    for phrase in NON_TRIGGER_PHRASES:
        if msg_lower == phrase or msg_lower.startswith(phrase + " "):
            return ("skip", "NONE")

    return None


def _llm_classify(message: str, context_messages: list[dict]) -> dict:
    """
    Use the LLM to classify whether a message should trigger a bot response.

    Args:
        message:          The new message to classify
        context_messages: Recent conversation messages for context

    Returns:
        {"decision": "YES"|"NO", "type": str, "reason": str}
    """
    # Build context string
    context_lines = []
    for msg in context_messages:
        if msg["role"] == "assistant":
            context_lines.append(f"[assistant]: {msg['content']}")
        else:
            context_lines.append(f"[{msg['user_id']}]: {msg['content']}")

    context_str = "\n".join(context_lines) if context_lines else "(no prior context)"

    user_prompt = f"""Recent conversation:
{context_str}

New message to classify:
{message}

Should the document-aware assistant respond to this new message?"""

    try:
        raw = chat_with_system(TRIGGER_CLASSIFIER_SYSTEM_PROMPT, user_prompt, timeout=30)

        # Parse the structured response
        result = {"decision": "NO", "type": "NONE", "reason": "parse_failed"}

        # Valid trigger types in priority order — most specific first
        valid_types_priority = [
            "DISAGREEMENT", "UNCERTAINTY", "CLARIFICATION",
            "QUESTION", "DIRECT_TAG", "NONE"
        ]

        for line in raw.split("\n"):
            line = line.strip()
            if line.upper().startswith("DECISION:"):
                val = line.split(":", 1)[1].strip().upper()
                result["decision"] = "YES" if "YES" in val else "NO"
            elif line.upper().startswith("TYPE:"):
                type_str = line.split(":", 1)[1].strip().upper()
                matched = [t for t in valid_types_priority if t in type_str]
                result["type"] = matched[0] if matched else "NONE"
            elif line.upper().startswith("REASON:"):
                result["reason"] = line.split(":", 1)[1].strip()

        # Enforce consistency: NO decision → NONE type
        if result["decision"] == "NO":
            result["type"] = "NONE"

        # Override: DIRECT_TAG without actual bot mention → QUESTION
        if result["type"] == "DIRECT_TAG":
            msg_lower = message.strip().lower()
            if not any(name in msg_lower for name in BOT_NAMES):
                result["type"] = "QUESTION"

        return result

    except Exception as e:
        print(f"  ⚠ Trigger classifier error: {e}")
        return {"decision": "NO", "type": "NONE", "reason": f"error: {e}"}


def detect_trigger(
    session_id: str,
    message: str,
    user_id: str,
    session_manager: MultiUserSessionManager
) -> dict:
    """
    Determine whether a new message should trigger a bot response.

    Two-stage pipeline:
        1. Rule-based fast path (instant)
        2. LLM classification (3-10 seconds, only if needed)

    Args:
        session_id:      The active session
        message:         The new message text
        user_id:         Who sent it
        session_manager: Session manager instance

    Returns:
        {
            "should_respond": bool,
            "trigger_type":   str,
            "reason":         str,
            "method":         "rule" or "llm"
        }
    """
    # Stage 1: Rule-based fast path
    rule_result = _rule_based_check(message)

    if rule_result is not None:
        action, trigger_type = rule_result
        return {
            "should_respond": action == "trigger",
            "trigger_type": trigger_type,
            "reason": "direct_mention" if trigger_type == "DIRECT_TAG" else "rule_filtered",
            "method": "rule"
        }

    # Stage 2: LLM classification
    context = session_manager.get_recent_messages(session_id, TRIGGER_CONTEXT_WINDOW)
    llm_result = _llm_classify(message, context)

    return {
        "should_respond": llm_result["decision"] == "YES",
        "trigger_type": llm_result["type"],
        "reason": llm_result["reason"],
        "method": "llm"
    }


# =============================================================================
# QUERY REFORMULATION
# =============================================================================

def _needs_reformulation(message: str) -> bool:
    """
    Decide whether a message needs LLM reformulation or is already
    specific enough for direct retrieval.
    """
    msg_lower = message.strip().lower()
    words = msg_lower.split()

    if len(words) < MIN_SPECIFIC_WORDS:
        return True

    for phrase in VAGUE_PHRASE_SIGNALS:
        if phrase in msg_lower:
            return True

    for pronoun in VAGUE_WORD_SIGNALS:
        if re.search(rf'\b{pronoun}\b', msg_lower):
            return True

    return False


def reformulate_query(
    session_id: str,
    message: str,
    user_id: str,
    session_manager: MultiUserSessionManager
) -> dict:
    """
    Reformulate a trigger message into a retrieval-ready query.

    If the message is already specific, returns it unchanged.
    Otherwise, uses the LLM to expand it with conversation context.

    Args:
        session_id:      Active session for conversation context
        message:         The trigger message to reformulate
        user_id:         Who sent it
        session_manager: Session manager instance

    Returns:
        {
            "original":      str,
            "reformulated":  str,
            "was_rewritten": bool
        }
    """
    if not _needs_reformulation(message):
        return {
            "original": message,
            "reformulated": message,
            "was_rewritten": False
        }

    # Get recent conversation context
    recent = session_manager.get_recent_messages(
        session_id, REFORMULATION_CONTEXT_TURNS * 2
    )

    context_lines = []
    for msg in recent:
        if msg["role"] == "assistant":
            context_lines.append(f"[assistant]: {msg['content']}")
        else:
            context_lines.append(f"[{msg['user_id']}]: {msg['content']}")

    context_str = "\n".join(context_lines) if context_lines else "(no prior context)"

    user_prompt = f"""Recent conversation:
{context_str}

New message to reformulate:
[{user_id}]: {message}

Rewrite this as a clear, self-contained search query:"""

    try:
        reformulated = chat_with_system(
            REFORMULATION_SYSTEM_PROMPT, user_prompt, timeout=30
        )
        reformulated = reformulated.strip('"').strip("'").strip()

        if len(reformulated) == 0 or len(reformulated.split()) > 30:
            return {
                "original": message,
                "reformulated": message,
                "was_rewritten": False
            }

        return {
            "original": message,
            "reformulated": reformulated,
            "was_rewritten": True
        }

    except Exception as e:
        print(f"  ⚠ Reformulation error: {e}")
        return {
            "original": message,
            "reformulated": message,
            "was_rewritten": False
        }


# =============================================================================
# ADAPTIVE THROTTLING
# =============================================================================

class AdaptiveThrottler:
    """
    Tracks bot response patterns and conversation tempo per session.
    Maintains gap history for adaptive threshold calculation and
    a queue for triggers that fire during rapid exchanges.
    """

    def __init__(self):
        self.session_data: dict[str, dict] = {}

    def _ensure_session(self, session_id: str):
        """Initialize tracking data for a new session."""
        if session_id not in self.session_data:
            self.session_data[session_id] = {
                "gap_history": [],
                "queued_triggers": []
            }

    def record_bot_response(self, session_id: str, gap: int):
        """
        Record that the bot responded after `gap` user messages.
        Updates the rolling gap history for adaptive threshold calculation.
        """
        self._ensure_session(session_id)
        history = self.session_data[session_id]["gap_history"]
        history.append(gap)
        if len(history) > GAP_HISTORY_SIZE:
            self.session_data[session_id]["gap_history"] = history[-GAP_HISTORY_SIZE:]

    def get_adaptive_gap(self, session_id: str) -> int:
        """
        Compute the current adaptive minimum gap for this session.

        Uses rolling average of recent gaps, clamped to [MIN, MAX].
        Applies momentum nudge when gaps are trending upward.
        """
        self._ensure_session(session_id)
        history = self.session_data[session_id]["gap_history"]

        if not history:
            return MIN_RESPONSE_GAP

        avg = sum(history) / len(history)

        # Momentum: if last 3+ gaps are strictly increasing, nudge up
        if len(history) >= 3:
            recent = history[-3:]
            if all(recent[i] < recent[i + 1] for i in range(len(recent) - 1)):
                avg += 1.0

        return max(MIN_RESPONSE_GAP, min(MAX_RESPONSE_GAP, round(avg)))

    def queue_trigger(self, session_id: str, message: str, user_id: str, trigger_type: str):
        """Store a trigger that fired during rapid exchange for later."""
        self._ensure_session(session_id)
        self.session_data[session_id]["queued_triggers"].append({
            "message": message,
            "user_id": user_id,
            "trigger_type": trigger_type,
            "timestamp": time.time()
        })

    def pop_queued_triggers(self, session_id: str) -> list[dict]:
        """Retrieve and clear all queued triggers for a session."""
        self._ensure_session(session_id)
        triggers = self.session_data[session_id]["queued_triggers"]
        self.session_data[session_id]["queued_triggers"] = []
        return triggers

    def has_queued_triggers(self, session_id: str) -> bool:
        """Check if there are triggers waiting for a pause."""
        self._ensure_session(session_id)
        return len(self.session_data[session_id]["queued_triggers"]) > 0

    def get_session_stats(self, session_id: str) -> dict:
        """Return current throttling state for debugging."""
        self._ensure_session(session_id)
        data = self.session_data[session_id]
        return {
            "gap_history": list(data["gap_history"]),
            "adaptive_gap": self.get_adaptive_gap(session_id),
            "queued_triggers": len(data["queued_triggers"])
        }


# =============================================================================
# TEMPO DETECTION
# =============================================================================

def detect_conversation_tempo(
    session_id: str,
    session_manager: MultiUserSessionManager
) -> dict:
    """
    Measure the tempo of recent conversation by looking at time deltas
    between user messages.

    Returns:
        {
            "is_rapid":   bool,
            "avg_delta":  float (seconds between recent messages),
            "num_deltas": int
        }
    """
    recent = session_manager.get_recent_messages(session_id, TEMPO_WINDOW + 1)
    user_msgs = [m for m in recent if m["role"] == "user"]

    if len(user_msgs) < 2:
        return {"is_rapid": False, "avg_delta": float("inf"), "num_deltas": 0}

    deltas = []
    for i in range(1, len(user_msgs)):
        delta = user_msgs[i]["timestamp"] - user_msgs[i - 1]["timestamp"]
        deltas.append(delta)

    avg_delta = sum(deltas) / len(deltas)

    return {
        "is_rapid": avg_delta < RAPID_TEMPO_THRESHOLD,
        "avg_delta": avg_delta,
        "num_deltas": len(deltas)
    }


# =============================================================================
# SUMMARY DETECTION AND GENERATION
# =============================================================================

def is_summary_request(message: str) -> bool:
    """Check if a message signals the user wants a conversation summary."""
    msg_lower = message.strip().lower()
    for phrase in SUMMARY_TRIGGER_PHRASES:
        if phrase in msg_lower:
            return True
    return False


def generate_summary(
    session_id: str,
    session_manager: MultiUserSessionManager
) -> str:
    """
    Generate a summary of the conversation so far.
    Uses conversation history (not document retrieval) to produce a recap.

    Returns:
        The summary text from the LLM.
    """
    history = session_manager.get_history_for_llm(session_id)

    if len(history) < 2:
        return "There hasn't been enough discussion to summarize yet."

    convo_text = "\n".join(
        f"{msg['role'].upper()}: {msg['content']}" for msg in history
    )

    summary_prompt = f"""Here is a team discussion. Provide a concise summary covering:
1. Main topics discussed
2. Key questions raised and whether they were answered
3. Any points of disagreement or uncertainty
4. Open questions still unanswered

Keep it brief — 4-6 sentences maximum.

Conversation:
{convo_text}

Summary:"""

    try:
        summary = chat_with_system(SUMMARY_SYSTEM_PROMPT, summary_prompt, timeout=60)

        session_manager.add_bot_response(
            session_id, f"📋 Discussion Summary:\n{summary}"
        )

        return summary

    except Exception as e:
        print(f"  ⚠ Summary generation error: {e}")
        return "Sorry, I couldn't generate a summary right now."


# =============================================================================
# THROTTLE DECISION
# =============================================================================

def should_throttle(
    session_id: str,
    trigger_type: str,
    message: str,
    user_id: str,
    session_manager: MultiUserSessionManager,
    throttler: AdaptiveThrottler
) -> dict:
    """
    Decide what the bot should do with a valid trigger.

    Args:
        session_id:      Active session
        trigger_type:    From detect_trigger() (DIRECT_TAG bypasses)
        message:         The trigger message (checked for summary requests)
        user_id:         Who sent it
        session_manager: Session manager instance
        throttler:       Adaptive throttler instance

    Returns:
        {
            "action":       "respond" | "queue" | "throttle" | "summary",
            "reason":       str,
            "gap":          int,
            "adaptive_gap": int,
            "ratio":        float,
            "tempo":        dict
        }
    """
    # Summary detection (highest priority after direct tag)
    if is_summary_request(message):
        return {
            "action": "summary",
            "reason": "summary_requested",
            "gap": session_manager.messages_since_last_bot_response(session_id),
            "adaptive_gap": throttler.get_adaptive_gap(session_id),
            "ratio": session_manager.get_bot_message_ratio(session_id),
            "tempo": detect_conversation_tempo(session_id, session_manager)
        }

    # Direct tag: always respond immediately
    if trigger_type == "DIRECT_TAG":
        return {
            "action": "respond",
            "reason": "direct_tag_bypass",
            "gap": session_manager.messages_since_last_bot_response(session_id),
            "adaptive_gap": throttler.get_adaptive_gap(session_id),
            "ratio": session_manager.get_bot_message_ratio(session_id),
            "tempo": detect_conversation_tempo(session_id, session_manager)
        }

    gap = session_manager.messages_since_last_bot_response(session_id)
    adaptive_gap = throttler.get_adaptive_gap(session_id)
    ratio = session_manager.get_bot_message_ratio(session_id)
    tempo = detect_conversation_tempo(session_id, session_manager)

    # Check 1: Adaptive gap
    if gap < adaptive_gap:
        return {
            "action": "throttle",
            "reason": f"adaptive_gap ({gap} < {adaptive_gap})",
            "gap": gap,
            "adaptive_gap": adaptive_gap,
            "ratio": ratio,
            "tempo": tempo
        }

    # Check 2: Bot ratio
    if ratio > MAX_BOT_RATIO:
        return {
            "action": "throttle",
            "reason": f"bot_ratio ({ratio:.0%} > {MAX_BOT_RATIO:.0%})",
            "gap": gap,
            "adaptive_gap": adaptive_gap,
            "ratio": ratio,
            "tempo": tempo
        }

    # Check 3: Tempo — queue if conversation is rapid
    if tempo["is_rapid"]:
        throttler.queue_trigger(session_id, message, user_id, trigger_type)
        return {
            "action": "queue",
            "reason": f"rapid_tempo (avg {tempo['avg_delta']:.1f}s < {RAPID_TEMPO_THRESHOLD}s)",
            "gap": gap,
            "adaptive_gap": adaptive_gap,
            "ratio": ratio,
            "tempo": tempo
        }

    # All checks passed
    return {
        "action": "respond",
        "reason": "ok",
        "gap": gap,
        "adaptive_gap": adaptive_gap,
        "ratio": ratio,
        "tempo": tempo
    }