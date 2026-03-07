# =============================================================================
# discussion.py — Dual-Context RAG Pipeline and Orchestrator
# =============================================================================
# Two responsibilities:
#
#   1. discussion_rag_query() — the dual-context RAG pipeline.
#      Reformulates the query, retrieves chunks, builds a prompt with
#      both document context and conversation history, generates a response.
#
#   2. process_message() — the top-level orchestrator.
#      Wires together trigger detection, throttling, queued trigger recovery,
#      summary generation, and the RAG pipeline. This is the single entry
#      point that main.py's WebSocket handler calls for every message.
#
# Extracted from: Notebook 2, Cells 6 (RAG pipeline) and 8 (orchestrator)

from config import TOP_K
from llm import chat
from retrieval import retrieve, format_context
from session import MultiUserSessionManager
from agent import (
    detect_trigger, reformulate_query, should_throttle,
    is_summary_request, generate_summary,
    detect_conversation_tempo,
    AdaptiveThrottler
)


# =============================================================================
# DISCUSSION AGENT SYSTEM PROMPT
# =============================================================================

DISCUSSION_SYSTEM_PROMPT = """You are a knowledgeable assistant participating in a team discussion. You have access to an uploaded document and are helping the team understand its contents.

GUIDELINES:
1. Be conversational — you're a colleague contributing to the discussion, not a search engine.
2. Ground your responses in the provided document context. Reference sections or findings naturally (e.g. "Based on the paper, the system uses..." or "Section 4 describes...").
3. If the context doesn't contain enough information, say so honestly — don't guess or make things up.
4. Keep responses focused and concise. The team is having a discussion, not reading an essay.
5. If a source is tagged [low-confidence source], treat it with caution and prefer other sources.
6. When correcting a misconception, be tactful — e.g. "Actually, the paper mentions..." rather than "You're wrong."
7. Address the specific point being discussed. Don't dump everything you know about a topic."""


# =============================================================================
# DUAL-CONTEXT RAG PIPELINE
# =============================================================================

def discussion_rag_query(
    session_id: str,
    message: str,
    user_id: str,
    trigger_type: str,
    session_manager: MultiUserSessionManager,
    collection,
    embedding_model,
    top_k: int = TOP_K
) -> dict:
    """
    Full discussion-aware RAG pipeline: reformulate → retrieve → generate.

    Uses the reformulated query for retrieval but passes the original
    message to the LLM so it responds naturally to what the user said.

    Args:
        session_id:      Active session
        message:         The original trigger message
        user_id:         Who sent it
        trigger_type:    From trigger detection — for logging
        session_manager: Session manager instance
        collection:      ChromaDB collection to search
        embedding_model: Loaded SentenceTransformer instance
        top_k:           Number of chunks to retrieve

    Returns:
        {
            "answer":           str,
            "original_message": str,
            "reformulated":     str,
            "was_rewritten":    bool,
            "trigger_type":     str,
            "sources":          list,
            "scores":           list,
            "session_id":       str
        }
    """
    # Step 1: Reformulate for better retrieval
    reform = reformulate_query(session_id, message, user_id, session_manager)

    # Step 2: Retrieve relevant chunks
    retrieved = retrieve(reform["reformulated"], collection, embedding_model, top_k=top_k)
    context = format_context(retrieved)

    # Step 3: Build dual-context prompt
    user_prompt = f"""Document context:
{context}

---

The following is a team discussion. Respond to the latest message using the document context above.

Latest message from {user_id}: {message}"""

    # Step 4: Assemble messages array with conversation history
    messages = [{"role": "system", "content": DISCUSSION_SYSTEM_PROMPT}]
    history = session_manager.get_history_for_llm(session_id)
    messages.extend(history)
    messages.append({"role": "user", "content": user_prompt})

    # Step 5: Generate response
    answer = chat(messages, timeout=60)

    # Step 6: Store bot response in session
    session_manager.add_bot_response(session_id, answer)

    return {
        "answer": answer,
        "original_message": message,
        "reformulated": reform["reformulated"],
        "was_rewritten": reform["was_rewritten"],
        "trigger_type": trigger_type,
        "sources": retrieved["metadatas"],
        "scores": retrieved["adjusted_scores"],
        "session_id": session_id
    }


# =============================================================================
# ORCHESTRATOR
# =============================================================================

def process_message(
    session_id: str,
    user_id: str,
    message: str,
    session_manager: MultiUserSessionManager,
    throttler: AdaptiveThrottler,
    collection,
    embedding_model,
    timestamp: float = None
) -> dict:
    """
    Main orchestrator — processes a single incoming message through the
    full discussion agent pipeline.

    Flow:
        1. Add message to session
        2. Check for summary request
        3. Check for queued triggers (if tempo has calmed)
        4. Run trigger detection
        5. Run throttling
        6. Based on action: respond / queue / throttle / summarize

    Args:
        session_id:      Active session
        user_id:         Who sent the message
        message:         Message text
        session_manager: Session manager instance
        throttler:       Adaptive throttler instance
        collection:      ChromaDB collection (can be None if no doc uploaded)
        embedding_model: Loaded SentenceTransformer instance
        timestamp:       Optional override for testing

    Returns:
        {
            "user_id":        str,
            "message":        str,
            "bot_responded":  bool,
            "bot_response":   str or None,
            "action":         str,
            "trigger":        dict or None,
            "throttle":       dict or None,
            "pipeline":       dict or None,
            "from_queue":     bool
        }
    """
    # Step 1: Add message to session
    session_manager.add_user_message(session_id, user_id, message)
    if timestamp is not None:
        session_manager.sessions[session_id][-1]["timestamp"] = timestamp

    result = {
        "user_id": user_id,
        "message": message,
        "bot_responded": False,
        "bot_response": None,
        "action": "no_trigger",
        "trigger": None,
        "throttle": None,
        "pipeline": None,
        "from_queue": False
    }

    # Step 2: Check for summary request BEFORE trigger detection
    if is_summary_request(message):
        summary = generate_summary(session_id, session_manager)
        gap = session_manager.messages_since_last_bot_response(session_id)
        throttler.record_bot_response(session_id, gap)

        result["bot_responded"] = True
        result["bot_response"] = summary
        result["action"] = "summary"
        return result

    # Step 3: Check for queued triggers
    tempo = detect_conversation_tempo(session_id, session_manager)
    if not tempo["is_rapid"] and throttler.has_queued_triggers(session_id):
        queued = throttler.pop_queued_triggers(session_id)
        latest = queued[-1]

        gap = session_manager.messages_since_last_bot_response(session_id)

        pipeline_result = discussion_rag_query(
            session_id,
            latest["message"],
            latest["user_id"],
            latest["trigger_type"],
            session_manager,
            collection,
            embedding_model
        )

        throttler.record_bot_response(session_id, gap)

        result["bot_responded"] = True
        result["bot_response"] = pipeline_result["answer"]
        result["action"] = "respond"
        result["pipeline"] = pipeline_result
        result["from_queue"] = True
        return result

    # No knowledge base — skip agent processing
    if collection is None or collection.count() == 0:
        return result

    # Step 4: Trigger detection
    trigger = detect_trigger(session_id, message, user_id, session_manager)
    result["trigger"] = trigger

    if not trigger["should_respond"]:
        result["action"] = "no_trigger"
        return result

    # Step 5: Throttling
    throttle = should_throttle(
        session_id, trigger["trigger_type"], message, user_id,
        session_manager, throttler
    )
    result["throttle"] = throttle
    result["action"] = throttle["action"]

    # Step 6: Act based on throttle decision
    if throttle["action"] == "throttle":
        return result

    elif throttle["action"] == "queue":
        return result

    elif throttle["action"] == "summary":
        summary = generate_summary(session_id, session_manager)
        gap = session_manager.messages_since_last_bot_response(session_id)
        throttler.record_bot_response(session_id, gap)

        result["bot_responded"] = True
        result["bot_response"] = summary
        return result

    elif throttle["action"] == "respond":
        gap = session_manager.messages_since_last_bot_response(session_id)

        pipeline_result = discussion_rag_query(
            session_id,
            message,
            user_id,
            trigger["trigger_type"],
            session_manager,
            collection,
            embedding_model
        )

        throttler.record_bot_response(session_id, gap)

        result["bot_responded"] = True
        result["bot_response"] = pipeline_result["answer"]
        result["pipeline"] = pipeline_result
        return result

    return result