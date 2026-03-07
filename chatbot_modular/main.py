# =============================================================================
# main.py — FastAPI App, Routes, WebSocket Handler, Startup
# =============================================================================
# Entry point for the application. Wires together all modules and serves
# the chat frontend.
#
# Run with:
#   python main.py
#
# Or with uvicorn directly:
#   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
#
# Extracted from: Notebook 2, Cells 9 (server) and 10 (HTML)

import os
import shutil
import asyncio
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from sentence_transformers import SentenceTransformer
import chromadb

from config import (
    OLLAMA_BASE_URL, OLLAMA_MODEL,
    EMBEDDING_MODEL_NAME, EMBEDDING_DIMENSIONS,
    CHROMA_PERSIST_DIR, CHROMA_COLLECTION,
    UPLOAD_DIR, SERVER_HOST, SERVER_PORT
)
from llm import check_connection
from ingestion import upload_and_ingest
from session import MultiUserSessionManager
from agent import AdaptiveThrottler
from discussion import process_message


# =============================================================================
# APPLICATION STATE
# =============================================================================
# These are created once at startup and shared across all requests.

# Embedding model — loaded once, used by ingestion and retrieval
embedding_model = None

# ChromaDB client and active collection
chroma_client = None
collection = None
doc_info = None

# Session manager and throttler — shared across all WebSocket connections
session_manager = MultiUserSessionManager()
throttler = AdaptiveThrottler()


# =============================================================================
# STARTUP
# =============================================================================

def initialize():
    """
    Load models and connect to services on startup.
    Called once when the server starts.
    """
    global embedding_model, chroma_client, collection

    print(f"\n{'=' * 60}")
    print("STARTING DISCUSSION AGENT")
    print(f"{'=' * 60}\n")

    # --- Check Ollama ---
    print("Checking Ollama connection...")
    status = check_connection()
    if status["reachable"]:
        print(f"  ✓ Ollama reachable")
        print(f"  Models: {', '.join(status['models'])}")
        if not status["has_model"]:
            print(f"  ⚠ '{OLLAMA_MODEL}' not found — run: ollama pull {OLLAMA_MODEL}")
    else:
        print(f"  ⚠ Ollama not reachable at {OLLAMA_BASE_URL}")

    # --- Load embedding model ---
    print(f"\nLoading embedding model: {EMBEDDING_MODEL_NAME}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    test_emb = embedding_model.encode("Dimension check.")
    assert test_emb.shape[0] == EMBEDDING_DIMENSIONS, \
        f"Expected {EMBEDDING_DIMENSIONS}d, got {test_emb.shape[0]}d"
    print(f"  ✓ Embedding model loaded ({EMBEDDING_DIMENSIONS}d)")

    # --- Connect ChromaDB ---
    print(f"\nConnecting ChromaDB at {CHROMA_PERSIST_DIR}")
    chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    print(f"  ✓ ChromaDB connected")

    # --- Check for existing collection ---
    try:
        collection = chroma_client.get_collection(name=CHROMA_COLLECTION)
        print(f"  Existing collection '{CHROMA_COLLECTION}': {collection.count()} chunks")
    except Exception:
        collection = None
        print(f"  No existing collection — upload a PDF to get started")

    print(f"\n{'=' * 60}")
    print("✓ READY")
    print(f"{'=' * 60}")
    print(f"  Server:    http://{SERVER_HOST}:{SERVER_PORT}")
    print(f"  LLM:       {OLLAMA_MODEL}")
    print(f"  Embedding: {EMBEDDING_MODEL_NAME}")
    print(f"  Knowledge: {collection.count() if collection else 0} chunks loaded")
    print()


# Run initialization
initialize()


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(title="Discussion Agent", version="0.1.0")

# Serve static files (index.html)
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)


# =============================================================================
# CONNECTION MANAGER
# =============================================================================

class ConnectionManager:
    """Manages WebSocket connections grouped by session_id."""

    def __init__(self):
        self.active_connections: dict[str, list[tuple[WebSocket, str]]] = {}

    async def connect(self, session_id: str, websocket: WebSocket, user_id: str):
        """Track a WebSocket connection (already accepted)."""
        if session_id not in self.active_connections:
            self.active_connections[session_id] = []
        self.active_connections[session_id].append((websocket, user_id))

    def disconnect(self, websocket: WebSocket, session_id: str):
        """Remove a WebSocket connection from the session."""
        if session_id in self.active_connections:
            self.active_connections[session_id] = [
                (ws, uid) for ws, uid in self.active_connections[session_id]
                if ws != websocket
            ]
            if not self.active_connections[session_id]:
                del self.active_connections[session_id]

    async def broadcast(self, session_id: str, message: dict):
        """Send a message to all connected users in a session."""
        if session_id not in self.active_connections:
            return
        disconnected = []
        for ws, uid in self.active_connections[session_id]:
            try:
                await ws.send_json(message)
            except Exception:
                disconnected.append(ws)
        for ws in disconnected:
            self.disconnect(ws, session_id)

    def get_users(self, session_id: str) -> list[str]:
        """Get list of connected user_ids in a session."""
        if session_id not in self.active_connections:
            return []
        return [uid for _, uid in self.active_connections[session_id]]


manager = ConnectionManager()


# =============================================================================
# HTTP ROUTES
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the chat frontend."""
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(
        content="<h1>Frontend not found</h1>"
                "<p>Place index.html in the static/ directory.</p>",
        status_code=404
    )


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF document for ingestion."""
    global collection, doc_info

    if not file.filename.lower().endswith(".pdf"):
        return JSONResponse(
            status_code=400,
            content={"error": "Only PDF files are supported"}
        )

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        collection, doc_info = upload_and_ingest(
            file_path, embedding_model, chroma_client
        )
        return JSONResponse(content={
            "status": "success",
            "document": doc_info
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Ingestion failed: {str(e)}"}
        )


@app.get("/sessions")
async def list_sessions():
    """List all active sessions with message counts."""
    return JSONResponse(content={
        "sessions": session_manager.list_sessions()
    })


@app.post("/sessions")
async def create_session():
    """Create a new chat session."""
    session_id = session_manager.create_session()
    return JSONResponse(content={"session_id": session_id})


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and disconnect all users."""
    if session_id in manager.active_connections:
        for ws, uid in manager.active_connections[session_id]:
            try:
                await ws.close()
            except Exception:
                pass
        del manager.active_connections[session_id]

    session_manager.delete_session(session_id)
    return JSONResponse(content={"status": "deleted", "session_id": session_id})


@app.delete("/collection")
async def reset_collection():
    """Reset the knowledge base."""
    global collection, doc_info
    try:
        chroma_client.delete_collection(name=CHROMA_COLLECTION)
        collection = None
        doc_info = None
        return JSONResponse(content={"status": "collection_reset"})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Reset failed: {str(e)}"}
        )


@app.get("/status")
async def get_status():
    """Health check — returns server and knowledge base status."""
    ollama = check_connection()
    return JSONResponse(content={
        "ollama": ollama,
        "document": doc_info,
        "chunks": collection.count() if collection else 0,
        "sessions": len(session_manager.sessions),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "llm_model": OLLAMA_MODEL
    })


# =============================================================================
# WEBSOCKET HANDLER
# =============================================================================

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    Real-time multi-user chat with agent monitoring.

    Client sends:
        {"type": "join",    "user_id": "alice"}
        {"type": "message", "user_id": "alice", "content": "Hello"}

    Server broadcasts:
        {"type": "user_message",  "user_id": "...", "content": "..."}
        {"type": "bot_thinking",  "status": "processing"}
        {"type": "bot_response",  "content": "...", "action": "...", ...}
        {"type": "bot_summary",   "content": "..."}
        {"type": "user_joined",   "user_id": "...", "users": [...]}
        {"type": "user_left",     "user_id": "...", "users": [...]}
    """
    user_id = None

    # Accept connection first — explicit accept for compatibility
    try:
        await websocket.accept()
    except Exception as e:
        print(f"  [WS] Accept failed: {e}")
        return

    try:
        # Ensure session exists
        if session_id not in session_manager.sessions:
            session_manager.sessions[session_id] = []

        # Wait for join message
        join_data = await websocket.receive_json()
        if join_data.get("type") != "join" or not join_data.get("user_id"):
            await websocket.close(code=4000, reason="First message must be a join")
            return

        user_id = join_data["user_id"]
        await manager.connect(session_id, websocket, user_id)

        # Broadcast join notification
        await manager.broadcast(session_id, {
            "type": "user_joined",
            "user_id": user_id,
            "users": manager.get_users(session_id)
        })

        print(f"  [WS] {user_id} joined session {session_id}")

        # Message loop
        while True:
            data = await websocket.receive_json()

            if data.get("type") != "message":
                continue

            content = data.get("content", "").strip()
            if not content:
                continue

            # Broadcast user message to all participants
            await manager.broadcast(session_id, {
                "type": "user_message",
                "user_id": user_id,
                "content": content
            })

            # Show thinking indicator
            await manager.broadcast(session_id, {
                "type": "bot_thinking",
                "status": "processing"
            })

            # Process through agent pipeline in thread pool
            result = await asyncio.to_thread(
                process_message,
                session_id,
                user_id,
                content,
                session_manager,
                throttler,
                collection,
                embedding_model
            )

            # Clear thinking indicator
            await manager.broadcast(session_id, {
                "type": "bot_thinking",
                "status": "idle"
            })

            # Broadcast bot response if any
            if result["bot_responded"]:
                if result["action"] == "summary":
                    await manager.broadcast(session_id, {
                        "type": "bot_summary",
                        "content": result["bot_response"]
                    })
                else:
                    trigger_type = ""
                    if result.get("trigger"):
                        trigger_type = result["trigger"]["trigger_type"]
                    elif result.get("from_queue"):
                        trigger_type = "QUEUED"

                    await manager.broadcast(session_id, {
                        "type": "bot_response",
                        "content": result["bot_response"],
                        "action": result["action"],
                        "trigger_type": trigger_type,
                        "from_queue": result["from_queue"]
                    })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"  [WS] Error for {user_id}: {e}")
    finally:
        if user_id:
            manager.disconnect(websocket, session_id)
            await manager.broadcast(session_id, {
                "type": "user_left",
                "user_id": user_id,
                "users": manager.get_users(session_id)
            })
            print(f"  [WS] {user_id} left session {session_id}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=SERVER_HOST,
        port=SERVER_PORT,
        reload=False,
        log_level="info"
    )