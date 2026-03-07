# 🤖 Discussion Agent

A passive-observer RAG chatbot that monitors multi-user conversations and selectively interjects with document-backed insights — only when it would actually help. The bot's default state is silence. It listens, waits for the right moment, and contributes like a knowledgeable colleague.

Runs entirely local. No paid API calls.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Ollama (Mistral 7B) |
| Embeddings | BAAI/bge-small-en-v1.5 (384d) |
| Vector Store | ChromaDB |
| Backend | FastAPI + WebSocket |
| Frontend | HTML/CSS/JS |
| Language | Python 3.13 |

---

## Architecture

```
User Message
    │
    ▼
  Summary Detection ──── "let's take a break" ──► Generate Summary
    │
    ▼
  Queued Triggers ──── tempo calmed? ──► Process queued trigger
    │
    ▼
  Trigger Detection ──── rule-based (instant) / LLM classifier (if ambiguous)
    │ YES
    ▼
  Adaptive Throttle ──── gap check / ratio check / tempo check
    │ RESPOND
    ▼
  Query Reformulation ──── expand vague queries with conversation context
    │
    ▼
  RAG Retrieval ──── boilerplate-penalized semantic search
    │
    ▼
  Dual-Context Generation ──── document chunks + conversation history
```

---

## Setup

**Prerequisites:** Python 3.13+, [Ollama](https://ollama.ai/) running with Mistral pulled (`ollama pull mistral`)

```bash
git clone https://github.com/yourusername/discussion-agent.git
cd discussion-agent
pip install -r requirements.txt
cd app
python main.py
```

Open `http://localhost:8000`. Enter a username, upload a PDF, and start chatting. Open a second tab with a different username and the same session ID for multi-user testing.

---

## Project Structure

```
app/
├── main.py          ← FastAPI app, routes, WebSocket handler
├── config.py        ← All tunable constants
├── llm.py           ← Ollama communication
├── ingestion.py     ← PDF → chunks → classify → score → ChromaDB
├── retrieval.py     ← Boilerplate-penalized semantic search
├── session.py       ← Multi-user session management
├── agent.py         ← Trigger detection, reformulation, throttling
├── discussion.py    ← Dual-context RAG pipeline + orchestrator
└── static/
    └── index.html   ← Chat frontend

notebooks/
├── 01_rag_pipeline.ipynb
└── 02_discussion_agent.ipynb
```