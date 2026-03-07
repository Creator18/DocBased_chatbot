# 🤖 Discussion Agent

A **passive-observer RAG chatbot** that monitors multi-user conversations and selectively interjects with document-backed insights — only when it would actually help.

Unlike standard RAG Q&A bots that respond to every message, this agent **listens silently by default** and makes an intelligent decision about when to speak. It reads the room, waits for natural pauses, and contributes like a knowledgeable colleague — not a search engine.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-WebSocket-green)
![Ollama](https://img.shields.io/badge/LLM-Mistral%207B-orange)
![ChromaDB](https://img.shields.io/badge/Vector%20Store-ChromaDB-purple)

---

## What Makes This Different

Most RAG systems follow a simple pattern: user asks → bot answers → repeat. This project inverts that. The bot's **default state is silence**. It monitors a live group discussion and only speaks when:

- Someone directly asks it (`@bot what does the paper say about...`)
- A factual question comes up that the uploaded document can answer
- Someone expresses uncertainty the document can resolve (*"I think they used Docker but I'm not sure"*)
- Two users disagree on something the document addresses
- Someone asks for clarification on a topic the document covers

It stays quiet during opinions, casual chat, agreements, and meta-discussion.

---

## Key Features

### Intelligent Trigger Detection
Two-stage pipeline: rule-based fast path catches obvious cases instantly (direct @bot mentions, short reactions like "ok" or "yeah"), while ambiguous messages go to an LLM classifier that understands intent in context. Most messages are resolved without an LLM call.

### Adaptive Response Throttling
The bot learns its own rhythm. It tracks how many user messages pass between its responses and adjusts its minimum gap dynamically. If it's been spacing responses 6-7 messages apart, it won't suddenly jump in after 2. Includes tempo detection — if users are in rapid back-and-forth, triggers are queued and fired at the next natural pause.

### Discussion-Aware Query Reformulation
Vague messages like *"What about the accuracy?"* are expanded using conversation context before retrieval. If the last few messages discussed emotion detection models, the retrieval query becomes *"emotion detection model accuracy on facial expression datasets"* — pulling the right chunks instead of garbage.

### Dual-Context Generation
When the bot responds, it sees both the uploaded document (RAG retrieval) and the recent conversation history. This lets it respond naturally to what's being discussed rather than giving decontextualized answers.

### Soft Boilerplate Scoring
Three-signal weighted system that scores (not deletes) likely boilerplate in uploaded documents. Scores are stored in metadata and used at retrieval time to penalize noise without losing potentially useful content.

### Conversation Summaries
Say *"let's take a break"* or *"summarize"* and the bot generates a recap of the discussion — main topics, questions raised, points of disagreement, and unresolved items.

### Real-Time Multi-User Chat
WebSocket-based chat with multiple participants. Visual distinction between user messages, bot responses (tagged by trigger type), and summaries. Live status indicator showing when the bot is monitoring vs. thinking.

---

## Architecture

```
User Message
    │
    ▼
┌─────────────────────┐
│  Summary Detection   │──── "let's take a break" ──► Generate Summary
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Queued Triggers     │──── tempo calmed down? ──► Process queued trigger
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Trigger Detection   │──── rule-based (instant)
│  (should bot speak?) │──── LLM classifier (3-10s, only if ambiguous)
└─────────┬───────────┘
          │ YES
          ▼
┌─────────────────────┐
│  Adaptive Throttle   │──── gap check (adaptive minimum)
│  (should bot wait?)  │──── ratio check (bot not dominating?)
│                      │──── tempo check (conversation paused?)
└─────────┬───────────┘
          │ RESPOND
          ▼
┌─────────────────────┐
│  Query Reformulation │──── expand vague queries with conversation context
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  RAG Retrieval       │──── BGE-prefixed query embedding
│                      │──── boilerplate-penalized scoring
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Dual-Context LLM    │──── document chunks + conversation history
│  Generation          │──── collaborative tone, grounded response
└─────────────────────┘
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Ollama (Mistral 7B, local) |
| Embeddings | `BAAI/bge-small-en-v1.5` via sentence-transformers (384d) |
| Vector Store | ChromaDB (persistent, cosine similarity) |
| Backend | FastAPI + Uvicorn (WebSocket) |
| Frontend | Vanilla HTML/CSS/JS (no build step) |
| Environment | Python 3.13, runs entirely local — no paid API calls |

---

## Project Structure

```
discussion-agent/
├── app/
│   ├── main.py            ← FastAPI app, routes, WebSocket handler
│   ├── config.py          ← All tunable constants
│   ├── llm.py             ← Ollama communication wrapper
│   ├── ingestion.py       ← PDF → chunks → classify → score → ChromaDB
│   ├── retrieval.py       ← Boilerplate-penalized semantic search
│   ├── session.py         ← Multi-user session management
│   ├── agent.py           ← Trigger detection, reformulation, throttling
│   ├── discussion.py      ← Dual-context RAG pipeline + orchestrator
│   └── static/
│       └── index.html     ← Chat frontend
├── notebooks/
│   ├── 01_rag_pipeline.ipynb      ← Core RAG (11 cells)
│   └── 02_discussion_agent.ipynb  ← Agent + testing (10 cells)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

### Prerequisites
- Python 3.13+
- [Ollama](https://ollama.ai/) installed and running
- Mistral 7B pulled: `ollama pull mistral`

### Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/discussion-agent.git
cd discussion-agent

# Install dependencies
pip install -r requirements.txt

# Make sure Ollama is running
ollama serve

# Start the app
cd app
python main.py
```

Open `http://localhost:8000` in your browser.

### Quick Test

1. Enter a username and click **Join Discussion**
2. Upload a PDF using the button in the top bar
3. Start chatting — the bot monitors silently
4. Ask a question about the document — the bot interjects
5. Type `@bot` to directly address the agent
6. Say *"let's take a break"* to get a summary

For multi-user testing, open a second browser tab at `localhost:8000`, enter a different username, and paste the same session ID from the first tab.

---

## How It Works

### The Notebooks Tell the Full Story

**Notebook 1** builds and validates the core RAG pipeline from scratch — document loading, sentence-aware chunking, embedding, three-signal boilerplate scoring, boilerplate-penalized retrieval, grounded generation, and conversation history. Each cell builds on the previous one with integration tests throughout.

**Notebook 2** builds the discussion agent on top of that foundation — multi-user sessions, trigger detection, query reformulation, dual-context generation, adaptive throttling, and a full conversation simulation that validates all components end-to-end. Then wraps everything in a FastAPI server with a live chat frontend.

### Design Decisions

**Why LLM-based trigger detection instead of keyword matching?**
Keywords can't distinguish *"I think they used Docker but I'm not sure"* (uncertainty the doc can resolve) from *"I think the CNN approach is better"* (opinion, bot should stay quiet). The LLM understands intent. The rule-based fast path handles the obvious cases without an LLM call to keep latency down.

**Why adaptive throttling instead of a fixed cooldown?**
Fixed cooldowns feel robotic. The adaptive system learns from the actual conversation pattern — if the bot naturally spaces responses 5-6 messages apart, it maintains that rhythm. The tempo detection prevents interrupting rapid exchanges. The result feels more like a colleague who knows when to speak up and when to hold back.

**Why soft boilerplate scoring instead of hard deletion?**
Hard deletion is irreversible and overconfident. A chunk that looks like boilerplate (title page, acknowledgements) might still contain useful metadata. Scoring preserves everything and lets the retrieval stage make a nuanced decision — penalize likely boilerplate, don't nuke it.

**Why reformulate queries before retrieval?**
In a group discussion, most messages are contextual — *"What about the accuracy?"* means nothing without knowing what was discussed before. Reformulation bridges conversational context into retrieval queries, closing the gap between vague queries (~54% relevance) and specific ones (~82% relevance).

---

## Configuration

All tunable parameters are in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `OLLAMA_MODEL` | `mistral` | LLM model name |
| `CHUNK_SIZE` | `800` | Max characters per chunk |
| `TOP_K` | `5` | Chunks retrieved per query |
| `MIN_RESPONSE_GAP` | `2` | Minimum messages between bot responses |
| `MAX_RESPONSE_GAP` | `8` | Maximum adaptive gap ceiling |
| `MAX_BOT_RATIO` | `0.3` | Max bot message ratio before backing off |
| `RAPID_TEMPO_THRESHOLD` | `2.0` | Seconds — below this, conversation is "rapid" |
| `TRIGGER_CONTEXT_WINDOW` | `3` | Recent messages shown to trigger classifier |

For production with real users, increase `RAPID_TEMPO_THRESHOLD` to 15-20 seconds.

---

## Known Limitations

- **Sessions are in-memory** — lost on server restart. Swap to Redis or SQLite for persistence.
- **Single document** — multi-document ingestion works at the pipeline level but cross-document retrieval hasn't been validated.
- **Mistral 7B cold start** takes ~30 seconds. Subsequent calls are 3-10 seconds.
- **No authentication** — anyone with the URL can join any session. Add auth for production.
- **Text-only** — no audio/voice input (the infrastructure from Notebook 1 supports it but it's disabled).

---

## Future Improvements

- Redis/SQLite session persistence
- Docker + docker-compose for one-command deployment
- Cross-encoder re-ranking for retrieval precision
- Hybrid search (semantic + BM25 keyword)
- Multi-document support with cross-document retrieval
- Authentication and session access control
- Desktop app wrapper (Electron/Tauri)

---

## License

MIT