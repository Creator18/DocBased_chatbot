# =============================================================================
# config.py — All Constants and Configuration
# =============================================================================
# Single source of truth for every tunable parameter in the application.
# Every other module imports from here.
#
# Extracted from: Notebook 2, Cell 1

import os

# =============================================================================
# OLLAMA
# =============================================================================
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "mistral"

# =============================================================================
# EMBEDDING MODEL
# =============================================================================
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSIONS = 384
BGE_QUERY_PREFIX = "Represent this sentence: "

# =============================================================================
# CHUNKING
# =============================================================================
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
CHUNK_MIN_WORDS = 20
CHUNK_MIN_LETTER_RATIO = 0.40

# =============================================================================
# RETRIEVAL
# =============================================================================
TOP_K = 5
CHROMA_COLLECTION = "rag_documents"
CHROMA_PERSIST_DIR = "./chroma_data"

# =============================================================================
# DOCUMENT CLASSIFICATION
# =============================================================================
ACADEMIC_CONTENT_MARKERS = [
    "abstract", "introduction", "background", "chapter 1",
    "1. introduction", "1 introduction", "literature review"
]
ACADEMIC_SIGNALS = [
    "thesis", "dissertation", "submitted in partial fulfillment",
    "degree of", "bachelor", "master", "doctor of philosophy",
    "university", "department of", "supervisor", "declaration",
    "certificate", "acknowledgement", "acknowledgment"
]
ACADEMIC_SIGNAL_THRESHOLD = 2

# =============================================================================
# CONVERSATION HISTORY
# =============================================================================
MAX_HISTORY_TURNS = 5

# =============================================================================
# DISCUSSION AGENT — TRIGGER DETECTION
# =============================================================================
TRIGGER_CONTEXT_WINDOW = 3

BOT_NAMES = ["@bot", "hey bot", "bot,", "bot:"]

NON_TRIGGER_PHRASES = [
    "yeah", "yep", "yes", "sure", "ok", "okay", "right",
    "agreed", "exactly", "makes sense", "good point", "true",
    "lol", "haha", "hah", "lmao",
    "let's take a break", "let's move on", "brb", "back",
    "should we move on", "next topic", "let's continue",
    "thanks", "thank you", "cool", "nice", "great",
]

# =============================================================================
# DISCUSSION AGENT — RESPONSE THROTTLING
# =============================================================================
MIN_RESPONSE_GAP = 2
MAX_RESPONSE_GAP = 8
MAX_BOT_RATIO = 0.3
RATIO_WINDOW = 10
GAP_HISTORY_SIZE = 5

RAPID_TEMPO_THRESHOLD = 2.0   # seconds — increase to 15-20 for production
TEMPO_WINDOW = 3

SUMMARY_TRIGGER_PHRASES = [
    "let's take a break", "let's wrap up", "that's it for now",
    "we're done", "let's stop here", "good discussion",
    "let's pause", "break time", "that's all", "let's end here",
    "conversation over", "meeting over", "we can stop",
    "summarize", "give us a summary", "sum it up",
    "@bot summarize", "@bot summary"
]

SILENCE_SUMMARY_THRESHOLD = 300.0  # 5 minutes

# =============================================================================
# DISCUSSION AGENT — QUERY REFORMULATION
# =============================================================================
REFORMULATION_CONTEXT_TURNS = 3
MIN_SPECIFIC_WORDS = 8

VAGUE_PHRASE_SIGNALS = [
    "what about", "how about", "and the", "the part about",
    "the thing about", "the bit about"
]

VAGUE_WORD_SIGNALS = ["it", "that", "this", "they", "them", "those"]

# =============================================================================
# AGENT IDENTITY
# =============================================================================
AGENT_USER_ID = "agent"

# =============================================================================
# FILE PATHS
# =============================================================================
UPLOAD_DIR = "./uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

# =============================================================================
# SERVER
# =============================================================================
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000