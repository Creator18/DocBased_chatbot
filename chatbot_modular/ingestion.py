# =============================================================================
# ingestion.py — Document Loading, Chunking, Classification, Scoring, Storage
# =============================================================================
# Full ingestion pipeline: PDF → text → chunks → classify → embed → score → ChromaDB
# Single entry point: upload_and_ingest(file_path, embedding_model, chroma_client)
#
# Extracted from: Notebook 2, Cell 2
# Original source: Notebook 1, Cells 3 (load), 4 (chunk), 6 (classify/score), 7 (ingest)

import re
import uuid
from pathlib import Path

import numpy as np
from numpy import dot
from numpy.linalg import norm
from PyPDF2 import PdfReader
import chromadb

from config import (
    CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_MIN_WORDS, CHUNK_MIN_LETTER_RATIO,
    ACADEMIC_CONTENT_MARKERS, ACADEMIC_SIGNALS, ACADEMIC_SIGNAL_THRESHOLD,
    CHROMA_COLLECTION
)


# =============================================================================
# HELPER
# =============================================================================

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return dot(a, b) / (norm(a) * norm(b))


# =============================================================================
# DOCUMENT LOADING
# =============================================================================

def extract_text_from_pdf(file_path: str) -> str:
    """Extract all text from a PDF file, page by page."""
    reader = PdfReader(file_path)
    pages_text = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages_text.append(text)
        else:
            print(f"  ⚠ Page {i+1}: no text extracted (image or blank)")
    return "\n\n".join(pages_text)


def load_document(file_path: str) -> dict:
    """
    Load a PDF and return extracted text with metadata.
    Returns: {"doc_id": str, "filename": str, "text": str}
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Unsupported format: {path.suffix}. Only .pdf is supported.")

    print(f"  Loading PDF: {path.name}")
    text = extract_text_from_pdf(file_path)
    doc_id = uuid.uuid4().hex[:12]

    return {"doc_id": doc_id, "filename": path.name, "text": text}


# =============================================================================
# TEXT CHUNKING WITH QUALITY FILTER
# =============================================================================

def split_oversized_sentence(sentence: str, chunk_size: int) -> list[str]:
    """Split a sentence exceeding chunk_size on clause boundaries."""
    if len(sentence) <= chunk_size:
        return [sentence]

    split_points = []
    for pattern in ["; ", ", ", " - ", " — "]:
        for match in re.finditer(re.escape(pattern), sentence):
            split_points.append(match.start() + len(pattern))

    if split_points:
        mid = len(sentence) // 2
        best = min(split_points, key=lambda x: abs(x - mid))
        left = sentence[:best].strip()
        right = sentence[best:].strip()
    else:
        mid = len(sentence) // 2
        space_pos = sentence.rfind(" ", 0, mid + 50)
        if space_pos == -1:
            space_pos = mid
        left = sentence[:space_pos].strip()
        right = sentence[space_pos:].strip()

    return split_oversized_sentence(left, chunk_size) + \
           split_oversized_sentence(right, chunk_size)


def is_quality_chunk(chunk: str) -> bool:
    """Filter out low-content chunks (title pages, signature lines, etc.)."""
    words = [w for w in chunk.split() if len(w) > 1 and any(c.isalpha() for c in w)]
    if len(words) < CHUNK_MIN_WORDS:
        return False
    if len(chunk) == 0:
        return False
    letters = sum(1 for c in chunk if c.isalpha())
    if letters / len(chunk) < CHUNK_MIN_LETTER_RATIO:
        return False
    return True


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    apply_quality_filter: bool = True
) -> list[str]:
    """
    Split text into overlapping, sentence-aware chunks.
    Quality filter removes low-content chunks automatically.
    """
    sentences = re.split(r'(?<=[.!?])\s+|\n\n+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    processed = []
    for s in sentences:
        processed.extend(split_oversized_sentence(s, chunk_size))
    sentences = processed

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)
        if current_length + sentence_length + 1 > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))

            overlap_sentences = []
            overlap_length = 0
            for s in reversed(current_chunk):
                if overlap_length + len(s) > overlap:
                    break
                overlap_sentences.insert(0, s)
                overlap_length += len(s) + 1

            current_chunk = overlap_sentences
            current_length = overlap_length

        current_chunk.append(sentence)
        current_length += sentence_length + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    if apply_quality_filter:
        before = len(chunks)
        chunks = [c for c in chunks if is_quality_chunk(c)]
        removed = before - len(chunks)
        if removed > 0:
            print(f"  Quality filter removed {removed} low-content chunk(s)")

    return chunks


# =============================================================================
# DOCUMENT CLASSIFIER AND BOILERPLATE SCORING
# =============================================================================

def classify_document(text: str) -> str:
    """Classify document as 'academic' or 'general' from first ~2000 chars."""
    sample = text[:2000].lower()
    matches = sum(1 for signal in ACADEMIC_SIGNALS if signal in sample)
    doc_type = "academic" if matches >= ACADEMIC_SIGNAL_THRESHOLD else "general"
    print(f"  Classification: {doc_type} ({matches} academic signal(s))")
    return doc_type


def _keyword_score_academic(chunks: list[str]) -> list[float]:
    """Keyword-based boilerplate scores for academic documents."""
    scores = [0.0] * len(chunks)

    content_start = 0
    for i, chunk in enumerate(chunks):
        chunk_lower = chunk.lower()
        if any(marker in chunk_lower for marker in ACADEMIC_CONTENT_MARKERS):
            content_start = i
            break
    for i in range(content_start):
        scores[i] = 1.0

    reference_markers = ["references", "bibliography", "works cited"]
    content_end = len(chunks)
    scan_start = max(0, int(len(chunks) * 0.80))
    for i in range(len(chunks) - 1, scan_start - 1, -1):
        chunk_lower = chunks[i].lower()
        for marker in reference_markers:
            if marker in chunk_lower:
                content_end = i
                break
    for i in range(content_end, len(chunks)):
        scores[i] = 0.8

    return scores


def _keyword_score_general(chunks: list[str]) -> list[float]:
    """Keyword-based boilerplate scores for general documents."""
    scores = [0.0] * len(chunks)
    if chunks:
        first = chunks[0]
        letters = sum(1 for c in first if c.isalpha())
        ratio = letters / len(first) if len(first) > 0 else 0
        words = len(first.split())
        if words < 30 or ratio < 0.5:
            scores[0] = 0.7
    return scores


def _edge_repetition_scores(chunk_embeddings: np.ndarray, edge_count: int = 5) -> list[float]:
    """Score edge chunks by mutual similarity (formulaic language detection)."""
    n = len(chunk_embeddings)
    scores = [0.0] * n
    if n < edge_count * 2 + 4:
        return scores

    def pairwise_mean(embeds):
        sims = []
        for i in range(len(embeds)):
            for j in range(i + 1, len(embeds)):
                sims.append(cosine_similarity(embeds[i], embeds[j]))
        return np.mean(sims) if sims else 0.0

    def remap(val):
        return max(0.0, min(1.0, (val - 0.3) / 0.4))

    front_score = remap(pairwise_mean(chunk_embeddings[:edge_count]))
    back_score = remap(pairwise_mean(chunk_embeddings[-edge_count:]))

    for i in range(edge_count):
        scores[i] = front_score
    for i in range(n - edge_count, n):
        scores[i] = back_score

    return scores


def _divergence_scores(chunk_embeddings: np.ndarray, edge_count: int = 5) -> list[float]:
    """Score edge chunks by distance from the interior centroid."""
    n = len(chunk_embeddings)
    scores = [0.0] * n
    if n < edge_count * 2 + 4:
        return scores

    interior_start = int(n * 0.20)
    interior_end = int(n * 0.80)
    interior_centroid = np.mean(chunk_embeddings[interior_start:interior_end], axis=0)

    for i in range(edge_count):
        sim = cosine_similarity(chunk_embeddings[i], interior_centroid)
        scores[i] = max(0.0, min(1.0, (0.6 - sim) / 0.4))
    for i in range(n - edge_count, n):
        sim = cosine_similarity(chunk_embeddings[i], interior_centroid)
        scores[i] = max(0.0, min(1.0, (0.6 - sim) / 0.4))

    return scores


def score_boilerplate(
    chunks: list[str],
    doc_type: str,
    chunk_embeddings: np.ndarray,
    keyword_weight: float = 0.5,
    repetition_weight: float = 0.2,
    divergence_weight: float = 0.3,
) -> list[float]:
    """
    Combined boilerplate score per chunk (0.0-1.0).
    Three signals: keyword/positional, edge repetition, edge divergence.
    """
    if doc_type == "academic":
        kw_scores = _keyword_score_academic(chunks)
    else:
        kw_scores = _keyword_score_general(chunks)

    rep_scores = _edge_repetition_scores(chunk_embeddings)
    div_scores = _divergence_scores(chunk_embeddings)

    combined = []
    for i in range(len(chunks)):
        score = (
            keyword_weight * kw_scores[i] +
            repetition_weight * rep_scores[i] +
            divergence_weight * div_scores[i]
        )
        combined.append(max(0.0, min(1.0, score)))

    return combined


# =============================================================================
# CHROMADB INGESTION
# =============================================================================

def ingest_document(
    chunks: list[str],
    chunk_embeddings: np.ndarray,
    boilerplate_scores: list[float],
    doc_id: str,
    doc_type: str,
    filename: str,
    chroma_client: chromadb.ClientAPI,
    collection_name: str = CHROMA_COLLECTION
) -> chromadb.Collection:
    """
    Store chunks + embeddings + metadata in ChromaDB.
    Clears existing collection on re-run (safe for iteration).
    """
    existing = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    if existing.count() > 0:
        print(f"  ⚠ Collection '{collection_name}' has {existing.count()} entries — clearing")
        chroma_client.delete_collection(name=collection_name)
        existing = chroma_client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "source": filename,
            "doc_id": doc_id,
            "doc_type": doc_type,
            "chunk_index": i,
            "char_count": len(chunk),
            "boilerplate_score": round(boilerplate_scores[i], 4)
        }
        for i, chunk in enumerate(chunks)
    ]

    BATCH_SIZE = 100
    for start in range(0, len(chunks), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(chunks))
        existing.add(
            ids=ids[start:end],
            embeddings=chunk_embeddings[start:end].tolist(),
            documents=chunks[start:end],
            metadatas=metadatas[start:end]
        )
        print(f"  Stored batch {start}-{end}")

    return existing


# =============================================================================
# TOP-LEVEL ENTRY POINT
# =============================================================================

def upload_and_ingest(
    file_path: str,
    embedding_model,
    chroma_client: chromadb.ClientAPI
) -> tuple[chromadb.Collection, dict]:
    """
    Full ingestion pipeline: load → chunk → classify → embed → score → store.

    Args:
        file_path:       Path to a PDF file
        embedding_model: Loaded SentenceTransformer instance
        chroma_client:   ChromaDB PersistentClient instance

    Returns:
        (collection, doc_info) where doc_info contains:
            doc_id, filename, doc_type, num_chunks, num_pages
    """
    print(f"{'=' * 60}")
    print(f"INGESTION PIPELINE")
    print(f"{'=' * 60}")

    # Step 1: Load PDF
    doc = load_document(file_path)
    num_pages = len(PdfReader(file_path).pages)
    print(f"  Extracted {len(doc['text']):,} chars from {num_pages} pages\n")

    # Step 2: Chunk
    print("  Chunking...")
    chunks = chunk_text(doc["text"])
    print(f"  → {len(chunks)} chunks "
          f"(range: {min(len(c) for c in chunks)}-{max(len(c) for c in chunks)} chars)\n")

    # Step 3: Classify
    doc_type = classify_document(doc["text"])

    # Step 4: Embed
    print(f"\n  Embedding {len(chunks)} chunks...")
    chunk_embs = embedding_model.encode(chunks, show_progress_bar=True)
    print(f"  → Embeddings shape: {chunk_embs.shape}\n")

    # Step 5: Boilerplate scoring
    print("  Scoring boilerplate...")
    bp_scores = score_boilerplate(chunks, doc_type, chunk_embs)
    high_bp = sum(1 for s in bp_scores if s > 0.3)
    print(f"  → {high_bp} chunk(s) flagged (score > 0.3)\n")

    # Step 6: Ingest into ChromaDB
    print("  Ingesting into ChromaDB...")
    coll = ingest_document(
        chunks=chunks,
        chunk_embeddings=chunk_embs,
        boilerplate_scores=bp_scores,
        doc_id=doc["doc_id"],
        doc_type=doc_type,
        filename=doc["filename"],
        chroma_client=chroma_client
    )

    doc_info = {
        "doc_id": doc["doc_id"],
        "filename": doc["filename"],
        "doc_type": doc_type,
        "num_chunks": len(chunks),
        "num_pages": num_pages
    }

    print(f"\n{'=' * 60}")
    print(f"✓ INGESTION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Doc ID:     {doc_info['doc_id']}")
    print(f"  File:       {doc_info['filename']}")
    print(f"  Type:       {doc_info['doc_type']}")
    print(f"  Pages:      {doc_info['num_pages']}")
    print(f"  Chunks:     {doc_info['num_chunks']}")
    print(f"  Collection: {CHROMA_COLLECTION} ({coll.count()} entries)")

    return coll, doc_info