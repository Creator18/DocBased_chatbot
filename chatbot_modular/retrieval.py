# =============================================================================
# retrieval.py — Query Embedding, Boilerplate-Penalized Search, Context Formatting
# =============================================================================
# Given a query string, embeds it, retrieves candidate chunks from ChromaDB,
# penalizes boilerplate, and returns the best results with formatted context.
#
# Extracted from: Notebook 2, Cell 6 (retrieve + format_context)
# Original source: Notebook 1, Cell 8

from config import BGE_QUERY_PREFIX, TOP_K


def retrieve(
    query: str,
    collection,
    embedding_model,
    top_k: int = TOP_K,
    boilerplate_penalty: float = 0.5
) -> dict:
    """
    Retrieve the most relevant chunks for a query, with boilerplate demotion.

    Strategy:
        1. Embed the query with BGE prefix
        2. Pull back top_k * 2 candidates from ChromaDB
        3. Adjust scores: penalize chunks with high boilerplate_score
        4. Re-sort and return the best top_k

    Args:
        query:               The search query (raw or reformulated)
        collection:          ChromaDB collection to search
        embedding_model:     Loaded SentenceTransformer instance
        top_k:               Number of chunks to return
        boilerplate_penalty: How much to penalize boilerplate (0.0-1.0)

    Returns:
        dict with:
            - documents:       list of chunk texts
            - metadatas:       list of metadata dicts
            - distances:       list of original distances
            - adjusted_scores: list of final scores after penalty
            - ids:             list of chunk IDs
    """
    # Step 1: Embed query with BGE prefix
    prefixed_query = BGE_QUERY_PREFIX + query
    query_embedding = embedding_model.encode(prefixed_query).tolist()

    # Step 2: Retrieve extra candidates for re-ranking
    candidate_count = min(top_k * 2, collection.count())
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=candidate_count,
        include=["documents", "metadatas", "distances"]
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    ids = results["ids"][0]

    # Step 3: Compute adjusted scores
    adjusted_scores = []
    for dist, meta in zip(distances, metadatas):
        base_score = 1 - dist
        bp_score = meta.get("boilerplate_score", 0.0)
        adjusted = base_score * (1 - boilerplate_penalty * bp_score)
        adjusted_scores.append(adjusted)

    # Step 4: Sort by adjusted score and take top_k
    ranked = sorted(
        zip(documents, metadatas, distances, adjusted_scores, ids),
        key=lambda x: x[3],
        reverse=True
    )[:top_k]

    return {
        "documents":       [r[0] for r in ranked],
        "metadatas":       [r[1] for r in ranked],
        "distances":       [r[2] for r in ranked],
        "adjusted_scores": [r[3] for r in ranked],
        "ids":             [r[4] for r in ranked]
    }


def format_context(retrieved: dict) -> str:
    """
    Format retrieved chunks into a context string for the LLM prompt.
    Includes source attribution, relevance score, and boilerplate warning.

    Args:
        retrieved: Output dict from retrieve()

    Returns:
        Formatted context string ready to embed in a prompt.
    """
    context_parts = []
    for i, (doc_text, meta, adj_score) in enumerate(zip(
        retrieved["documents"],
        retrieved["metadatas"],
        retrieved["adjusted_scores"]
    )):
        bp = meta.get("boilerplate_score", 0.0)
        bp_tag = " [low-confidence source]" if bp > 0.3 else ""

        context_parts.append(
            f"[Source: {meta['source']}, Chunk {meta['chunk_index']}, "
            f"Relevance: {adj_score:.2%}{bp_tag}]\n{doc_text}"
        )

    return "\n\n---\n\n".join(context_parts)