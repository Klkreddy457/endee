"""
reranker.py – Cross-encoder reranking for retrieved document chunks.

Uses sentence-transformers' CrossEncoder model to score (query, passage) pairs
and rerank the top-k candidates retrieved from Endee.

The model is lazy-loaded on first use so startup time is not impacted when
reranking is disabled.
"""

from typing import List, Dict, Any
from backend.utils import logger

# Lightweight model that downloads once (~67 MB) and runs entirely locally.
# Replace with 'BAAI/bge-reranker-v2-m3' for better multilingual accuracy.
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_cross_encoder = None   # lazy singleton


def _load_model(model_name: str = DEFAULT_RERANKER_MODEL):
    global _cross_encoder
    if _cross_encoder is None:
        logger.info(f"Loading cross-encoder reranker: {model_name}")
        try:
            from sentence_transformers import CrossEncoder
            _cross_encoder = CrossEncoder(model_name, max_length=512)
            logger.info("Reranker loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load reranker model: {e}")
            raise


def rerank(query: str, chunks: List[Dict[str, Any]], top_n: int = 3) -> List[Dict[str, Any]]:
    """
    Score (query, chunk_text) pairs with a cross-encoder and return the
    top_n chunks ordered by reranker score (descending).

    Args:
        query:   The user's question.
        chunks:  List of dicts with at minimum a 'text' key.
        top_n:   How many reranked chunks to keep.

    Returns:
        Top-n chunks sorted by reranker score, with an added 'reranker_score' key.
    """
    if not chunks:
        return chunks

    _load_model()

    pairs = [(query, c["text"]) for c in chunks]
    scores = _cross_encoder.predict(pairs)     # numpy array of floats

    # Attach scores and sort
    for chunk, score in zip(chunks, scores):
        chunk["reranker_score"] = float(score)

    ranked = sorted(chunks, key=lambda c: c["reranker_score"], reverse=True)
    top = ranked[:top_n]

    logger.info(
        f"Reranking: {len(chunks)} → {len(top)} chunks. "
        f"Top score: {top[0]['reranker_score']:.4f}"
    )
    return top
