"""
embedding.py

- Embeddings: generated LOCALLY using sentence-transformers (no API key needed).
- LLM responses: generated via OpenRouter (or any OpenAI-compatible API).

This split is necessary because OpenRouter does not support the /v1/embeddings endpoint.
"""

from openai import OpenAI
from typing import List, Optional

from backend.config import settings
from backend.utils import logger

# ── Local embedding model (lazy-loaded) ──────────────────────────────────────
_embedding_model = None
_embedding_model_name = "all-MiniLM-L6-v2"   # 80MB, fast, good quality
_embedding_dim = 384                           # dimension for all-MiniLM-L6-v2


def _load_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading local embedding model: {_embedding_model_name}")
        try:
            from sentence_transformers import SentenceTransformer
            _embedding_model = SentenceTransformer(_embedding_model_name)
            logger.info("Local embedding model loaded successfully.")
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            )
    return _embedding_model


class EmbeddingService:
    def __init__(self):
        # LLM client — OpenRouter (or any OpenAI-compatible API)
        self.llm_client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_api_base
        )
        self.llm_model = settings.llm_model

    def get_embedding(self, text: str) -> List[float]:
        """Generate a vector embedding locally using sentence-transformers."""
        try:
            model = _load_embedding_model()
            text = text.replace("\n", " ")
            embedding = model.encode(text, normalize_embeddings=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating local embedding: {e}")
            raise e

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts locally."""
        try:
            model = _load_embedding_model()
            clean_texts = [t.replace("\n", " ") for t in texts]
            embeddings = model.encode(clean_texts, normalize_embeddings=True, show_progress_bar=False)
            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            raise e

    def generate_response(self, system_prompt: str, user_query: str, history: list = None) -> str:
        """Generate a text response using the LLM via OpenRouter."""
        try:
            messages = [{"role": "system", "content": system_prompt}]
            if history:
                messages.extend(history)
            messages.append({"role": "user", "content": user_query})

            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=0.2,
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            raise e

    def rewrite_query(self, query: str) -> str:
        """Rewrite a user query for better vector retrieval. Falls back to original on error."""
        try:
            system = (
                "You are a search query optimizer. "
                "Rewrite the user's question into a clear, detailed search query "
                "that will help retrieve the most relevant documents. "
                "Output ONLY the rewritten query, no explanations."
            )
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": query}
                ],
                temperature=0.0,
                max_tokens=120,
                stream=False
            )
            rewritten = response.choices[0].message.content.strip()
            logger.info(f"Query rewrite: '{query}' → '{rewritten}'")
            return rewritten if rewritten else query
        except Exception as e:
            logger.warning(f"Query rewrite failed, using original: {e}")
            return query


embedding_service = EmbeddingService()
