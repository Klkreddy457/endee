from openai import OpenAI
from typing import List, Optional

from backend.config import settings
from backend.utils import logger

class EmbeddingService:
    def __init__(self):
        # Configure the client (works for both OpenAI and local Ollama compatible APIs)
        self.client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_api_base
        )
        self.embedding_model = settings.embedding_model
        self.llm_model = settings.llm_model

    def get_embedding(self, text: str) -> List[float]:
        """Generate vector embedding for a single text string."""
        try:
            text = text.replace("\n", " ")
            response = self.client.embeddings.create(
                input=[text],
                model=self.embedding_model
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise e

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate vector embeddings for a list of texts."""
        try:
            # Replace newlines
            clean_texts = [text.replace("\n", " ") for text in texts]
            response = self.client.embeddings.create(
                input=clean_texts,
                model=self.embedding_model
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            raise e

    def generate_response(self, system_prompt: str, user_query: str, history: list = None) -> str:
        """Generate a text response using the LLM model, with optional conversation history."""
        try:
            messages = [{"role": "system", "content": system_prompt}]
            # Inject previous turns for memory
            if history:
                messages.extend(history)
            messages.append({"role": "user", "content": user_query})

            response = self.client.chat.completions.create(
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
        """
        Rewrite / expand a user query into a more descriptive, self-contained
        version that is better suited for dense vector retrieval.
        Falls back to the original query on any error.
        """
        try:
            system = (
                "You are a search query optimizer. "
                "Rewrite the user's question into a clear, detailed search query "
                "that will help retrieve the most relevant documents. "
                "Output ONLY the rewritten query, no explanations."
            )
            response = self.client.chat.completions.create(
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
