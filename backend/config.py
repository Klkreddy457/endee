import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    endee_url: str = os.getenv("ENDEE_URL", "http://localhost:8080")
    endee_auth_token: str = os.getenv("ENDEE_AUTH_TOKEN", "")
    
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_api_base: str = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    
    # RAG Settings
    chunk_size: int = 1000
    chunk_overlap: int = 100
    top_k: int = 3
    
    # Endee Vector Settings
    vector_dimension: int = 1536 # Default for text-embedding-3-small
    index_name: str = "rag_docs"

    class Config:
        env_file = ".env"

settings = Settings()
