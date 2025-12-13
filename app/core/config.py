from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings and configuration."""
    
    # App settings
    app_name: str = "AI Study Buddy"
    version: str = "1.0.0"
    debug: bool = False
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # File upload settings
    upload_dir: str = "uploads"
    max_upload_size: int = 10 * 1024 * 1024  # 10MB
    allowed_extensions: list = [".pdf"]
    
    # Model settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "google/flan-t5-base"
    
    # RAG settings
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k_results: int = 3
    
    # FAISS settings
    faiss_index_path: str = "faiss_index"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
