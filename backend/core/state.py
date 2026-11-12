from backend.services.embeddings import EmbeddingService
from backend.services.vector_store import VectorStore
from backend.services.llm import LLMService

embedding_service = EmbeddingService()
vector_store = VectorStore(embedding_dim=384)
llm_service = LLMService()