from backend.services.embeddings import EmbeddingService
from backend.services.vector_store import VectorStore


embedding_service = EmbeddingService()
vector_store = VectorStore(embedding_dim=384)