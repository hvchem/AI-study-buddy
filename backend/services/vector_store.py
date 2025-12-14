import faiss
import numpy as np
from typing import List

class VectorStore:
    """
    FAISS-based vector store for similarity search.
    """

    def __init__(self, embedding_dim: int):
        # L2 distance index (simple and effective)
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.text_chunks: List[str] = []

    def add(self, embeddings: np.ndarray, chunks: List[str]):
        """
        Add embeddings and corresponding text chunks to the index.
        """
        self.index.add(embeddings)
        self.text_chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 3):
        """
        Search for the most relevant chunks.
        """
        distances, indices = self.index.search(query_embedding, top_k)
        return [self.text_chunks[i] for i in indices[0]]
