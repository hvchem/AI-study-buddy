import faiss
import numpy as np
from typing import List, Dict

class VectorStore:
    def __init__(self, embedding_dim: int):
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.text_chunks: List[str] = []
        self.metadata: List[Dict] = []

    def add(self, embeddings: np.ndarray, chunks: List[str], source: str = "unknown"):
        self.index.add(embeddings)
        for chunk in chunks:
            self.metadata.append({"source": source})
        self.text_chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 3):
        if self.index.ntotal == 0:
            return []
        distances, indices = self.index.search(query_embedding, top_k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.text_chunks):
                results.append({
                    "text": self.text_chunks[idx],
                    "source": self.metadata[idx]["source"],
                    "score": float(distances[0][i])
                })
        return results