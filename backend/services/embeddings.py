from sentence_transformers import SentenceTransformer
from typing import List

class EmbeddingService:
    """
    Handles text-to-vector embedding using Sentence Transformers.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Load embedding model once
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]):

        """
        Convert a list of text chunks into embeddings.
        """

        return self.model.encode(texts, convert_to_numpy=True)