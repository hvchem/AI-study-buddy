import os
import pickle
from typing import List, Dict, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from app.core.config import settings


class EmbeddingService:
    """Service for generating and managing embeddings."""
    
    def __init__(self):
        self.model_name = settings.embedding_model
        self.model: Optional[SentenceTransformer] = None
        self.index_path = settings.faiss_index_path
        self.index: Optional[faiss.Index] = None
        self.document_chunks: Dict[str, List[str]] = {}
        self.chunk_to_doc: Dict[int, str] = {}
        self.doc_start_idx: Dict[str, int] = {}  # Cache for document start indices
        self._initialized = False
    
    def _initialize(self):
        """Initialize the service (lazy loading)."""
        if not self._initialized:
            self._load_or_create_index()
            self._initialized = True
    
    def _load_model(self):
        """Load the sentence transformer model."""
        if self.model is None:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
    
    def _load_or_create_index(self):
        """Load existing FAISS index or create a new one."""
        os.makedirs(self.index_path, exist_ok=True)
        index_file = os.path.join(self.index_path, "index.faiss")
        metadata_file = os.path.join(self.index_path, "metadata.pkl")
        
        if os.path.exists(index_file) and os.path.exists(metadata_file):
            try:
                self.index = faiss.read_index(index_file)
                with open(metadata_file, "rb") as f:
                    metadata = pickle.load(f)
                    self.document_chunks = metadata.get("document_chunks", {})
                    self.chunk_to_doc = metadata.get("chunk_to_doc", {})
                    self.doc_start_idx = metadata.get("doc_start_idx", {})
                print(f"Loaded existing FAISS index with {self.index.ntotal} vectors")
            except Exception as e:
                print(f"Error loading index: {e}. Creating new index.")
                self._create_new_index()
        else:
            self._create_new_index()
    
    def _create_new_index(self):
        """Create a new FAISS index."""
        self._load_model()
        dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(dimension)
        self.document_chunks = {}
        self.chunk_to_doc = {}
        self.doc_start_idx = {}
        print(f"Created new FAISS index with dimension {dimension}")
    
    def _save_index(self):
        """Save FAISS index and metadata to disk."""
        os.makedirs(self.index_path, exist_ok=True)
        index_file = os.path.join(self.index_path, "index.faiss")
        metadata_file = os.path.join(self.index_path, "metadata.pkl")
        
        faiss.write_index(self.index, index_file)
        with open(metadata_file, "wb") as f:
            pickle.dump({
                "document_chunks": self.document_chunks,
                "chunk_to_doc": self.chunk_to_doc,
                "doc_start_idx": self.doc_start_idx
            }, f)
        print(f"Saved FAISS index with {self.index.ntotal} vectors")
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Numpy array of embeddings
        """
        self._load_model()
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings
    
    def add_document(self, document_id: str, chunks: List[str]):
        """
        Add document chunks to the index.
        
        Args:
            document_id: Unique document identifier
            chunks: List of text chunks from the document
        """
        self._initialize()
        
        if not chunks:
            return
        
        # Generate embeddings
        embeddings = self.generate_embeddings(chunks)
        
        # Get current index size
        start_idx = self.index.ntotal
        
        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Update metadata
        self.document_chunks[document_id] = chunks
        self.doc_start_idx[document_id] = start_idx
        for i, chunk in enumerate(chunks):
            self.chunk_to_doc[start_idx + i] = document_id
        
        # Save index
        self._save_index()
        print(f"Added {len(chunks)} chunks for document {document_id}")
    
    def search(self, query: str, top_k: int = None, document_id: str = None) -> List[Tuple[str, float, str]]:
        """
        Search for similar chunks using a query.
        
        Args:
            query: Search query
            top_k: Number of results to return
            document_id: Optional document ID to filter results
            
        Returns:
            List of tuples (chunk_text, distance, doc_id)
        """
        self._initialize()
        
        if self.index.ntotal == 0:
            return []
        
        if top_k is None:
            top_k = settings.top_k_results
        
        # Generate query embedding
        query_embedding = self.generate_embeddings([query])
        
        # Search in FAISS
        # Search for more results if we need to filter by document_id
        search_k = top_k * 10 if document_id else top_k
        distances, indices = self.index.search(query_embedding.astype('float32'), search_k)
        
        # Collect results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            
            doc_id = self.chunk_to_doc.get(idx)
            if doc_id is None:
                continue
            
            # Filter by document_id if specified
            if document_id and doc_id != document_id:
                continue
            
            # Calculate the chunk index within the document
            doc_start = self.doc_start_idx.get(doc_id, 0)
            chunk_idx = idx - doc_start
            
            # Safely get the chunk text
            chunks = self.document_chunks.get(doc_id, [])
            if 0 <= chunk_idx < len(chunks):
                chunk_text = chunks[chunk_idx]
                results.append((chunk_text, float(dist), doc_id))
            
            if len(results) >= top_k:
                break
        
        return results
    
    def get_document_chunks(self, document_id: str) -> List[str]:
        """
        Get all chunks for a specific document.
        
        Args:
            document_id: Document identifier
            
        Returns:
            List of chunks
        """
        self._initialize()
        return self.document_chunks.get(document_id, [])


embedding_service = EmbeddingService()
