import os
import uuid
from typing import List, Tuple
from PyPDF2 import PdfReader
from app.core.config import settings


class PDFProcessor:
    """Service for processing PDF files."""
    
    def __init__(self):
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """
        Extract text content from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text as a string
        """
        try:
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to be chunked
            
        Returns:
            List of text chunks
        """
        # Remove extra whitespace and split into words
        words = text.split()
        chunks = []
        
        # Create chunks with overlap
        start = 0
        while start < len(words):
            end = start + self.chunk_size
            chunk = " ".join(words[start:end])
            if chunk.strip():  # Only add non-empty chunks
                chunks.append(chunk.strip())
            start = end - self.chunk_overlap
            
            # Prevent infinite loop if chunk_overlap >= chunk_size
            if self.chunk_overlap >= self.chunk_size:
                start = end
        
        return chunks
    
    def process_pdf(self, file_path: str) -> Tuple[str, List[str]]:
        """
        Process a PDF file: extract text and chunk it.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Tuple of (full_text, chunks)
        """
        text = self.extract_text_from_pdf(file_path)
        chunks = self.chunk_text(text)
        return text, chunks
    
    def save_uploaded_file(self, file_content: bytes, filename: str) -> Tuple[str, str]:
        """
        Save uploaded file to disk.
        
        Args:
            file_content: File content as bytes
            filename: Original filename
            
        Returns:
            Tuple of (document_id, file_path)
        """
        # Generate unique document ID
        document_id = str(uuid.uuid4())
        
        # Create uploads directory if it doesn't exist
        os.makedirs(settings.upload_dir, exist_ok=True)
        
        # Save file with document ID as prefix
        file_extension = os.path.splitext(filename)[1]
        new_filename = f"{document_id}{file_extension}"
        file_path = os.path.join(settings.upload_dir, new_filename)
        
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        return document_id, file_path


pdf_processor = PDFProcessor()
