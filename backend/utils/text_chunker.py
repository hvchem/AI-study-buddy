import re
from typing import List

def clean_text(text: str) -> str:
    """
    Clean extracted PDF text.

    - Removes extra whitespace
    - Removes null characters
    - Normalizes spacing

    Args:
        text (str): Raw extracted text.

    Returns:
        str: Cleaned text.
    """
    # Replace multiple spaces/newlines with a single space
    text = re.sub(r"\s+", " ", text)

    # Remove null characters (sometimes appear in PDFs)
    text = text.replace("\x00", "")

    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100
) -> List[str]:
    """
    Split text into overlapping chunks.

    This improves context retrieval for transformer models
    and avoids losing important information between chunks.

    Args:
        text (str): Cleaned text.
        chunk_size (int): Number of words per chunk.
        overlap (int): Number of overlapping words between chunks.

    Returns:
        List[str]: List of text chunks.
    """
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size

        # Create chunk from word range
        chunk = words[start:end]
        chunks.append(" ".join(chunk))

        # Move start forward with overlap
        start = end - overlap

    return chunks
