import fitz  # PyMuPDF

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract raw text from a PDF file.

    Args:
        file_bytes (bytes): The PDF file content in bytes.

    Returns:
        str: Extracted text from all pages.
    """
    text = ""

    # Open PDF from byte stream (no need to save it to disk)
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        # Iterate over each page in the PDF
        for page in doc:
            text += page.get_text()

    return text
