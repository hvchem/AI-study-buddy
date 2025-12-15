from fastapi import APIRouter, UploadFile, File
from backend.utils.pdf_reader import extract_text_from_pdf
from backend.utils.text_chunker import clean_text, chunk_text

from backend.core.state import embedding_service, vector_store
router = APIRouter()


@router.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file and process it.

    Steps:
    1. Read PDF file bytes
    2. Extract raw text
    3. Clean the text
    4. Split text into chunks

    Returns metadata for validation.
    """
    # Read uploaded file as bytes
    file_bytes = await file.read()

    # Extract text from PDF
    raw_text = extract_text_from_pdf(file_bytes)

    # Clean extracted text
    cleaned_text = clean_text(raw_text)

    # Split text into overlapping chunks
    chunks = chunk_text(cleaned_text)
    
    # Convert chunks to embeddings
    embeddings = embedding_service.embed_texts(chunks)


    # Store in FAISS
    vector_store.add(embeddings, chunks)

    return {
        "filename": file.filename,
        "num_chunks": len(chunks),
        "status":  "indexed successfully"
    }
