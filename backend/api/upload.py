from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.utils.pdf_reader import extract_text_from_pdf
from backend.utils.text_chunker import clean_text, chunk_text

from backend.core.state import embedding_service, vector_store
router = APIRouter()


@router.post("/upload-pdf")
def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    file_bytes = file.file.read()

    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="File is empty")

    if len(file_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 10MB)")

    try:
        raw_text = extract_text_from_pdf(file_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read PDF. File may be corrupt or invalid.")

    if not raw_text.strip():
        raise HTTPException(status_code=400, detail="PDF contains no extractable text")

    cleaned_text = clean_text(raw_text)
    chunks = chunk_text(cleaned_text)
    embeddings = embedding_service.embed_texts(chunks)
    vector_store.add(embeddings, chunks, source=file.filename)

    return {
        "filename": file.filename,
        "num_chunks": len(chunks),
        "status": "indexed successfully"
    }