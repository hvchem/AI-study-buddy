from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
import os
from app.models.schemas import UploadResponse
from app.services.pdf_processor import pdf_processor
from app.services.embedding_service import embedding_service
from app.core.config import settings

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF document for processing.
    
    Args:
        file: PDF file to upload
        
    Returns:
        Upload response with document ID and processing status
    """
    # Validate file extension
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Check file size
    file_content = await file.read()
    if len(file_content) > settings.max_upload_size:
        raise HTTPException(
            status_code=400,
            detail=f"File size exceeds maximum allowed size of {settings.max_upload_size / 1024 / 1024}MB"
        )
    
    try:
        # Save uploaded file
        document_id, file_path = pdf_processor.save_uploaded_file(file_content, file.filename)
        
        # Process PDF: extract and chunk text
        full_text, chunks = pdf_processor.process_pdf(file_path)
        
        # Generate and store embeddings
        embedding_service.add_document(document_id, chunks)
        
        return UploadResponse(
            document_id=document_id,
            filename=file.filename,
            status="success",
            num_chunks=len(chunks),
            message=f"Document processed successfully with {len(chunks)} chunks"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


@router.get("/list")
async def list_documents():
    """
    List all uploaded documents.
    
    Returns:
        List of document IDs and metadata
    """
    try:
        documents = []
        for doc_id, chunks in embedding_service.document_chunks.items():
            documents.append({
                "document_id": doc_id,
                "num_chunks": len(chunks)
            })
        return {"documents": documents, "total": len(documents)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")
