from fastapi import APIRouter, HTTPException
from backend.core.state import embedding_service, vector_store, llm_service

router = APIRouter()

@router.post("/summarize")
def summarize_document():
    if vector_store.index.ntotal == 0:
        raise HTTPException(status_code=400, detail="No documents uploaded yet.")

    raw = " ".join(vector_store.text_chunks)[:4000]
    summary = llm_service.summarize(raw)
    return {"summary": summary}