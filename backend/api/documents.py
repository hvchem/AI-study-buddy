from fastapi import APIRouter
from backend.core.state import vector_store

router = APIRouter()

@router.get("/documents")
def list_documents():
    sources = list(set(meta["source"] for meta in vector_store.metadata))
    return {
        "count": len(sources),
        "documents": sources
    }

@router.delete("/documents")
def clear_documents():
    vector_store.text_chunks.clear()
    vector_store.metadata.clear()
    vector_store.index.reset()
    return {"status": "All documents cleared"}