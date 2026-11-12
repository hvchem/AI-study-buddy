from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.core.state import embedding_service, vector_store, llm_service
import re
router = APIRouter()

class QuestionRequest(BaseModel):
    question: str

@router.post("/ask")
def ask_question(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if len(request.question) > 1000:
        raise HTTPException(status_code=400, detail="Question too long (max 1000 characters)")

    if vector_store.index.ntotal == 0:
        raise HTTPException(status_code=400, detail="No documents uploaded yet. Upload a PDF first.")

    try:
        question_embedding = embedding_service.embed_texts([request.question])
        results = vector_store.search(question_embedding)
        raw = " ".join([r["text"] for r in results])
        raw = re.sub(r"\d{2}:\d{2}\s+\d+\s*", "", raw)
        raw = re.sub(r"\s+", " ", raw).strip()
        context = raw[:2000]
        answer = llm_service.generate_answer(context, request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

    return {
        "question": request.question,
        "answer": answer,
        "sources": results
    }