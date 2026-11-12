import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.core.state import embedding_service, vector_store, llm_service

router = APIRouter()

class QuizRequest(BaseModel):
    num_questions: int = 5

@router.post("/generate-quiz")
def generate_quiz(request: QuizRequest):
    if vector_store.index.ntotal == 0:
        raise HTTPException(status_code=400, detail="No documents uploaded yet.")

    raw = " ".join(vector_store.text_chunks)[:4000]
    response = llm_service.generate_quiz(raw, request.num_questions)

    try:
        start = response.index("[")
        end = response.rindex("]") + 1
        questions = json.loads(response[start:end])
    except (ValueError, json.JSONDecodeError):
        questions = []

    return {"questions": questions}