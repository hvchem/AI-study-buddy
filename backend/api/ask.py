from fastapi import APIRouter
from pydantic import BaseModel
from backend.core.state import embedding_service, vector_store

router = APIRouter()

# Use the same  shared instances 

class QuestionRequest(BaseModel):
    question: str


@router.post("/ask")
def ask_question(request: QuestionRequest):
    # Convert question to embedding 
    question_embedding = embedding_service.embed_texts([request.question])

    # Retrieve most relevant chunks
    results = vector_store.search(question_embedding)

    return {
        "question": request.question,
        "answers": results
    }