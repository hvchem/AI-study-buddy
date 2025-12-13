from fastapi import APIRouter, HTTPException
from app.models.schemas import (
    QuestionRequest,
    QuestionResponse,
    SummaryRequest,
    SummaryResponse,
    QuizRequest,
    QuizResponse,
    QuizQuestion,
)
from app.services.llm_service import llm_service
from app.services.embedding_service import embedding_service

router = APIRouter(prefix="/study", tags=["study"])


@router.post("/question", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a question and get an answer using RAG.
    
    Args:
        request: Question request with question and optional document ID
        
    Returns:
        Answer with sources and confidence score
    """
    try:
        result = llm_service.rag_answer(
            question=request.question,
            document_id=request.document_id
        )
        
        return QuestionResponse(
            question=request.question,
            answer=result["answer"],
            sources=result["sources"],
            confidence=result.get("confidence")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error answering question: {str(e)}")


@router.post("/summarize", response_model=SummaryResponse)
async def summarize_document(request: SummaryRequest):
    """
    Generate a summary of a document.
    
    Args:
        request: Summary request with document ID and options
        
    Returns:
        Document summary
    """
    try:
        # Get document chunks
        chunks = embedding_service.get_document_chunks(request.document_id)
        
        if not chunks:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Combine chunks for summarization (limit to avoid token limits)
        text = " ".join(chunks[:10])  # Use first 10 chunks
        
        # Generate summary
        summary = llm_service.summarize_text(text, max_length=request.max_length)
        
        return SummaryResponse(
            document_id=request.document_id,
            summary=summary
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error summarizing document: {str(e)}")


@router.post("/quiz", response_model=QuizResponse)
async def generate_quiz(request: QuizRequest):
    """
    Generate a quiz from a document.
    
    Args:
        request: Quiz request with document ID and number of questions
        
    Returns:
        Generated quiz questions
    """
    try:
        # Get document chunks
        chunks = embedding_service.get_document_chunks(request.document_id)
        
        if not chunks:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Combine chunks for quiz generation
        text = " ".join(chunks[:15])  # Use first 15 chunks
        
        # Generate quiz questions
        raw_questions = llm_service.generate_quiz_questions(text, num_questions=request.num_questions)
        
        # Format questions
        questions = []
        for q in raw_questions:
            questions.append(QuizQuestion(
                question=q["question"],
                options=[],  # Could be enhanced to generate multiple choice options
                correct_answer=None
            ))
        
        return QuizResponse(
            document_id=request.document_id,
            questions=questions
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating quiz: {str(e)}")
