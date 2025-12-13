from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from app.core.config import settings


class LLMService:
    """Service for LLM-based text generation tasks."""
    
    def __init__(self):
        self.model_name = settings.llm_model
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def _load_model(self):
        """Load the language model and tokenizer."""
        if self.model is None:
            print(f"Loading LLM model: {self.model_name} on {self.device}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
    
    def generate_text(self, prompt: str, max_length: int = 200, min_length: int = 20) -> str:
        """
        Generate text using the LLM.
        
        Args:
            prompt: Input prompt
            max_length: Maximum length of generated text
            min_length: Minimum length of generated text
            
        Returns:
            Generated text
        """
        self._load_model()
        
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                min_length=min_length,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    
    def answer_question(self, question: str, context_chunks: List[str]) -> str:
        """
        Answer a question based on context chunks using RAG.
        
        Args:
            question: The question to answer
            context_chunks: List of relevant text chunks
            
        Returns:
            Generated answer
        """
        # Combine context chunks
        context = " ".join(context_chunks[:3])  # Use top 3 chunks
        
        # Create prompt for the model
        prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        
        answer = self.generate_text(prompt, max_length=200, min_length=10)
        return answer
    
    def summarize_text(self, text: str, max_length: int = 150) -> str:
        """
        Summarize a given text.
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary
            
        Returns:
            Summary text
        """
        # Truncate text if too long
        max_input_length = 1000
        if len(text.split()) > max_input_length:
            text = " ".join(text.split()[:max_input_length])
        
        prompt = f"Summarize the following text:\n\n{text}\n\nSummary:"
        
        summary = self.generate_text(prompt, max_length=max_length, min_length=30)
        return summary
    
    def generate_quiz_questions(self, text: str, num_questions: int = 5) -> List[Dict[str, str]]:
        """
        Generate quiz questions from text.
        
        Args:
            text: Source text for quiz generation
            num_questions: Number of questions to generate
            
        Returns:
            List of question dictionaries
        """
        questions = []
        
        # Split text into chunks if too long
        chunks = text.split(". ")
        
        # Generate questions from different parts of the text
        for i in range(min(num_questions, len(chunks))):
            if i < len(chunks) and chunks[i].strip():
                chunk = chunks[i * (len(chunks) // num_questions)] if len(chunks) > num_questions else chunks[i]
                
                prompt = f"Generate a question about this text:\n\n{chunk}\n\nQuestion:"
                question = self.generate_text(prompt, max_length=50, min_length=5)
                
                # Clean up the question
                if question and not question.endswith("?"):
                    question += "?"
                
                questions.append({
                    "question": question,
                    "context": chunk
                })
        
        return questions
    
    def rag_answer(self, question: str, document_id: str = None) -> Dict:
        """
        Answer a question using Retrieval-Augmented Generation.
        
        Args:
            question: The question to answer
            document_id: Optional document ID to search in
            
        Returns:
            Dictionary with answer and sources
        """
        # Import here to avoid circular dependency
        from app.services.embedding_service import embedding_service
        
        # Retrieve relevant chunks
        search_results = embedding_service.search(
            query=question,
            top_k=settings.top_k_results,
            document_id=document_id
        )
        
        if not search_results:
            return {
                "answer": "I don't have enough information to answer this question.",
                "sources": [],
                "confidence": 0.0
            }
        
        # Extract chunks and calculate confidence
        chunks = [chunk for chunk, dist, doc_id in search_results]
        sources = [f"Doc: {doc_id}, Distance: {dist:.2f}" for chunk, dist, doc_id in search_results]
        
        # Calculate confidence based on distance (lower distance = higher confidence)
        avg_distance = sum(dist for _, dist, _ in search_results) / len(search_results)
        confidence = max(0.0, 1.0 - (avg_distance / 100.0))  # Normalize distance to confidence
        
        # Generate answer
        answer = self.answer_question(question, chunks)
        
        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence
        }


llm_service = LLMService()
