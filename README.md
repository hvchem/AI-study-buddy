# AI Study Buddy

AI Study Buddy is an AI-powered study assistant that helps students learn more effectively by allowing them to upload PDF documents and interact with them using natural language. The application leverages state-of-the-art AI technologies to answer questions, summarize content, and generate quizzes.

## Features

- **PDF Upload & Processing**: Upload PDF documents and automatically extract and chunk text
- **Question Answering**: Ask questions about your documents and get AI-powered answers using RAG (Retrieval-Augmented Generation)
- **Document Summarization**: Generate concise summaries of your study materials
- **Quiz Generation**: Automatically create quiz questions from your documents
- **Vector Search**: Fast semantic search using FAISS and sentence embeddings
- **RESTful API**: Clean FastAPI-based API with interactive documentation

## Technology Stack

- **Backend Framework**: FastAPI
- **PDF Processing**: PyPDF2
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector Store**: FAISS (Facebook AI Similarity Search)
- **Language Model**: Hugging Face Transformers (FLAN-T5)
- **Python**: 3.8+

## Project Structure

```
AI-study-buddy/
├── app/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── documents.py      # Document upload and management endpoints
│   │   └── study.py           # Study-related endpoints (Q&A, summarization, quiz)
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py          # Application configuration
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py         # Pydantic models for request/response
│   └── services/
│       ├── __init__.py
│       ├── pdf_processor.py   # PDF text extraction and chunking
│       ├── embedding_service.py # Embedding generation and FAISS management
│       └── llm_service.py     # LLM-based text generation
├── uploads/                    # Uploaded PDF files (gitignored)
├── faiss_index/               # FAISS index storage (gitignored)
├── main.py                    # Application entry point
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/hvchem/AI-study-buddy.git
   cd AI-study-buddy
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   Note: The first run will download AI models (~500MB total):
   - sentence-transformers/all-MiniLM-L6-v2 (~90MB)
   - google/flan-t5-base (~900MB)

4. **Run the application**
   ```bash
   python main.py
   ```

   Or using uvicorn directly:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

5. **Access the API documentation**
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

## API Endpoints

### Health Check

- **GET** `/` - Root endpoint
- **GET** `/health` - Health check

### Document Management

- **POST** `/documents/upload` - Upload a PDF document
  ```json
  {
    "file": "<PDF file>"
  }
  ```
  Response:
  ```json
  {
    "document_id": "uuid",
    "filename": "example.pdf",
    "status": "success",
    "num_chunks": 42,
    "message": "Document processed successfully"
  }
  ```

- **GET** `/documents/list` - List all uploaded documents

### Study Features

- **POST** `/study/question` - Ask a question about documents
  ```json
  {
    "question": "What is photosynthesis?",
    "document_id": "optional-uuid"
  }
  ```
  Response:
  ```json
  {
    "question": "What is photosynthesis?",
    "answer": "Photosynthesis is the process...",
    "sources": ["Doc: uuid, Distance: 0.23"],
    "confidence": 0.85
  }
  ```

- **POST** `/study/summarize` - Summarize a document
  ```json
  {
    "document_id": "uuid",
    "max_length": 150
  }
  ```

- **POST** `/study/quiz` - Generate quiz questions
  ```json
  {
    "document_id": "uuid",
    "num_questions": 5
  }
  ```

## Usage Examples

### Using cURL

1. **Upload a PDF**
   ```bash
   curl -X POST "http://localhost:8000/documents/upload" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/your/document.pdf"
   ```

2. **Ask a question**
   ```bash
   curl -X POST "http://localhost:8000/study/question" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is the main topic?"}'
   ```

3. **Generate a summary**
   ```bash
   curl -X POST "http://localhost:8000/study/summarize" \
     -H "Content-Type: application/json" \
     -d '{"document_id": "your-document-id", "max_length": 150}'
   ```

### Using Python Requests

```python
import requests

# Upload a document
with open("study_material.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/documents/upload",
        files={"file": f}
    )
doc_id = response.json()["document_id"]

# Ask a question
response = requests.post(
    "http://localhost:8000/study/question",
    json={
        "question": "What are the key concepts?",
        "document_id": doc_id
    }
)
print(response.json()["answer"])

# Generate quiz
response = requests.post(
    "http://localhost:8000/study/quiz",
    json={
        "document_id": doc_id,
        "num_questions": 5
    }
)
questions = response.json()["questions"]
```

## Configuration

Configuration can be customized via environment variables or a `.env` file:

```env
# App Settings
APP_NAME="AI Study Buddy"
VERSION="1.0.0"
DEBUG=False

# API Settings
API_HOST=0.0.0.0
API_PORT=8000

# File Upload Settings
UPLOAD_DIR=uploads
MAX_UPLOAD_SIZE=10485760  # 10MB in bytes

# Model Settings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL=google/flan-t5-base

# RAG Settings
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K_RESULTS=3

# FAISS Settings
FAISS_INDEX_PATH=faiss_index
```

## How It Works

### RAG (Retrieval-Augmented Generation) Pipeline

1. **Document Upload**: PDF is uploaded and text is extracted
2. **Chunking**: Text is split into overlapping chunks (default: 500 words with 50-word overlap)
3. **Embedding**: Each chunk is converted to a vector embedding using sentence-transformers
4. **Indexing**: Embeddings are stored in a FAISS vector database for fast similarity search
5. **Query Processing**: When a question is asked:
   - Question is converted to an embedding
   - FAISS retrieves the most similar chunks
   - Relevant chunks are passed as context to the LLM
   - LLM generates an answer based on the context

### Models Used

- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
  - Fast and efficient for semantic similarity
  - 384-dimensional embeddings
  - Great for question-context matching

- **Language Model**: `google/flan-t5-base`
  - Instruction-tuned T5 model
  - Good for Q&A, summarization, and text generation
  - Can run on CPU (slower) or GPU (faster)

## Performance Considerations

- **First-time setup**: Models will be downloaded on first run (~1.5GB total)
- **CPU vs GPU**: The app works on CPU but is much faster with a GPU
- **Memory**: Requires ~4GB RAM for models and operations
- **PDF Size**: Large PDFs (>100 pages) may take longer to process

## Troubleshooting

### Models not downloading
- Ensure you have internet connection
- Models are cached in `~/.cache/huggingface/`

### Out of memory errors
- Reduce `chunk_size` in configuration
- Use a smaller LLM model like `google/flan-t5-small`

### Slow performance
- Enable GPU support by installing `torch` with CUDA
- Reduce `top_k_results` for faster retrieval
- Use `faiss-gpu` instead of `faiss-cpu` if you have a GPU

## Future Enhancements

- [ ] Support for more document formats (DOCX, TXT, HTML)
- [ ] Multiple choice quiz generation
- [ ] User authentication and document management
- [ ] Support for multiple languages
- [ ] Chat history and conversation context
- [ ] Web UI for easier interaction
- [ ] Batch processing of documents
- [ ] Export summaries and quizzes

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- FastAPI for the excellent web framework
- Hugging Face for transformers and models
- Facebook AI Research for FAISS
- sentence-transformers for embedding models
