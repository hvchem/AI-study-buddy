# AI Study Buddy

Upload a PDF and turn it into a study session. The app extracts text, embeds it
into vectors, retrieves the most relevant passages, and uses a local LLM to
answer questions, summarize the document, and generate quizzes.

## Features

- **PDF ingestion** — extract, clean, and chunk text, then index it in FAISS
- **Question answering (RAG)** — retrieve relevant chunks and generate a grounded answer
- **Summarization** — produce a short summary of the whole document
- **Quiz generation** — generate multiple-choice questions as structured JSON
- **Multiple documents** — every chunk tracks its source filename
- **Validation** — PDF type/size/empty checks with clear error messages

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support (tested on RTX 3060)
- ~6 GB VRAM for the 4-bit quantized LLM
- ~6 GB disk space for models

## Setup

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

> Make sure PyTorch is the CUDA build, otherwise the LLM will fall back to CPU:
>
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
> ```

## Run

```bash
uvicorn backend.main:app --reload
```

Server starts at `http://127.0.0.1:8000`.
Interactive API docs at `http://127.0.0.1:8000/docs`.

Models are loaded once on startup, so the first request is fast.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/upload-pdf` | Upload and index a PDF (max 10 MB) |
| POST | `/ask` | Ask a question about the uploaded documents |
| POST | `/summarize` | Summarize the uploaded documents |
| POST | `/generate-quiz` | Generate multiple-choice quiz questions |
| GET | `/documents` | List indexed documents |
| DELETE | `/documents` | Clear the index |

### Example

```bash
curl -X POST http://127.0.0.1:8000/upload-pdf -F "file=@notes.pdf"

curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is transfer learning?"}'

curl -X POST http://127.0.0.1:8000/summarize

curl -X POST http://127.0.0.1:8000/generate-quiz \
  -H "Content-Type: application/json" \
  -d '{"num_questions": 5}'
```

## Project Structure

```
backend/
├── main.py              # FastAPI app, lifespan/model preloading, routers
├── api/
│   ├── upload.py        # POST /upload-pdf
│   ├── ask.py           # POST /ask
│   ├── summarize.py     # POST /summarize
│   ├── quiz.py          # POST /generate-quiz
│   └── documents.py     # GET/DELETE /documents
├── core/
│   └── state.py         # shared service singletons
├── services/
│   ├── embeddings.py    # sentence-transformers embeddings
│   ├── vector_store.py  # FAISS index + metadata
│   └── llm.py           # local Mistral (4-bit) generation
└── utils/
    ├── pdf_reader.py    # PyMuPDF text extraction
    └── text_chunker.py  # cleaning + overlapping chunking
```

## Roadmap

- [ ] Per-document summarization and quizzes
- [ ] Subject/topic detection with 2 quizzes per subject
- [ ] Quiz grading with per-question explanations
- [ ] React frontend
- [ ] SaaS foundation (auth, Postgres + pgvector, object storage, workers)
