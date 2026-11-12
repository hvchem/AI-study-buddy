from contextlib import asynccontextmanager
from fastapi import FastAPI
from backend.api.upload import router as upload_router
from backend.api.ask import router as ask_router
from backend.api.documents import router as documents_router
from backend.core.state import embedding_service, llm_service
from backend.api.summarize import router as summarize_router
from backend.api.quiz import router as quiz_router
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading embedding model...")
    _ = embedding_service.model
    print("Loading LLM model...")
    _ = llm_service.model
    print("All models loaded!")
    yield

app = FastAPI(title="AI Study Buddy", lifespan=lifespan)

app.include_router(upload_router)
app.include_router(ask_router)
app.include_router(documents_router)
app.include_router(summarize_router)
app.include_router(quiz_router)
@app.get("/")
def root():
    return {"message": "AI Study Buddy API running"}