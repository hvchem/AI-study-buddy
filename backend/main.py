from fastapi import FastAPI
from backend.api.upload import router as upload_router
from backend.api.ask import router as ask_router

app = FastAPI(title="AI Study Buddy")
 
app.include_router(upload_router)
app.include_router(ask_router)

@app.get("/")
def root():
    return {"message": "AI Study Buddy API running"}
