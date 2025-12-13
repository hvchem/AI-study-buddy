from fastapi import FastAPI
from backend.api.upload import router as upload_router

app = FastAPI(title="AI Study Buddy")
 
app.include_router(upload_router)

@app.get("/")
def root():
    return {"message": "AI Study Buddy API running"}
