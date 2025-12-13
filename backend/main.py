from fastapi import FastAPI

app = FastAPI(title="AI Study Buddy")

@app.get("/")
def root():
    return {"message": "AI Study Buddy API running"}
