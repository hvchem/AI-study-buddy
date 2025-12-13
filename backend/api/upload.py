from fastapi import APIRouter, UploadFile, File
from backend.utils.pdf_reader import extract_text_from_pdf

router = APIRouter()

@router.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    file_bytes = await file.read()
    text = extract_text_from_pdf(file_bytes)
    # Here you can process the extracted text as needed

    return {
        "filename": file.filename,
        "text_length": len(text)
    }
