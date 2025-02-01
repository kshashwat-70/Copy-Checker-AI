from fastapi import UploadFile, File, APIRouter
from fastapi.responses import JSONResponse
from io import BytesIO
import fitz
from services.pdfer import extract_text_and_images

router = APIRouter()

@router.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    pdf_stream = BytesIO(await file.read())
    cont = extract_text_and_images(pdf_stream)

    return JSONResponse(content=cont)