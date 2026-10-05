from pathlib import Path
import shutil

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    UploadFile
)

from ingestion.ingest_documents import ingest_document
from models.model import langchain_embeddings

from rag.kpi_extractor_rag import extract_financial_metrics
from database.save_metrics import save_metrics


router = APIRouter(
    prefix="/api",
    tags=["Ingestion"]
)


UPLOAD_DIR = Path("data/raw_pdfs")

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    company: str = Form(...),
    year: int = Form(...),
):


    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is missing."
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )


    company = company.strip()

    if not company:
        raise HTTPException(
            status_code=400,
            detail="Company name is required."
        )


    if year < 1900 or year > 2100:
        raise HTTPException(
            status_code=400,
            detail="Invalid year. Please enter a year between 1900 and 2100."
        )

    try:


        file_path = (
            UPLOAD_DIR /
            Path(file.filename).name
        )

        with file_path.open("wb") as buffer:
            shutil.copyfileobj(
                file.file,
                buffer
            )


        ingest_document(
            pdf_path=str(file_path),
            embeddings=langchain_embeddings,
            company=company,
            year=str(year),
        )


        metrics = extract_financial_metrics(
            company=company,
            year=year
        )


        save_metrics(
            company=company,
            year=year,
            metrics=metrics
        )


        return {
            "status": "success",

            "message": (
                "Document uploaded, indexed, "
                "financial metrics extracted, "
                "and saved successfully."
            ),

            "file_name": file.filename,

            "company": company,

            "year": year,

            "metrics": metrics
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Document processing failed: {str(e)}"
        )