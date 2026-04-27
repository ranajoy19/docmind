import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from ingest import ingest_pdf
from query import ask_document

os.makedirs("data", exist_ok=True)
os.makedirs("chroma_store", exist_ok=True)

app = FastAPI(
    title="DocMind", description="RAG-powered document Q&A engine", version="1.0.0"
)

UPLOAD_DIR = "./data"


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list


@app.get("/")
def root():
    return FileResponse("static/index.html")


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    save_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        chunk_count = ingest_pdf(save_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
    return {"status": "success", "file": file.filename, "chunks_stored": chunk_count}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        result = ask_document(request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")
    return QueryResponse(
        question=result["question"], answer=result["answer"], sources=result["sources"]
    )


app.mount("/static", StaticFiles(directory="static"), name="static")
