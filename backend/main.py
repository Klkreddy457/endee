from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import time
import os
import uuid
from typing import Optional, List, Dict, Any

from backend.utils import logger
from backend.endee_client import endee_client
from backend.rag_pipeline import rag_pipeline
from backend.document_loader import document_loader, get_job_status, set_job_status, SUPPORTED_EXTENSIONS

# ── In-memory session store: session_id -> list of {role, content} ──────────
_sessions: Dict[str, List[Dict[str, str]]] = {}

# ── In-memory document registry: filename -> {chunks, job_id, timestamp} ────
_documents: Dict[str, Dict[str, Any]] = {}

app = FastAPI(
    title="Endee RAG Assistant",
    description="A complete production-ready RAG pipeline using Endee Vector Database.",
    version="1.0.0"
)

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure static directory exists
os.makedirs("backend/static", exist_ok=True)
app.mount("/static", StaticFiles(directory="backend/static"), name="static")

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None  # For conversation memory

class SourceChunk(BaseModel):
    text: str
    source: str
    score: float

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    chunks: list[SourceChunk] = []
    processing_time_ms: float
    session_id: str

class HealthResponse(BaseModel):
    status: str
    endee_connected: bool

@app.get("/", response_class=FileResponse)
def root():
    """Serve the Web UI page."""
    return FileResponse("backend/static/index.html")

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Check backend and Endee connectivity."""
    endee_status = endee_client.health_check()
    return {
        "status": "healthy",
        "endee_connected": endee_status
    }

def _run_ingestion(job_id: str, text: str, filename: str):
    """Background worker: embed text chunks and insert into Endee."""
    try:
        set_job_status(job_id, "processing", "Ingesting document...")
        start = time.time()
        result = rag_pipeline.ingest_document(text, filename)
        elapsed = round((time.time() - start) * 1000, 2)

        if result.get("status") == "error":
            set_job_status(job_id, "failed", result.get("message", "Unknown error"))
        else:
            chunks_count_msg = result.get("message", "")
            set_job_status(job_id, "complete", chunks_count_msg)
            # Register document
            _documents[filename] = {
                "filename": filename,
                "job_id": job_id,
                "message": chunks_count_msg,
                "indexed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
            logger.info(f"Job {job_id} complete in {elapsed}ms")
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        set_job_status(job_id, "failed", str(e))


@app.post("/upload")
async def upload_document(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """Upload a document for ingestion. Supported: .txt, .pdf, .docx, .md, .csv, .json"""
    ext = os.path.splitext(file.filename.lower())[1]
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    try:
        content = await file.read()

        # Parse the document text immediately (fast)
        text = document_loader.parse_file(file.filename, content)
        if not text.strip():
            raise HTTPException(status_code=400, detail="File appears to be empty or unreadable.")

        logger.info(f"Parsed '{file.filename}' ({len(text)} chars). Queuing ingestion...")

        # Create job and run ingestion in background
        job_id = str(uuid.uuid4())
        set_job_status(job_id, "queued", "Waiting to start...")
        background_tasks.add_task(_run_ingestion, job_id, text, file.filename)

        return {
            "message": f"'{file.filename}' accepted. Ingestion running in background.",
            "job_id": job_id,
            "status": "queued"
        }

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    """Poll the status of a background ingestion job."""
    job = get_job_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, **job}


@app.get("/documents")
def list_documents():
    """Return all indexed documents with metadata."""
    return {"documents": list(_documents.values())}


@app.delete("/documents/{filename}")
def delete_document(filename: str):
    """Remove a document from the registry (does not remove vectors from Endee)."""
    if filename not in _documents:
        raise HTTPException(status_code=404, detail="Document not found in registry")
    del _documents[filename]
    return {"message": f"'{filename}' removed from document registry."}

@app.get("/query", response_model=QueryResponse)
def ask_question_get(q: str, session_id: Optional[str] = None):
    """Ask a question to the RAG assistant via GET parameters."""
    return process_query(q, session_id)

@app.post("/query", response_model=QueryResponse)
def ask_question_post(request: QueryRequest):
    """Ask a question to the RAG assistant via POST body."""
    return process_query(request.query, request.session_id)

def process_query(query_text: str, session_id: Optional[str] = None):
    if not query_text or len(query_text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # Get or create session
    if not session_id:
        session_id = str(uuid.uuid4())
    history = _sessions.setdefault(session_id, [])

    try:
        start_time = time.time()
        result = rag_pipeline.query(query_text, history=history)
        elapsed = round((time.time() - start_time) * 1000, 2)

        # Append to session history
        history.append({"role": "user", "content": query_text})
        history.append({"role": "assistant", "content": result["answer"]})
        # Keep last 10 turns to avoid context overflow
        _sessions[session_id] = history[-20:]

        chunks = [
            {"text": c["text"], "source": c["source"], "score": round(c.get("score", 0.0), 4)}
            for c in result.get("chunks", [])
        ]

        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "chunks": chunks,
            "processing_time_ms": elapsed,
            "session_id": session_id
        }
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    """Clear conversation history for a session."""
    _sessions.pop(session_id, None)
    return {"message": "Session cleared."}


if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
