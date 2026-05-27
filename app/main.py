from __future__ import annotations
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Any
import os, uuid, shutil, re
from pathlib import Path
from dotenv import load_dotenv

from app.database import get_db, engine
from app.models import Base, Document, QueryLog
from app.vector_store import vector_store
from app.generator import generator
from app.image_captioner import caption_image, get_image_metadata
from app.chunker import chunk_text, chunk_image

load_dotenv()
Base.metadata.create_all(bind=engine)

UPLOAD_DIR = Path("./data/images")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Multimodal RAG", version="1.0.0")

@app.on_event("startup")
def startup():
    generator.load_model()
    print(f"Text chunks:  {vector_store.count('text')}")
    print(f"Image chunks: {vector_store.count('image')}")

# Query

class QueryReq(BaseModel):
    query: str
    top_k: int = 5
    modality: str = None    # None=all, "text"=text only, "image"=image only

@app.post("/query", tags=["rag"])
def query(req: QueryReq, db: Session = Depends(get_db)):
    total = vector_store.count()
    if total == 0:
        raise HTTPException(400, "No documents indexed. Run: make ingest-docs and make ingest-images")

    chunks = vector_store.query(req.query, top_k=req.top_k, modality=req.modality)
    if not chunks:
        raise HTTPException(404, "No relevant content found")

    answer = generator.answer(req.query, chunks)
    modalities_used = list(set(c.get("modality", "text") for c in chunks))

    log = QueryLog(
        query=req.query,
        answer=answer,
        retrieved_ids=[c["chunk_id"] for c in chunks],
        modalities_used=modalities_used,
    )
    db.add(log)
    db.commit()

    return {
        "query":    req.query,
        "answer":   answer,
        "chunks":   chunks,
        "modalities_used": modalities_used,
    }

# Image ingest

@app.post("/ingest/image", tags=["ingest"])
async def ingest_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    suffix = Path(file.filename).suffix.lower()
    allowed = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
    if suffix not in allowed:
        raise HTTPException(400, f"Unsupported format: {suffix}. Allowed: {allowed}")

    doc_id = str(uuid.uuid4())[:8]
    save_path = UPLOAD_DIR / f"{doc_id}{suffix}"

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        meta = get_image_metadata(str(save_path))
        caption = caption_image(str(save_path))
    except Exception as e:
        save_path.unlink(missing_ok=True)
        raise HTTPException(500, f"Captioning failed: {e}")

    doc = Document(
        doc_id=doc_id,
        title=Path(file.filename).stem,
        content=caption,
        modality="image",
        source_path=str(save_path),
        metadata_={**meta, "original_filename": file.filename},
    )
    db.add(doc)
    db.commit()

    chunks = chunk_image(
        doc_id=doc_id,
        caption=caption,
        metadata={
            "title":    Path(file.filename).stem,
            "modality": "image",
            "filename": file.filename,
            **meta,
        },
    )
    vector_store.add_chunks(chunks)

    return {
        "doc_id":   doc_id,
        "filename": file.filename,
        "caption":  caption[:300],
        "metadata": meta,
        "chunks_added": len(chunks),
    }

@app.post("/ingest/text", tags=["ingest"])
def ingest_text(
    title: str,
    content: str,
    db: Session = Depends(get_db),
):
    existing = db.query(Document).filter_by(title=title).first()
    if existing:
        raise HTTPException(400, f"Document '{title}' already ingested")

    doc_id = str(uuid.uuid4())[:8]
    doc = Document(
        doc_id=doc_id,
        title=title,
        content=content,
        modality="text",
    )
    db.add(doc)
    db.commit()

    chunks = chunk_text(
        doc_id=doc_id,
        text=content,
        metadata={"title": title, "modality": "text"},
    )
    vector_store.add_chunks(chunks)

    return {
        "doc_id":       doc_id,
        "title":        title,
        "chunks_added": len(chunks),
    }

@app.get("/health")
def health():
    return {
        "status":       "ok",
        "model_loaded": generator.model is not None,
    }
