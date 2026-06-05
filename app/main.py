from __future__ import annotations
from contextlib import asynccontextmanager
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    generator.load_model()
    print(f"Text chunks:  {vector_store.count('text')}")
    print(f"Image chunks: {vector_store.count('image')}")
    yield


app = FastAPI(title="Multimodal RAG", version="1.0.0", lifespan=lifespan)

# ── Query ─────────────────────────────────────────────────

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

    # Simple faithfulness proxy
    context_text = " ".join(c["text"] for c in chunks).lower()
    answer_tokens = set(re.findall(r"\w+", answer.lower()))
    context_tokens = set(re.findall(r"\w+", context_text))
    stopwords = {"the","a","an","is","are","in","on","at","to","of","and","or","it","this","that"}
    answer_tokens -= stopwords
    faithfulness = round(
        len(answer_tokens & context_tokens) / max(len(answer_tokens), 1), 4
    )

    log = QueryLog(
        query=req.query,
        answer=answer,
        retrieved_ids=[c["chunk_id"] for c in chunks],
        modalities_used=modalities_used,
        faithfulness=faithfulness,
    )
    db.add(log)
    db.commit()

    return {
        "query":    req.query,
        "answer":   answer,
        "chunks":   chunks,
        "modalities_used": modalities_used,
        "eval": {
            "faithfulness": faithfulness,
            "text_chunks_retrieved":  sum(1 for c in chunks if c.get("modality") != "image"),
            "image_chunks_retrieved": sum(1 for c in chunks if c.get("modality") == "image"),
        },
    }

# ── Image upload + ingest ─────────────────────────────────

@app.post("/ingest/image", tags=["ingest"])
async def ingest_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Upload and ingest an image on the fly.
    Captions it with mlx-lm and adds to the index immediately.
    """
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

# ── Monitoring ────────────────────────────────────────────

@app.get("/index/stats", tags=["monitoring"])
def index_stats():
    return {
        "total_chunks": vector_store.count(),
        "text_chunks":  vector_store.count("text"),
        "image_chunks": vector_store.count("image"),
        "embed_model":  os.getenv("EMBED_MODEL"),
        "gen_model":    os.getenv("GEN_MODEL"),
    }

@app.delete("/documents/{doc_id}", tags=["ingest"])
def delete_document(doc_id: str, db: Session = Depends(get_db)):
    doc = db.query(Document).filter_by(doc_id=doc_id).first()
    if doc is None:
        raise HTTPException(404, f"Document '{doc_id}' not found")

    if doc.source_path:
        path = Path(doc.source_path)
        if path.exists():
            path.unlink()

    vector_store.delete_by_doc(doc_id)
    db.delete(doc)
    db.commit()
    return {"doc_id": doc_id, "deleted": True}

@app.get("/documents", tags=["monitoring"])
def list_documents(modality: str = None, limit: int = 20, db: Session = Depends(get_db)):
    q = db.query(Document)
    if modality:
        q = q.filter_by(modality=modality)
    docs = q.order_by(Document.created_at.desc()).limit(limit).all()
    return [
        {
            "doc_id":   d.doc_id,
            "title":    d.title,
            "modality": d.modality,
            "content":  d.content[:150] + "..." if len(d.content) > 150 else d.content,
        }
        for d in docs
    ]

@app.get("/query/history", tags=["monitoring"])
def query_history(limit: int = 20, db: Session = Depends(get_db)):
    logs = (
        db.query(QueryLog)
        .order_by(QueryLog.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "query":           l.query,
            "answer":          l.answer[:200] + "..." if len(l.answer) > 200 else l.answer,
            "modalities_used": l.modalities_used,
            "faithfulness":    l.faithfulness,
            "created_at":      str(l.created_at),
        }
        for l in logs
    ]

@app.get("/query/stats", tags=["monitoring"])
def query_stats(db: Session = Depends(get_db)):
    logs = db.query(QueryLog).all()
    if not logs:
        return {"message": "No queries yet"}
    total = len(logs)
    multimodal = sum(1 for l in logs if len(l.modalities_used or []) > 1)
    avg_faith = round(
        sum(l.faithfulness for l in logs if l.faithfulness) /
        max(sum(1 for l in logs if l.faithfulness), 1), 4
    )
    return {
        "total_queries":      total,
        "multimodal_queries": multimodal,
        "text_only_queries":  sum(1 for l in logs if l.modalities_used == ["text"]),
        "image_only_queries": sum(1 for l in logs if l.modalities_used == ["image"]),
        "avg_faithfulness":   avg_faith,
    }

@app.get("/health")
def health():
    return {
        "status":        "ok",
        "model_loaded":  generator.model is not None,
        "text_chunks":   vector_store.count("text"),
        "image_chunks":  vector_store.count("image"),
    }
