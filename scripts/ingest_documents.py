"""
Ingests .txt files from data/documents/ into ChromaDB + Postgres.
Run: make ingest-docs
"""
from __future__ import annotations
import sys, uuid
sys.path.insert(0, ".")

from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from app.database import engine, SessionLocal
from app.models import Base, Document
from app.chunker import chunk_text
from app.vector_store import vector_store

Base.metadata.create_all(bind=engine)
DATA_DIR = Path("./data/documents")

def ingest():
    db = SessionLocal()
    files = list(DATA_DIR.glob("*.txt"))
    if not files:
        print("No .txt files in data/documents/. Add some first.")
        sys.exit(1)

    for path in files:
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            continue

        existing = db.query(Document).filter_by(title=path.stem).first()
        if existing:
            print(f"  Skipping {path.stem}: already ingested")
            continue

        doc_id = str(uuid.uuid4())[:8]
        doc = Document(
            doc_id=doc_id,
            title=path.stem,
            content=content,
            modality="text",
            source_path=str(path),
        )
        db.add(doc)
        db.commit()

        chunks = chunk_text(
            doc_id=doc_id,
            text=content,
            metadata={"title": path.stem, "modality": "text", "source": str(path)},
        )
        vector_store.add_chunks(chunks)
        print(f"  Ingested: {path.stem} ({len(chunks)} chunks)")

    db.close()
    print(f"\nText chunks in index: {vector_store.count('text')}")

if __name__ == "__main__":
    ingest()
