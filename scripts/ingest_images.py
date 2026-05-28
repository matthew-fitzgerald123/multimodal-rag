"""
Ingests images from data/images/ -captions them with mlx-lm
then embeds captions into ChromaDB.
Supports: .jpg, .jpeg, .png, .gif, .webp
Run: make ingest-images
Note: first run loads the LLM for captioning -takes ~30s on Apple Silicon.
"""
from __future__ import annotations
import sys, uuid, os
sys.path.insert(0, ".")

from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from app.database import engine, SessionLocal
from app.models import Base, Document
from app.chunker import chunk_image
from app.vector_store import vector_store
from app.image_captioner import caption_image, get_image_metadata

Base.metadata.create_all(bind=engine)

IMAGE_DIR = Path("./data/images")
MAX_MB = float(os.getenv("MAX_IMAGE_SIZE_MB", 10))
SUPPORTED = {".jpg", ".jpeg", ".png", ".gif", ".webp"}

def ingest():
    db = SessionLocal()
    images = [p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in SUPPORTED]

    if not images:
        print("No images in data/images/ -add .jpg/.png/.webp files first")
        sys.exit(1)

    print(f"Found {len(images)} images. Captioning with mlx-lm...\n")

    for path in images:
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > MAX_MB:
            print(f"  Skipping {path.name} -{size_mb:.1f}MB exceeds {MAX_MB}MB limit")
            continue

        existing = db.query(Document).filter_by(title=path.stem).first()
        if existing:
            print(f"  Skipping {path.name} -already ingested")
            continue

        print(f"  Captioning: {path.name}...")
        try:
            meta = get_image_metadata(str(path))
            caption = caption_image(str(path))
        except Exception as e:
            print(f"  Error captioning {path.name}: {e}")
            continue

        doc_id = str(uuid.uuid4())[:8]
        doc = Document(
            doc_id=doc_id,
            title=path.stem,
            content=caption,
            modality="image",
            source_path=str(path),
            metadata_={**meta, "caption": caption[:200]},
        )
        db.add(doc)
        db.commit()

        chunks = chunk_image(
            doc_id=doc_id,
            caption=caption,
            metadata={
                "title":    path.stem,
                "modality": "image",
                "filename": path.name,
                "source":   str(path),
                **meta,
            },
        )
        vector_store.add_chunks(chunks)
        print(f"  Ingested: {path.name}")
        print(f"    Caption: {caption[:120]}...\n")

    db.close()
    print(f"Text chunks:  {vector_store.count('text')}")
    print(f"Image chunks: {vector_store.count('image')}")

if __name__ == "__main__":
    ingest()
