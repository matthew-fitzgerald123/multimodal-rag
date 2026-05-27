from __future__ import annotations
from dataclasses import dataclass

@dataclass
class Chunk:
    chunk_id:  str
    doc_id:    str
    text:      str
    modality:  str    # text | image
    metadata:  dict

def chunk_text(
    doc_id: str,
    text: str,
    metadata: dict,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[Chunk]:
    chunks = []
    start = 0
    idx = 0
    text = text.strip()

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text_ = text[start:end].strip()
        if chunk_text_:
            chunks.append(Chunk(
                chunk_id=f"{doc_id}_chunk_{idx}",
                doc_id=doc_id,
                text=chunk_text_,
                modality="text",
                metadata={**metadata, "chunk_index": idx, "doc_id": doc_id},
            ))
        start += chunk_size - overlap
        idx += 1

    return chunks

def chunk_image(doc_id: str, caption: str, metadata: dict) -> list[Chunk]:
    """
    Images are stored as a single chunk — their caption.
    The caption IS the searchable content.
    """
    return [Chunk(
        chunk_id=f"{doc_id}_image_0",
        doc_id=doc_id,
        text=caption,
        modality="image",
        metadata={**metadata, "chunk_index": 0, "doc_id": doc_id, "modality": "image"},
    )]
