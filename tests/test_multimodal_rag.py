from __future__ import annotations
import pytest
from fastapi.testclient import TestClient
from pathlib import Path
from PIL import Image
import io, sys, uuid
sys.path.insert(0, ".")

# Ingest sample docs before tests
from scripts.ingest_documents import ingest as ingest_docs
ingest_docs()

from app.main import app

client = TestClient(app)

def make_test_image(width: int = 100, height: int = 100, color: str = "blue") -> bytes:
    """Creates a minimal test image in memory."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# ── Health + index ─────────────────────────────────────────

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True

def test_index_stats():
    r = client.get("/index/stats")
    assert r.status_code == 200
    data = r.json()
    assert data["total_chunks"] > 0
    assert data["text_chunks"] >= 0
    assert data["image_chunks"] >= 0

# ── Text query ─────────────────────────────────────────────

def test_text_query():
    r = client.post("/query", json={
        "query": "What is a candlestick chart?",
        "top_k": 3,
    })
    assert r.status_code == 200
    data = r.json()
    assert "answer" in data
    assert len(data["answer"]) > 0
    assert "chunks" in data
    assert "eval" in data

def test_text_only_query():
    r = client.post("/query", json={
        "query": "Explain AUC-ROC score",
        "top_k": 3,
        "modality": "text",
    })
    assert r.status_code == 200
    chunks = r.json()["chunks"]
    assert all(c.get("modality") != "image" for c in chunks)

def test_modalities_tracked():
    r = client.post("/query", json={
        "query": "What is precision and recall?",
        "top_k": 3,
    })
    assert r.status_code == 200
    data = r.json()
    assert "modalities_used" in data
    assert isinstance(data["modalities_used"], list)

# ── Image ingest ───────────────────────────────────────────

def test_image_upload_and_ingest():
    img_bytes = make_test_image(color="red")
    r = client.post(
        "/ingest/image",
        files={"file": ("test_chart.png", img_bytes, "image/png")},
    )
    assert r.status_code == 200
    data = r.json()
    assert "doc_id" in data
    assert "caption" in data
    assert data["chunks_added"] >= 1

def test_unsupported_image_format():
    r = client.post(
        "/ingest/image",
        files={"file": ("doc.pdf", b"fake pdf content", "application/pdf")},
    )
    assert r.status_code == 400

# ── Text ingest ────────────────────────────────────────────

def test_text_ingest():
    r = client.post(
        "/ingest/text",
        params={
            "title":   f"test_document_{uuid.uuid4().hex[:8]}",
            "content": "This is a test document about machine learning evaluation metrics including accuracy and F1 score.",
        },
    )
    assert r.status_code == 200
    assert r.json()["chunks_added"] >= 1

def test_duplicate_text_rejected():
    client.post(
        "/ingest/text",
        params={"title": "duplicate_test_doc", "content": "Some content here."},
    )
    r = client.post(
        "/ingest/text",
        params={"title": "duplicate_test_doc", "content": "Some content here."},
    )
    assert r.status_code == 400

# ── Monitoring ─────────────────────────────────────────────

def test_list_documents():
    r = client.get("/documents")
    assert r.status_code == 200
    assert isinstance(r.json(), list)

def test_list_documents_filter_by_modality():
    r = client.get("/documents?modality=text")
    assert r.status_code == 200
    docs = r.json()
    assert all(d["modality"] == "text" for d in docs)

def test_query_history():
    r = client.get("/query/history")
    assert r.status_code == 200
    assert isinstance(r.json(), list)

def test_query_stats():
    r = client.get("/query/stats")
    assert r.status_code == 200
    data = r.json()
    assert "total_queries" in data or "message" in data
