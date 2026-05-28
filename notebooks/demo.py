"""
Multimodal RAG demo.
Run: make ingest-docs then make ingest-images then make serve then make demo
"""
from __future__ import annotations
import requests, json
from pathlib import Path
from PIL import Image
import io

BASE = "http://localhost:8086"

def post(path, payload=None, **kwargs):
    return requests.post(f"{BASE}{path}", json=payload, **kwargs)

def get(path):
    return requests.get(f"{BASE}{path}").json()

print("\n=== Multimodal RAG Demo ===\n")

# 1. Health + index stats
health = get("/health")
print(f"1. Health:")
print(f"   model loaded:  {health['model_loaded']}")
print(f"   text chunks:   {health['text_chunks']}")
print(f"   image chunks:  {health['image_chunks']}")

# 2. Index stats
stats = get("/index/stats")
print(f"\n2. Index stats:")
print(f"   {json.dumps(stats, indent=4)}")

# 3. Text-only queries
print("\n3. Text queries...")
text_queries = [
    "What is a candlestick chart used for?",
    "How is AUC-ROC score interpreted?",
    "What does high trading volume during a price move indicate?",
]
for q in text_queries:
    r = post("/query", {"query": q, "top_k": 3, "modality": "text"}).json()
    print(f"\n   Q: {q}")
    print(f"   A: {r['answer'][:200]}...")
    print(f"   faithfulness={r['eval']['faithfulness']}")

# 4. Upload a test image and query it
print("\n4. Uploading a test image...")
img = Image.new("RGB", (200, 200), color="steelblue")
buf = io.BytesIO()
img.save(buf, format="PNG")
buf.seek(0)

r = requests.post(
    f"{BASE}/ingest/image",
    files={"file": ("blue_chart.png", buf.getvalue(), "image/png")},
)
if r.status_code == 200:
    ingest_result = r.json()
    print(f"   Ingested: {ingest_result['filename']}")
    print(f"   Caption:  {ingest_result['caption'][:150]}...")
else:
    print(f"   Upload failed: {r.status_code}")

# 5. Multimodal query — retrieves from both text and images
print("\n5. Multimodal query (text + images)...")
r = post("/query", {
    "query": "Show me examples of blue or financial charts",
    "top_k": 5,
}).json()
print(f"   Q: {r['query']}")
print(f"   A: {r['answer'][:300]}...")
print(f"   Modalities used: {r['modalities_used']}")
print(f"   Text chunks:  {r['eval']['text_chunks_retrieved']}")
print(f"   Image chunks: {r['eval']['image_chunks_retrieved']}")

# 6. Image-only query
print("\n6. Image-only query...")
r = post("/query", {
    "query": "What images are in the index?",
    "top_k": 3,
    "modality": "image",
}).json()
print(f"   Retrieved {len(r['chunks'])} image chunks")
for c in r["chunks"]:
    print(f"   [{c['score']:.3f}] {c['text'][:100]}...")

# 7. Documents list
print("\n7. Indexed documents:")
docs = get("/documents?limit=10")
for d in docs:
    print(f"   [{d['modality']}] {d['title']}")

# 8. Query stats
print("\n8. Query stats:")
qs = get("/query/stats")
print(f"   {json.dumps(qs, indent=4)}")

print(f"\nAPI docs → http://localhost:8086/docs")
print("\nDone.")
