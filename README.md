# Multimodal RAG

A retrieval-augmented generation pipeline that handles both text and images. Images are captioned on ingest using a local vision-language model, then stored alongside text in ChromaDB. Queries retrieve from both modalities and generate a grounded answer with a faithfulness score.

## Stack

| Component | Library |
|---|---|
| API | FastAPI + uvicorn (port 8086) |
| Vector store | ChromaDB 0.5.3 (local, persistent) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 (CPU) |
| Generation | mlx-lm + Mistral-7B-Instruct-v0.3-4bit (Apple Silicon) |
| Image captioning | mlx-lm (vision-language model) |
| Image processing | Pillow |
| Document store + query logs | PostgreSQL + SQLAlchemy |

## Setup

```bash
# Create database
createdb multimodal_rag

# Install dependencies
pip install -r requirements.txt

# Ingest text documents from data/
make ingest-docs

# Ingest images from data/images/
make ingest-images
```

## Running

```bash
# Start API server (downloads Mistral-7B on first run, ~4GB)
make serve

# Run end-to-end demo
make demo

# Run tests
make test
```

## API Endpoints

### Query

| Method | Path | Description |
|---|---|---|
| POST | `/query` | Retrieve from text + image index, generate grounded answer |

Query supports `modality` filtering: `null` (all), `"text"`, or `"image"`.

### Ingest

| Method | Path | Description |
|---|---|---|
| POST | `/ingest/text` | Ingest a text document by title + content |
| POST | `/ingest/image` | Upload and caption an image (jpg/png/webp/gif) |

### Monitoring

| Method | Path | Description |
|---|---|---|
| GET | `/index/stats` | Total chunk count by modality, model names |
| GET | `/documents` | List ingested documents, filter by modality |
| GET | `/query/history` | Recent queries with answers and faithfulness |
| GET | `/query/stats` | Multimodal query breakdown, avg faithfulness |
| GET | `/health` | Server status + chunk counts |

Interactive docs at `http://localhost:8086/docs`.

## How It Works

Text documents are split into overlapping chunks and embedded with MiniLM. Images are captioned into a text description, then chunked and embedded the same way. All chunks live in a single ChromaDB collection with a `modality` metadata field. At query time, the top-k chunks across both modalities are retrieved, injected into a prompt, and passed to Mistral-7B for answer generation.

Faithfulness is estimated as the fraction of non-stopword answer tokens that also appear in the retrieved context.

## Project Structure

```
app/
  chunker.py          text + image chunk splitting
  vector_store.py     ChromaDB wrapper with modality filtering
  generator.py        MLX Mistral-7B answer generator
  image_captioner.py  vision-language captioning + metadata extraction
  main.py             FastAPI app
  models.py           SQLAlchemy models (Document, QueryLog)
  database.py         engine + session
scripts/
  ingest_documents.py  bulk ingest .txt files from data/
  ingest_images.py     bulk caption + ingest images from data/images/
data/
  images/              image upload directory (created on startup)
chroma_db/             persistent ChromaDB storage
notebooks/
  demo.py              end-to-end query demo
tests/
```

## Notes

- SentenceTransformer runs on CPU; MPS has stability issues with this model on Apple Silicon
- ChromaDB telemetry is disabled to prevent startup hangs
- The LLM agent in project_05 can query this service via its `search_documents` tool (port 8086)
