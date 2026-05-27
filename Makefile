serve:
	uvicorn app.main:app --reload --port 8086

test:
	pytest tests/ -v

demo:
	python notebooks/demo.py

ingest-docs:
	python scripts/ingest_documents.py

ingest-images:
	python scripts/ingest_images.py
