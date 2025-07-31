# BoundLy (DallasAI Summer 2025)

**BoundLy** is an H-1B eligibility assistant with a hybrid RAG pipeline:
- Ingest redacted USCIS cases & policy PDFs
- Hybrid vector (AzureOpenAI embeddings) + BM25 search in Elasticsearch
- FastAPI endpoint wrapping an AzureOpenAI LLM for grounded answers

## Structure

```text
BoundLy_H1b/
├── backend/
├── data/
├── .env
├── .gitignore
└── README.md
```

## Getting started
1. `python3 -m venv .venv && source .venv/bin/activate`  
2. `pip install -r backend/requirements.txt`  
3. Fill in `.env` (see `.env.example`)  
4. `python backend/ingest.py` → index data  
5. `uvicorn backend.api:app --reload --host 0.0.0.0`  
6. POST to `/boundly/query`
