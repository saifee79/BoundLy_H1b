# backend/hybrid_retriever.py

import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from elasticsearch import Elasticsearch
from sklearn.preprocessing import normalize
import numpy as np

load_dotenv()

# ─── ES client (API key auth) ────────────────────────────────
ES_HOST    = os.getenv("ELASTIC_HOST")
ES_API_KEY = os.getenv("ELASTIC_API_KEY")
ES_INDEX   = "boundly_h1b"

es = Elasticsearch(
    [ES_HOST],
    api_key=ES_API_KEY,
    verify_certs=True,
    timeout=60,
    max_retries=3,
    retry_on_timeout=True,
)

# ─── AzureOpenAI client for embeddings ───────────────────────
client = AzureOpenAI(
    azure_endpoint=os.getenv("OPENAI_API_BASE"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_version=os.getenv("OPENAI_API_VERSION"),
)

EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")

def embed_query(text: str) -> np.ndarray:
    """Get a single embedding vector via AzureOpenAI."""
    resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
    vec = np.array(resp.data[0].embedding, dtype=np.float32)
    return vec

def search(query: str, k_dense: int = 5, k_sparse: int = 5) -> list[dict]:
    """
    Hybrid search: combine dense-vector (kNN) with BM25 sparse.
    Returns up to k_dense+k_sparse unique docs sorted by combined score.
    Each doc is the _source dict containing 'content','source','dense_vec', etc.
    """
    # 1) Dense vector search
    qvec = embed_query(query).tolist()
    dense_resp = es.search(
        index=ES_INDEX,
        knn={
            "field":        "dense_vec",
            "query_vector": qvec,
            "k":            k_dense,
            "num_candidates": k_dense * 4,
        },
        _source=True,
    )["hits"]["hits"]

    # 2) Sparse BM25
    sparse_resp = es.search(
        index=ES_INDEX,
        query={"match": {"content": query}},
        size=k_sparse,
        _source=True,
    )["hits"]["hits"]

    # 3) Merge & fuse scores
    combined = {}
    for hit in dense_resp + sparse_resp:
        doc_id = hit["_id"]
        score  = hit["_score"]
        src    = hit["_source"]
        if doc_id not in combined:
            combined[doc_id] = {"doc": src, "score": 0.0}
        combined[doc_id]["score"] += score

    # 4) Return top-k by fused score
    top = sorted(combined.values(), key=lambda x: -x["score"])
    top_docs = [entry["doc"] for entry in top[: k_dense + k_sparse]]
    return top_docs
