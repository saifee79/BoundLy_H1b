# backend/ingest.py

import os
import glob
import time
import asyncio

from dotenv import load_dotenv
from openai import AzureOpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from elasticsearch import helpers
from hybrid_retriever import es, ES_INDEX

# Load environment (for Azure OpenAI proxy settings & Elastic API key)
load_dotenv()

# AzureOpenAI proxy config (from .env)
client = AzureOpenAI(
    azure_endpoint=os.getenv("OPENAI_API_BASE"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_version=os.getenv("OPENAI_API_VERSION"),
)

# Retry + backoff wrapper for embeddings
@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(6),
    reraise=True,
)
def get_embeddings(batch: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002"),
        input=batch,
    )
    # resp.data is a list of objects with an .embedding field
    return [item.embedding for item in resp.data]

async def main():
    # 1) Chunk all PDFs into in-memory docs list
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    docs = []
    for folder in ("data/cases", "data/regs", "data/articles"):
        for pdf_path in glob.glob(f"{folder}/**/*.pdf", recursive=True):
            for page in PyPDFLoader(pdf_path).load():
                for chunk in splitter.split_text(page.page_content):
                    docs.append({
                        "content": chunk,
                        "source":  os.path.basename(pdf_path),
                    })

    texts = [d["content"] for d in docs]
    batch_size = 20  # throttle to stay within quota

    # 2) Embed in small batches with retry/backoff + pause
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            vectors = get_embeddings(batch)
        except Exception as e:
            print(f"[Batch {i}-{i+batch_size}] embedding error:", e)
            continue

        for doc, vec in zip(docs[i : i + batch_size], vectors):
            doc["dense_vec"] = vec

        time.sleep(1)  # gentle pause between calls

    # 3) Bulk‐index into ES (using the client from hybrid_retriever)
    actions = [
        {"_op_type": "index", "_index": ES_INDEX, **doc}
        for doc in docs
    ]
    success, _ = helpers.bulk(
        es,
        actions,
        request_timeout=60,
        max_retries=2,
        initial_backoff=2,
    )
    print(f"Indexed {success} chunks")

if __name__ == "__main__":
    asyncio.run(main())
