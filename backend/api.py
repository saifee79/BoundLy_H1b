import os
import json
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import AzureOpenAI

from .hybrid_retriever import search
from .prompts import build_messages

# ─── Load config ───────────────────────────
load_dotenv()
ENDPOINT    = os.getenv("OPENAI_API_BASE")
API_KEY     = os.getenv("OPENAI_API_KEY")
API_VERSION = os.getenv("OPENAI_API_VERSION")
CHAT_MODEL  = os.getenv("CHAT_MODEL", "gpt-4.1")

# ─── AzureOpenAI client for chat ───────────
client = AzureOpenAI(
    azure_endpoint=ENDPOINT,
    api_key=API_KEY,
    api_version=API_VERSION,
)

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/boundly/query")
async def boundly_query(q: Query):
    # 1. Retrieve context chunks
    chunks = search(q.question)

    # 2. Build messages with few-shots + system prompt
    messages = build_messages(q.question, chunks)

    # 3. Call AzureOpenAI chat
    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.2,
            top_p=0.95,
            response_format={"type": "json_object"},   # <-- updated here
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # 4. Parse & return
    answer    = resp.choices[0].message.content
    citations = [c["source"] for c in chunks]
    return {"answer": answer, "citations": citations}
