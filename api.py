import json
import faiss
import numpy as np
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

INDEX_PATH = "icd10.faiss"
META_PATH  = "icd10_meta.json"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "deepseek-r1:1.5b"

index = faiss.read_index(INDEX_PATH)
meta = json.load(open(META_PATH, "r", encoding="utf-8"))
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

class Req(BaseModel):
    note: str
    top_k: int = 10

def retrieve(note: str, top_k: int):
    q = embedder.encode([note], normalize_embeddings=True).astype("float32")
    D, I = index.search(q, top_k)
    hits = []
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        d = meta[idx]
        hits.append({"code": d["code"], "title": d["title"], "score": float(score)})
    return hits

def ask_llm(note: str, candidates):
    cand_text = "\n".join([f"{i+1}) {c['code']} — {c['title']}" for i, c in enumerate(candidates)])
    prompt = f"""
You are a medical coding assistant.
Using ONLY the ICD-10 candidates below, select the most appropriate ICD-10 code(s).
Do NOT invent codes outside the list.

Doctor note:
{note}

ICD-10 candidates:
{cand_text}

Return STRICT JSON (no extra text):
{{
  "primary_code": "CODE",
  "confidence": 0.0,
  "reason": "short",
  "alternatives": [{{"code":"CODE","why":"short"}}]
}}
If uncertain, still pick best match but use low confidence.
""".strip()

    r = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=120
    )
    r.raise_for_status()
    return r.json()["response"].strip()

@app.post("/icd-suggest")
def icd_suggest(req: Req):
    candidates = retrieve(req.note, req.top_k)
    llm_json = ask_llm(req.note, candidates)
    return {
        "candidates": candidates,
        "llm": llm_json,
        "disclaimer": "Coding assistance only. Not a medical diagnosis."
    }