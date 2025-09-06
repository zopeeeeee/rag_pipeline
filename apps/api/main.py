from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os, shutil, json, uuid
from services.ingest import parse_file
from services.retriever import Retriever
from services.adapter import call_llm_with_fallback
from dotenv import load_dotenv
load_dotenv()  # now .env values are available to os.getenv


app = FastAPI(title="RAG API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

retriever = None

@app.on_event("startup")
def startup():
    global retriever
    retriever = Retriever()

@app.post("/ingest-file")
async def ingest_file(file: UploadFile = File(...)):
    os.makedirs("data/raw", exist_ok=True)
    file_path = f"data/raw/{uuid.uuid4().hex}_{file.filename}"
    with open(file_path,"wb") as f: shutil.copyfileobj(file.file,f)
    chunks = parse_file(file_path)
    preview_path = file_path+".chunks.json"
    with open(preview_path,"w",encoding="utf-8") as p: json.dump(chunks,p)
    return {"status":"saved","path":file_path,"chunks":len(chunks)}

@app.post("/ask")
async def ask(payload: dict):
    q = payload.get("query")
    if not q: raise HTTPException(400,"Missing query")
    hits = retriever.hybrid_search(q,k=6)
    context = "\n".join([f"{i+1}) [source:{h['source']}] {h['text'][:1000]}" for i,h in enumerate(hits)])
    prompt = f"""SYSTEM:
Use only the CONTEXT to answer. Cite sources [1],[2]. If unsupported, say "I don't know".

USER QUERY:
{q}

CONTEXT:
{context}"""
    llm_out = call_llm_with_fallback(prompt)
    return {"query":q,"answer":llm_out["answer"],"provider":llm_out["provider"],"retrieved":[{"source":h["source"],"score":h["score"]} for h in hits]}

@app.get("/")
def root():
    return {"message": "RAG API is running 🚀. Go to /docs to explore."}