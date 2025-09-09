from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os, shutil, json, uuid
from services.ingest import parse_file
from services.retriever import Retriever
from services.adapter import call_llm_with_fallback
from dotenv import load_dotenv
import subprocess

# ----------------------------
# Load .env from project root
# ----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # up 3 levels
dotenv_path = os.path.join(BASE_DIR, ".env")
print("Loading .env from:", dotenv_path)
load_dotenv(dotenv_path=dotenv_path)
print("Gemini key loaded:", bool(os.getenv("GEMINI_API_KEY")))
# ----------------------------

app = FastAPI(title="RAG API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # frontend can call
    allow_methods=["*"],
    allow_headers=["*"]
)

retriever = None

# ----------------------------
# Startup: initialize retriever
# ----------------------------

@app.on_event("startup")
def startup():
    global retriever
    retriever = Retriever()
    print("Retriever initialized with chunks:", len(retriever.chunks))


# ----------------------------
# Ingest file endpoint
# ----------------------------
@app.post("/ingest-file")
async def ingest_file(file: UploadFile = File(...)):
    os.makedirs("data/raw", exist_ok=True)
    file_path = f"data/raw/{uuid.uuid4().hex}_{file.filename}"

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Parse file into chunks
    chunks = parse_file(file_path)
    preview_path = file_path + ".chunks.json"
    with open(preview_path, "w", encoding="utf-8") as p:
        json.dump(chunks, p)

    # Reload retriever immediately
    global retriever
    retriever = Retriever()

    return {"status": "indexed", "path": file_path, "chunks": len(chunks)}

# ----------------------------
# Ask endpoint
# ----------------------------
@app.post("/ask")
async def ask(payload: dict):
    q = payload.get("query")
    if not q:
        raise HTTPException(400, "Missing query")

    hits = retriever.hybrid_search(q, k=6)
    relevant_hits = [h for h in hits if h["score"] > 0.2]

    if relevant_hits:
        context = "\n".join(
            [f"{i+1}) [source:{h['source']}] {h['text'][:1000]}" for i, h in enumerate(relevant_hits)]
        )
    else:
        context = "No relevant context found."

    prompt = f"""SYSTEM:
You are a helpful assistant.  
- If the query is related to the provided CONTEXT, answer using the CONTEXT and cite sources like [1],[2].  
- If the query is general and not found in CONTEXT, answer from your own knowledge.  
- If unsure, say "I don't know".  

USER QUERY:
{q}

CONTEXT (may or may not be relevant):
{context}"""

    llm_out = call_llm_with_fallback(prompt)

    return {
        "query": q,
        "answer": llm_out["answer"],
        "provider": llm_out["provider"],
        "retrieved": [{"source": h["source"], "score": h["score"]} for h in relevant_hits],
    }

# ----------------------------
# Root endpoint
# ----------------------------
@app.get("/")
def root():
    return {"message": "RAG API is running 🚀. Go to /docs to explore."}
