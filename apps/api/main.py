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

    # Retrieve top chunks
    hits = retriever.hybrid_search(q, k=6)
    relevant_hits = hits  # use all hits to maximize context

    # Prepare context
    if relevant_hits:
        context = "\n".join([h['text'][:1000] for h in relevant_hits])
    else:
        context = "No relevant context found."

    # ChatGPT-style prompt (no citations or sources)
    prompt = f"""SYSTEM:
You are a helpful assistant that answers questions in a ChatGPT-like conversational style.
- Use CONTEXT if relevant to the query.
- If CONTEXT is incomplete, answer confidently using your own knowledge.
- Present the answer in clear paragraphs or bullet points, friendly and informative.
- Do NOT include citations or sources.
- Avoid saying "I don't know" unless literally impossible to answer.

USER QUERY:
{q}

CONTEXT:
{context}"""

    # Call LLM
    llm_out = call_llm_with_fallback(prompt)

    # Format response as chat-style message
    chat_response = {
        "role": "assistant",
        "content": llm_out["answer"].strip()
    }

    return {
        "query": q,
        "response": chat_response,
        "provider": llm_out["provider"],
        "retrieved": [{"source": h["source"], "score": h.get("score", None)} for h in relevant_hits],
    }

# ----------------------------
# Root endpoint
# ----------------------------
@app.get("/")
def root():
    return {"message": "RAG API is running 🚀. Go to /docs to explore."}