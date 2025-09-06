import os, pickle, json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from pathlib import Path

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
INDEX_DIR = Path(os.getenv("INDEX_DIR", "data/index"))
INDEX_DIR.mkdir(parents=True, exist_ok=True)

def build_embeddings(chunks):
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [c["text"] for c in chunks]
    vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return vectors

def normalize_vectors(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    return vectors / norms

def build_faiss_index(vectors, d):
    index = faiss.IndexFlatIP(d)
    index.add(vectors.astype("float32"))
    return index

def save_index(index, chunks, vectors):
    faiss.write_index(index, str(INDEX_DIR / "index.faiss"))
    with open(INDEX_DIR / "metadata.pkl", "wb") as f:
        pickle.dump(chunks, f)
    np.save(INDEX_DIR / "vectors.npy", vectors)

if __name__ == "__main__":
    import sys
    chunks_file = sys.argv[1]
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    vectors = build_embeddings(chunks)
    vectors = normalize_vectors(vectors)
    d = vectors.shape[1]
    index = build_faiss_index(vectors, d)
    save_index(index, chunks, vectors)
    print("Index saved to", INDEX_DIR)
