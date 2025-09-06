import numpy as np
import faiss, pickle
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import os

INDEX_DIR = os.getenv("INDEX_DIR", "data/index")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

class Retriever:
    def __init__(self):
        self.index = faiss.read_index(f"{INDEX_DIR}/index.faiss")
        with open(f"{INDEX_DIR}/metadata.pkl", "rb") as f:
            self.chunks = pickle.load(f)
        self.vectors = np.load(f"{INDEX_DIR}/vectors.npy")
        self.tokenized = [c["text"].split() for c in self.chunks]
        self.bm25 = BM25Okapi(self.tokenized)
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL)

    def vector_search(self, query, k=5):
        qv = self.embed_model.encode([query], convert_to_numpy=True)
        qv = qv / np.linalg.norm(qv, axis=1, keepdims=True)
        D, I = self.index.search(qv.astype("float32"), k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0: continue
            meta = self.chunks[idx]
            results.append({"text": meta["text"], "source": meta["source"], "score": float(score), "idx": idx})
        return results

    def bm25_search(self, query, k=5):
        scores = self.bm25.get_scores(query.split())
        best = np.argsort(scores)[::-1][:k]
        return [{"text": self.chunks[i]["text"], "source": self.chunks[i]["source"], "score": float(scores[i]), "idx": i} for i in best]

    def hybrid_search(self, query, k=5, alpha=0.6):
        vres = self.vector_search(query, k*3)
        bres = self.bm25_search(query, k*3)
        scores = {}
        for r in vres: scores[r["idx"]] = {"vec": r["score"], "meta": r}
        for r in bres:
            if r["idx"] not in scores: scores[r["idx"]] = {"vec": 0.0, "meta": r}
            scores[r["idx"]]["bm25"] = r["score"]
        bm25_vals = [scores[i].get("bm25",0) for i in scores]
        max_b = max(bm25_vals) if bm25_vals else 1
        ranked = []
        for idx,v in scores.items():
            bm = v.get("bm25",0)/(max_b+1e-9)
            final = alpha*v["vec"]+(1-alpha)*bm
            meta=v["meta"]
            ranked.append({"text": meta["text"],"source":meta["source"],"score":final})
        return sorted(ranked,key=lambda x:x["score"],reverse=True)[:k]
