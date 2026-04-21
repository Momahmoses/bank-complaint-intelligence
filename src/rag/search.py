"""
Semantic search over historical complaints using sentence embeddings + FAISS.
When an agent opens a ticket, retrieves top-5 similar past complaints + resolutions.
"""
import numpy as np
import pandas as pd
import logging
import os
import json

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class ComplaintSearchIndex:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.embedder = None
        self.index = None
        self.complaints = []
        self.dimension = 384
        self._load_embedder()

    def _load_embedder(self):
        if ST_AVAILABLE:
            try:
                self.embedder = SentenceTransformer(self.model_name)
                self.dimension = self.embedder.get_sentence_embedding_dimension()
                logger.info(f"Loaded sentence transformer: {self.model_name}")
            except Exception as e:
                logger.warning(f"Could not load sentence transformer: {e}")
        else:
            logger.warning("sentence-transformers not installed. Using TF-IDF fallback.")

    def _embed(self, texts: list[str]) -> np.ndarray:
        if self.embedder:
            return self.embedder.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
        from sklearn.feature_extraction.text import TfidfVectorizer
        if not hasattr(self, "_tfidf"):
            self._tfidf = TfidfVectorizer(max_features=384, ngram_range=(1, 2))
            self._tfidf.fit(texts)
        vecs = self._tfidf.transform(texts).toarray().astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return vecs / norms

    def build_index(self, complaints: list[dict]):
        self.complaints = complaints
        texts = [c.get("complaint_text", "") for c in complaints]
        embeddings = self._embed(texts).astype(np.float32)

        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(embeddings.shape[1])
            self.index.add(embeddings)
            logger.info(f"FAISS index built with {len(complaints)} complaints")
        else:
            self._embeddings_matrix = embeddings
            logger.info(f"NumPy fallback index built with {len(complaints)} complaints")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if not self.complaints:
            return []

        query_vec = self._embed([query]).astype(np.float32)

        if FAISS_AVAILABLE and self.index:
            scores, indices = self.index.search(query_vec, top_k)
            results_idx = indices[0]
            result_scores = scores[0]
        elif hasattr(self, "_embeddings_matrix"):
            sims = (self._embeddings_matrix @ query_vec.T).flatten()
            top_idx = np.argsort(sims)[::-1][:top_k]
            results_idx = top_idx
            result_scores = sims[top_idx]
        else:
            return []

        results = []
        for i, score in zip(results_idx, result_scores):
            if i < len(self.complaints) and score > 0.2:
                complaint = dict(self.complaints[i])
                complaint["similarity_score"] = round(float(score), 3)
                results.append(complaint)
        return results

    def save(self, path: str = "models/complaint_index"):
        os.makedirs(path, exist_ok=True)
        with open(f"{path}/complaints.json", "w") as f:
            json.dump(self.complaints, f, indent=2, default=str)
        if FAISS_AVAILABLE and self.index:
            faiss.write_index(self.index, f"{path}/faiss.index")
        elif hasattr(self, "_embeddings_matrix"):
            np.save(f"{path}/embeddings.npy", self._embeddings_matrix)
        logger.info(f"Index saved to {path}")

    @classmethod
    def load(cls, path: str = "models/complaint_index"):
        instance = cls()
        with open(f"{path}/complaints.json") as f:
            instance.complaints = json.load(f)
        if FAISS_AVAILABLE and os.path.exists(f"{path}/faiss.index"):
            instance.index = faiss.read_index(f"{path}/faiss.index")
        elif os.path.exists(f"{path}/embeddings.npy"):
            instance._embeddings_matrix = np.load(f"{path}/embeddings.npy")
        return instance


def build_rag_context(query: str, search_index: ComplaintSearchIndex, max_context: int = 3) -> str:
    similar = search_index.search(query, top_k=max_context)
    if not similar:
        return "No similar past complaints found."

    context_parts = [f"Query complaint: {query}\n\nSimilar past cases:"]
    for i, complaint in enumerate(similar, 1):
        resolution = complaint.get("resolution", "No resolution recorded")
        context_parts.append(
            f"\n[Case {i}] Similarity: {complaint['similarity_score']:.2f}\n"
            f"  Issue: {complaint.get('complaint_text', '')[:200]}\n"
            f"  Resolution: {resolution}"
        )
    return "\n".join(context_parts)
