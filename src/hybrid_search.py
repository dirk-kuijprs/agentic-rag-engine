from rank_bm25 import BM25Okapi
import torch
from sentence_transformers import SentenceTransformer

class HybridSearchEngine:
    def __init__(self, corpus: list[str]):
        self.corpus = corpus
        # Sparse Retrieval (BM25)
        self.tokenized_corpus = [doc.lower().split() for doc in corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        # Dense Retrieval (Embeddings)
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.corpus_embeddings = self.encoder.encode(corpus, convert_to_tensor=True)

    def search(self, query: str, top_k: int = 5):
        # 1. Sparse scores
        tokenized_query = query.lower().split()
        sparse_scores = self.bm25.get_scores(tokenized_query)
        
        # 2. Dense scores
        query_embedding = self.encoder.encode(query, convert_to_tensor=True)
        dense_scores = torch.nn.functional.cosine_similarity(query_embedding, self.corpus_embeddings).cpu().numpy()
        
        # 3. Hybrid fusion (normalized)
        hybrid_scores = (sparse_scores / max(sparse_scores)) + dense_scores
        top_indices = hybrid_scores.argsort()[-top_k:][::-1]
        
        return [self.corpus[i] for i in top_indices]

if __name__ == "__main__":
    docs = ["RAG stands for Retrieval Augmented Generation", "AI agents can plan tasks", "BM25 is a ranking function"]
    engine = HybridSearchEngine(docs)
    print("Hybrid Results:", engine.search("What is RAG?"))
